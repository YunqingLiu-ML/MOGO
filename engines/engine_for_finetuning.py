import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from typing import Iterable, Optional
from scipy.optimize import linear_sum_assignment
import math
import sys
from timm.utils import accuracy, ModelEma
from datasets.mixup import Mixup
import os
import time
import numpy as np
from typing import Iterable, Optional
import utils
from scipy.special import softmax

from torchmetrics.detection.mean_ap import MeanAveragePrecision

from sklearn.metrics import precision_recall_fscore_support
from engines.util.box_ops import box_iou

CLASS_NAME_TO_IDX = {
    'brush_hair': 0,
    'catch': 1,
    'clap': 2,
    'climb_stairs': 3,
    'golf': 4,
    'jump': 5,
    'kick_ball': 6,
    'pick': 7,
    'pour': 8,
    'pullup': 9,
    'push': 10,
    'run': 11,
    'shoot_ball': 12,
    'shoot_bow': 13,
    'shoot_gun': 14,
    'sit': 15,
    'stand': 16,
    'swing_baseball': 17,
    'throw': 18,
    'walk': 19,
    'wave': 20
}

@torch.no_grad()
def final_test(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    amp_autocast,
    num_classes: int,
):
    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test'

    metric_map = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True
    ).to(device)

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = samples.to(device, non_blocking=True)

        target_list = []
        for t in targets:
            tgt_dict = {}
            tgt_dict['boxes'] = t['boxes'].to(device)
            tgt_dict['labels'] = t['labels'].to(device)
            target_list.append(tgt_dict)

        with amp_autocast():
            outputs, _ = model(samples)

        pred_list = []
        bs = outputs['pred_logits'].shape[0]
        for i in range(bs):
            logit_i = outputs['pred_logits'][i]
            box_i = outputs['pred_boxes'][i]

            prob_i = logit_i.softmax(-1)[:, :-1]
            scores_i, labels_i = prob_i.max(dim=-1)

            keep = (scores_i > 0.0)
            scores_i = scores_i[keep]
            labels_i = labels_i[keep]
            box_i = box_i[keep]

            pred_dict = {
                "boxes":  box_i,       # (N_pred, 4)
                "scores": scores_i,    # (N_pred,)
                "labels": labels_i,    # (N_pred,)
            }
            pred_list.append(pred_dict)

        metric_map.update(pred_list, target_list)

    results = metric_map.compute()

    mAP_50_95 = float(results["map"])
    mAP_50 = float(results["map_50"])
    mAP_75 = float(results["map_75"])
    map_per_class = results.get("map_per_class", None)

    print(f" * [Overall] mAP(0.5:0.95)={mAP_50_95:.3f}  mAP@0.5={mAP_50:.3f}  mAP@0.75={mAP_75:.3f}")

    if map_per_class is not None:
        print(" * [Per Class] AP(0.5:0.95):")
        for class_id, ap_value in enumerate(map_per_class):
            print(f"    Class {class_id}: AP={ap_value:.3f}")

    metric_logger.meters['mAP_50_95'].update(mAP_50_95)
    metric_logger.meters['mAP_50'].update(mAP_50)
    metric_logger.meters['mAP_75'].update(mAP_75)
    metric_logger.synchronize_between_processes()

    print('   Final metric: [mAP_50_95={:.3f} | mAP_50={:.3f} | mAP_75={:.3f}]'.format(
        mAP_50_95, mAP_50, mAP_75
    ))

    test_stats = {
        'mAP_50_95': mAP_50_95,
        'mAP_50': mAP_50,
        'mAP_75': mAP_75,
    }

    if map_per_class is not None:
        test_stats['map_per_class'] = map_per_class.cpu().numpy().tolist()

    return test_stats 


def get_token_coords(num_patches_h, num_patches_w, num_frames=8, device='cpu'):
    x_coords = (torch.arange(num_patches_w, device=device) + 0.5) / num_patches_w
    y_coords = (torch.arange(num_patches_h, device=device) + 0.5) / num_patches_h

    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    token_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
    token_coords = token_coords.unsqueeze(0).repeat(num_frames, 1, 1)
    token_coords = token_coords.reshape(-1, 2)
    return token_coords

def compute_target_importance(token_coords, targets):
    B = len(targets)
    num_tokens = token_coords.shape[0]
    tokens_xy = token_coords.unsqueeze(0).repeat(B, 1, 1)
    target_importance = torch.zeros(B, num_tokens, device=token_coords.device)

    for b in range(B):
        boxes = targets[b]['boxes']
        if boxes.numel() == 0:
            continue
        tokens_xy_b = tokens_xy[b].unsqueeze(1)
        boxes_b = boxes[:, [0, 1, 2, 3]]
        boxes_b = boxes_b.unsqueeze(0)

        x_in_box = (tokens_xy_b[:, :, 0] >= boxes_b[:, :, 0]) & (tokens_xy_b[:, :, 0] <= boxes_b[:, :, 2])
        y_in_box = (tokens_xy_b[:, :, 1] >= boxes_b[:, :, 1]) & (tokens_xy_b[:, :, 1] <= boxes_b[:, :, 3])

        tokens_in_box = x_in_box & y_in_box
        tokens_in_any_box = tokens_in_box.any(dim=1).float()
        target_importance[b] = tokens_in_any_box
    return target_importance

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, amp_autocast, max_norm: float = 1.0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=1, no_amp=False, bf16=False):
    model.train()
    criterion.train()
    print_freq = 10
    start_time = time.time()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = f'Epoch: [{epoch}]'

    optimizer.zero_grad()

    token_coords = get_token_coords(14, 14, num_frames=8, device=device)
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if num_training_steps_per_epoch is not None:
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step
        else:
            it = data_iter_step

        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    if "lr_scale" in param_group:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    else:
                        param_group["lr"] = lr_schedule_values[it]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        for target in targets:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)

        grad_norm = None
        if loss_scaler is not None:
            with amp_autocast():
                outputs, importance_scores = model(samples)
                target_importance = compute_target_importance(token_coords, targets)
                importance_loss = F.binary_cross_entropy_with_logits(importance_scores, target_importance)

                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                losses = losses + 0.5 * importance_loss

            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.")
                sys.exit(1)

            losses /= update_freq
            loss_scaler.scale(losses).backward()

            if (data_iter_step + 1) % update_freq == 0:
                loss_scaler.unscale_(optimizer)
                if max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                else:
                    grad_norm = None
                loss_scaler.step(optimizer)
                loss_scaler.update()
                optimizer.zero_grad()

                if model_ema is not None:
                    model_ema.update(model)
        else:
            outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.")
                sys.exit(1)

            losses /= update_freq
            losses.backward()

            if (data_iter_step + 1) % update_freq == 0:
                if max_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                else:
                    grad_norm = None
                optimizer.step()
                optimizer.zero_grad()

                if model_ema is not None:
                    model_ema.update(model)

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        if grad_norm is not None:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            if grad_norm is not None:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
