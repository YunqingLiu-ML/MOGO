import os
import io
import random
import numpy as np
import torch
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from .random_erasing import RandomErasing
from .video_transforms import (
    Compose, Resize, CenterCrop, Normalize,
    create_random_augment, random_short_side_scale_jitter,
    random_crop, random_resized_crop_with_shift, random_resized_crop,
    horizontal_flip, random_short_side_scale_jitter, uniform_crop,
)
from .volume_transforms import ClipToTensor
from . import cv2_transform as cv2_transform

try:
    from petrel_client.client import Client
    has_client = True
except ImportError:
    has_client = False
from collections import defaultdict

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

def tensor_normalize(tensor, mean, std):
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor

def spatial_sampling(
    frames,
    bboxes,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):

    from torchvision.transforms.functional import resize
    original_height, original_width = frames.shape[-2], frames.shape[-1]
    frames = resize(frames, (crop_size, crop_size), antialias=True)

    scale_x = crop_size / original_width
    scale_y = crop_size / original_height

    final_bboxes = []
    for frame_bboxes in bboxes:
        resized_frame_bboxes = []
        for bbox in frame_bboxes:
            xmin, ymin, xmax, ymax = bbox
            resized_bbox = [
                float(xmin * scale_x),
                float(ymin * scale_y),
                float(xmax * scale_x),
                float(ymax * scale_y),
            ]
            resized_frame_bboxes.append(resized_bbox)
        resized_frame_bboxes = torch.tensor(resized_frame_bboxes)
        final_bboxes.append(resized_frame_bboxes)
    return frames, final_bboxes


class VideoClsDataset_sparse(Dataset):
    def __init__(self, video_root, annotation_root, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,
                 args=None):
        self.video_root = video_root
        self.annotation_root = annotation_root
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args

        self.aug = (self.mode == 'train')
        self.rand_erase = bool(getattr(self.args, 'reprob', 0) > 0) if self.aug else False

        self._data_mean = [0.45, 0.45, 0.45]
        self._data_std = [0.225, 0.225, 0.225]
        self._use_bgr = False
        self.random_horizontal_flip = True
        self._split = 'train' if self.mode == 'train' else 'val'
        if self._split == "train":
            self._crop_size = 224
            self._jitter_min_scale = 256
            self._jitter_max_scale = 320
            self._use_color_augmentation = False
            self._pca_jitter_only = True
            self._pca_eigval = [0.225, 0.224, 0.229]
            self._pca_eigvec = [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        else:
            self._crop_size = 224
            self._test_force_flip = False


        indices = [1, 5, 6, 12, 15, 16, 18]
        if self.mode == 'train':
            list_file = "/path/to/JHMDB/data/split/trainlist01.txt"
        else:
            list_file = "/path/to/JHMDB/data/split/testlist01.txt"

        with open(list_file, 'r') as f:
            lines = [x.strip() for x in f if x.strip()]

        self.dataset_samples_frames = []
        self.class_sample_count = defaultdict(int)

        for rel in lines:
            rel_path = rel.split(' ')[0]
            cls_path = rel.split(' ')[1]
            class_idx = CLASS_NAME_TO_IDX[cls_path]

            parts = os.path.join(cls_path, rel_path)
            video_path = os.path.join(self.video_root, parts)
            try:
                vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
                num_frames = len(vr)

                if class_idx in indices:
                   step = 1
                else:
                   step = 4

                for starting_frame_idx in range(0, num_frames - self.clip_len + 1, step):
                    last_frame_idx = starting_frame_idx + self.clip_len - 1
                    video_base_name = os.path.splitext(os.path.basename(video_path))[0]
                    frame_number_str = "{:04d}".format(last_frame_idx)
                    annotation_file_name = f"{video_base_name}_{frame_number_str}.txt"
                    annotation_file_path = os.path.join(self.annotation_root, annotation_file_name)
                    if os.path.exists(annotation_file_path):
                        self.dataset_samples_frames.append((video_path, starting_frame_idx))
                        self.class_sample_count[class_idx] += 1
            except Exception as e:
                print(f"Error reading video {video_path}: {e}")


        print(f"[{self.mode}] total videos: {len(self.dataset_samples_frames)}")
        print(f"Sample per clas: {self.class_sample_count}")

        self.dataset_samples = self.dataset_samples_frames

        self.client = None
        if has_client:
            self.client = Client('~/petreloss.conf')

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, index):
        video_path, _ = self.dataset_samples[index]
        buffer, frame_indices = self.loadvideo_decord(video_path, frame_idx=None)

        if len(buffer) == 0:
            while len(buffer) == 0:
                warnings.warn("video {} not correctly loaded during {} mode".format(video_path, self.mode))
                index = np.random.randint(self.__len__())
                video_path, frame_idx = self.dataset_samples[index]
                buffer, frame_indices = self.loadvideo_decord(video_path, frame_idx)

        annotation_frame_idx = frame_indices[4]
        bboxes, labels = self._load_annotations(video_path, annotation_frame_idx)
        index_list = [index]
        args = self.args
        if args and args.num_sample > 1:
            frame_list = []
            bbox_list = []
            label_list = []
            index_list = []
            for _ in range(args.num_sample):
                new_frames, new_bboxes = self._ava_style_augmentation(buffer, bboxes[0].clone())
                frame_list.append(new_frames)
                bbox_list.append([new_bboxes])
                label_list.append(labels)
                index_list.append(index)

            return frame_list, bbox_list, label_list, index_list
        else:
            buffer, new_bboxes = self._ava_style_augmentation(buffer, bboxes[0].clone())
        return buffer, [new_bboxes], labels, index_list

    def _load_annotations(self, video_path, frame_idx):
        relative_video_path = os.path.relpath(video_path, self.video_root)
        video_dir, video_file = os.path.split(relative_video_path)
        video_base_name, ext = os.path.splitext(video_file)
        annotation_video_dir = self.annotation_root

        bboxes_list = []
        labels_list = []
        frame_number = frame_idx
        frame_number_str = "{:04d}".format(frame_number)
        annotation_file_name = "{}_{}.txt".format(video_base_name, frame_number_str)
        annotation_file_path = os.path.join(annotation_video_dir, annotation_file_name)
        bboxes = []
        labels = []

        if os.path.exists(annotation_file_path):
            with open(annotation_file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        label_name = parts[0]
                        if label_name in CLASS_NAME_TO_IDX:
                            label = CLASS_NAME_TO_IDX[label_name]
                        else:
                            raise ValueError(f"Unknown class label '{label_name}' in {annotation_file_path}")
                        labels.append([label])
                        xmin = int(parts[1])
                        ymin = int(parts[2])
                        xmax = int(parts[3])
                        ymax = int(parts[4])
                        bboxes.append([xmin, ymin, xmax, ymax])
        else:
            # If the annotation file does not exist.
            warnings.warn("Annotation file not found: {}".format(annotation_file_path))
            labels.append([1e6])
            bboxes.append([1e6, 1e6, 1e6, 1e6])

        bboxes = torch.tensor(bboxes, dtype=torch.int32).clone().detach()
        labels = torch.tensor(labels, dtype=torch.int64)
        return [bboxes], [labels]

    def _ava_style_augmentation(self, buffer, bboxes):
        imgs = [buffer[i] for i in range(buffer.shape[0])]
        height, width, _ = imgs[0].shape

        bboxes = cv2_transform.clip_boxes_to_image(bboxes.clone().float().numpy(), height, width)
        boxes = [bboxes]
        if self._split == "train":
            imgs, boxes = cv2_transform.random_short_side_scale_jitter_list(
                imgs,
                min_size=self._jitter_min_scale,
                max_size=self._jitter_max_scale,
                boxes=boxes,
            )
            imgs, boxes = cv2_transform.random_crop_list(
                imgs, self._crop_size, order="HWC", boxes=boxes
            )

            if self.random_horizontal_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    0.5, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "val":
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]
            imgs, boxes = cv2_transform.spatial_shift_crop_list(
                self._crop_size, imgs, 1, boxes=boxes
            )

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        elif self._split == "test":
            imgs = [cv2_transform.scale(self._crop_size, img) for img in imgs]
            boxes = [
                cv2_transform.scale_boxes(
                    self._crop_size, boxes[0], height, width
                )
            ]

            if self._test_force_flip:
                imgs, boxes = cv2_transform.horizontal_flip_list(
                    1, imgs, order="HWC", boxes=boxes
                )
        else:
            raise NotImplementedError(
                "Unsupported split mode {}".format(self._split)
            )

        imgs = [cv2_transform.HWC2CHW(img) for img in imgs]

        imgs = [img / 255.0 for img in imgs]

        imgs = [
            np.ascontiguousarray(
                img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
            ).astype(np.float32)
            for img in imgs
        ]

        if self._split == "train" and self._use_color_augmentation:
            if not self._pca_jitter_only:
                imgs = cv2_transform.color_jitter_list(
                    imgs,
                    img_brightness=0.4,
                    img_contrast=0.4,
                    img_saturation=0.4,
                )

            imgs = cv2_transform.lighting_list(
                imgs,
                alphastd=0.1,
                eigval=np.array(self._pca_eigval).astype(np.float32),
                eigvec=np.array(self._pca_eigvec).astype(np.float32),
            )

        imgs = [
            cv2_transform.color_normalization(
                img,
                np.array(self._data_mean, dtype=np.float32),
                np.array(self._data_std, dtype=np.float32),
            )
            for img in imgs
        ]

        imgs = np.concatenate(
            [np.expand_dims(img, axis=1) for img in imgs], axis=1
        )

        if not self._use_bgr:
            imgs = imgs[::-1, ...]

        imgs = np.ascontiguousarray(imgs)
        imgs = torch.from_numpy(imgs)
        boxes = cv2_transform.clip_boxes_to_image(
            boxes[0], imgs[0].shape[1], imgs[0].shape[2]
        )
        return imgs, boxes

    def loadvideo_decord(self, sample, frame_idx):
        """Load frames from the entire video using Decord"""
        fname = sample
        try:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            num_video_frames = len(vr)
            if num_video_frames < self.clip_len:
                return [],[]
            frame_indices = sorted(random.sample(range(num_video_frames), self.clip_len))
            buffer = vr.get_batch(frame_indices).asnumpy()
            return buffer, frame_indices
        except Exception as e:
            print("video cannot be loaded by decord: ", fname)
            print(e)
            return [], []
