import os
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, List, Dict, Tuple
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.vision_transformer import _cfg, _load_weights
from timm.models.registry import register_model

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError(f"Activation should be relu/gelu, not {activation}.")

def _get_clones(module, N, layer_share=False):
    if layer_share:
        return nn.ModuleList([module for _ in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x

class VisionMambaEncoder(nn.Module):
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            depth=24,
            embed_dim=192,
            channels=3,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None,
            norm_epsilon=1e-5,
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True,
            residual_in_fp32=True,
            bimamba=True,
            kernel_size=1,
            num_frames=8,
            device=None,
            dtype=None,
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num

        self.d_model = self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        self.token_importance_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        # original init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.temporal_pos_embedding, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)
        x = x + self.pos_embed
        # temporal pos, made fixes to the original code.
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)

        x = self.pos_drop(x)

        importance_logits = self.token_importance_mlp(x)
        importance_logits = importance_logits.squeeze(-1)
        importance_scores = torch.sigmoid(importance_logits).unsqueeze(-1)

        x_masked = x * importance_scores
        x = x_masked.reshape(B, -1, C)

        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states, importance_logits  # [B, N, D], [B, T*N]

    def forward(self, x, inference_params=None):
        features, importance_logits = self.forward_features(x, inference_params)
        return features, importance_logits

class MambaDecoderLayer(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, activation="relu", ssm_cfg=None):
        super().__init__()
        self.mamba_block = Mamba(d_model, **(ssm_cfg if ssm_cfg else {}))
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.cross_mamba_block = Mamba(d_model, **(ssm_cfg if ssm_cfg else {}))
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-Mamba block
        tgt2 = self.mamba_block(tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-Mamba block
        memory = memory.transpose(0, 1)  # [N, B, D] -> [B, N, D]
        tgt = tgt.transpose(0, 1)
        combined = torch.cat((tgt, memory), dim=1)
        tgt2 = self.cross_mamba_block(combined)
        tgt2 = tgt2[:, :tgt.size(1), :]
        tgt = self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feed-forward
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        tgt = tgt.transpose(0, 1)
        return tgt

class MambaDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, layer_share=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers, layer_share=layer_share)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, **kwargs):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, **kwargs)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)

class VisionMambaObjectDetector(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_frames=8,
        embed_dim=576,
        encoder_depth=32,
        decoder_depth=1,
        num_queries=100,
        num_classes=22,
        ssm_cfg=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        device=None,
        dtype=None,
        select_topk_ratio: float = 0.4,
        fixed_key_frame_index = 4,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.num_frames = num_frames
        self.select_topk_ratio = select_topk_ratio
        self.fixed_key_frame_index = fixed_key_frame_index

        self.encoder = VisionMambaEncoder(
            img_size=img_size,
            patch_size=patch_size,
            depth=encoder_depth,
            embed_dim=embed_dim,
            channels=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            ssm_cfg=ssm_cfg,
            num_frames=num_frames,
            device=device,
            dtype=dtype,
        )

        self.d_model = embed_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        nn.init.uniform_(self.query_embed.weight)

        decoder_layer = MambaDecoderLayer(d_model=embed_dim, ssm_cfg=ssm_cfg)
        self.decoder = MambaDecoder(decoder_layer, num_layers=decoder_depth, norm=nn.LayerNorm(embed_dim),
                                    return_intermediate=True)

        # Output layers
        self.class_embed = nn.Linear(embed_dim, num_classes + 1)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)


    def forward(self, x):
        batch_size = x.shape[0]

        # Encoder
        memory, importance_logits = self.encoder(x)  # [B, N, D]
        B, N, D = memory.shape
        tokens_per_frame = N // self.num_frames

        start = self.fixed_key_frame_index * tokens_per_frame
        end = (self.fixed_key_frame_index + 1) * tokens_per_frame

        fixed_memory = memory[:, start:end, :]

        selected_memory = []
        for i in range(self.num_frames):
            start_idx = i * tokens_per_frame
            end_idx = (i + 1) * tokens_per_frame
            frame_memory = memory[:, start_idx:end_idx, :]
            frame_importance_logits = importance_logits[:, start_idx:end_idx]

            top_k = max(1, int(tokens_per_frame * self.select_topk_ratio))
            _, indices = torch.topk(frame_importance_logits, top_k, dim=1, largest=True, sorted=False)
            selected_frame_memory = torch.gather(frame_memory, 1, indices.unsqueeze(-1).expand(-1, -1, D))

            selected_memory.append(selected_frame_memory)
        memory1 = torch.cat(selected_memory, dim=1)
        memory = torch.cat((fixed_memory, memory1), dim=1)

        # Prepare input for decoder
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, B, D]
        tgt = torch.zeros_like(query_embed)  # [num_queries, B, D]

        # Decoder
        hs = self.decoder(tgt, memory.transpose(0, 1))  # List of decoder outputs

        outputs_classes = []
        outputs_coords = []
        for output in hs:
            output = output.transpose(0, 1)  # [B, num_queries, D]
            outputs_class = self.class_embed(output)  # [B, num_queries, num_classes]
            outputs_coord = self.bbox_embed(output).sigmoid()  # [B, num_queries, 4]
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)  # [num_layers, B, num_queries, num_classes]
        outputs_coord = torch.stack(outputs_coords)   # [num_layers, B, num_queries, 4]

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        return out, importance_logits

@register_model
def videomamba_middle(pretrained=False, **kwargs):
    model = VisionMambaObjectDetector(
        patch_size=16,
        embed_dim=576,
        **kwargs
    )

    model.default_cfg = _cfg()

    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)

    return model


# Example usage
if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224

    device = torch.device("cuda")
    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    model = videomamba_middle(num_frames=num_frames).to(device)
    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size, device=device))
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)
