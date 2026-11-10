"""MiT-B2 and 768-wide SegFormer head, following NVlabs/SegFormer.

Copyright (c) 2021 NVIDIA Corporation. Research use follows the upstream
NVIDIA Source Code License: https://github.com/NVlabs/SegFormer/blob/master/LICENSE
The registry/MMCV/timm wrappers are replaced with equivalent PyTorch ops.
"""
from __future__ import annotations

import torch
import re
from torch import nn
from torch.nn import functional as F


CHANNELS = (64, 128, 320, 512)
DEPTHS = (3, 4, 6, 3)
HEADS = (1, 2, 5, 8)
RATIOS = (8, 4, 2, 1)


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * (torch.rand(shape, device=x.device, dtype=x.dtype) < keep) / keep


class DWConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)

    def forward(self, x, height, width):
        batch, _, channels = x.shape
        return self.dwconv(x.transpose(1, 2).reshape(batch, channels, height, width)).flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels * 4)
        self.dwconv = DWConv(channels * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels * 4, channels)
        self.drop = nn.Dropout(0.)

    def forward(self, x, height, width):
        x = self.drop(self.act(self.dwconv(self.fc1(x), height, width)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, channels, heads, ratio):
        super().__init__()
        self.num_heads = heads
        self.scale = (channels // heads) ** -0.5
        self.q = nn.Linear(channels, channels)
        self.kv = nn.Linear(channels, channels * 2)
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(0.)
        self.sr_ratio = ratio
        if ratio > 1:
            self.sr = nn.Conv2d(channels, channels, ratio, stride=ratio)
            self.norm = nn.LayerNorm(channels)

    def forward(self, x, height, width):
        batch, count, channels = x.shape
        q = self.q(x).reshape(batch, count, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            reduced = self.sr(x.transpose(1, 2).reshape(batch, channels, height, width))
            source = self.norm(reduced.flatten(2).transpose(1, 2))
        else:
            source = x
        kv = self.kv(source).reshape(batch, -1, 2, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        attention = self.attn_drop(((q @ kv[0].transpose(-2, -1)) * self.scale).softmax(-1))
        return self.proj_drop(self.proj((attention @ kv[1]).transpose(1, 2).reshape(batch, count, channels)))


class Block(nn.Module):
    def __init__(self, channels, heads, ratio, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.attn = Attention(channels, heads, ratio)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = Mlp(channels)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, channels, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, channels, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class MiTB2(nn.Module):
    def __init__(self):
        super().__init__()
        rates = torch.linspace(0, .1, sum(DEPTHS)).tolist()
        offset = 0
        for stage, (channels, depth, heads, ratio) in enumerate(zip(CHANNELS, DEPTHS, HEADS, RATIOS), 1):
            setattr(self, f'patch_embed{stage}', OverlapPatchEmbed(3 if stage == 1 else CHANNELS[stage-2], channels,
                                                                  7 if stage == 1 else 3, 4 if stage == 1 else 2))
            setattr(self, f'block{stage}', nn.ModuleList(Block(channels, heads, ratio, rates[offset + i]) for i in range(depth)))
            setattr(self, f'norm{stage}', nn.LayerNorm(channels, eps=1e-6))
            offset += depth
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels // module.groups
            nn.init.normal_(module.weight, std=(2 / fan_out) ** .5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image):
        outputs = []
        for stage in range(1, 5):
            sequence, height, width = getattr(self, f'patch_embed{stage}')(image)
            for block in getattr(self, f'block{stage}'):
                sequence = block(sequence, height, width)
            sequence = getattr(self, f'norm{stage}')(sequence)
            image = sequence.transpose(1, 2).reshape(sequence.shape[0], CHANNELS[stage-1], height, width)
            outputs.append(image)
        return tuple(outputs)


class MiTB2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MiTB2()

    def forward(self, image):
        features = self.backbone(image)
        return features[-1], features


class LinearProjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Linear(channels, 768)

    def forward(self, feature):
        batch, _, height, width = feature.shape
        return self.proj(feature.flatten(2).transpose(1, 2)).transpose(1, 2).reshape(batch, 768, height, width)


class SegFormerB2Decoder(nn.Module):
    def __init__(self, classes=6):
        super().__init__()
        for stage, channels in enumerate(CHANNELS, 1):
            setattr(self, f'linear_c{stage}', LinearProjection(channels))
        self.linear_fuse = nn.Sequential(nn.Conv2d(768 * 4, 768, 1, bias=False),
                                         nn.BatchNorm2d(768), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(.1)
        self.linear_pred = nn.Conv2d(768, classes, 1)

    def fuse_features(self, features):
        if len(features) != 4:
            raise ValueError('SegFormer B2 decoder requires four feature scales')
        target = features[0].shape[-2:]
        projected = [getattr(self, f'linear_c{i}')(features[i-1]) for i in range(4, 0, -1)]
        projected = [F.interpolate(x, size=target, mode='bilinear', align_corners=False)
                     if x.shape[-2:] != target else x for x in projected]
        return self.linear_fuse(torch.cat(projected, dim=1))

    def classify_fused(self, fused, size=None):
        logits = self.linear_pred(self.dropout(fused))
        if getattr(self, 'scout_capture_logits', False):
            self.scout_native_logits = logits.detach()
        return F.interpolate(logits, size=size, mode='bilinear', align_corners=False) if size is not None else logits

    def forward(self, features, size=None):
        return self.classify_fused(self.fuse_features(features), size)


def build_segformer_b2(classes=6):
    return MiTB2Encoder(), SegFormerB2Decoder(classes)


def load_imagenet_b2(encoder, checkpoint):
    """Load every backbone tensor; ignore only a possible ImageNet classifier."""
    state = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state, dict):
        raise ValueError('Invalid MiT-B2 checkpoint')
    state = {key.removeprefix('module.').removeprefix('backbone.'): value
             for key, value in state.items() if torch.is_tensor(value)}
    if any(key.startswith('layers.0.') for key in state):
        converted = {}
        for key, value in state.items():
            match = re.fullmatch(r'layers\.(\d+)\.(\d+)(?:\.(\d+))?\.(.+)', key)
            if not match:
                continue  # A possible ImageNet classifier is intentionally discarded.
            stage, part, block, tail = match.groups()
            stage = int(stage) + 1
            if part == '0':
                converted[f'patch_embed{stage}.' + tail.replace('projection.', 'proj.')] = value
            elif part == '2':
                converted[f'norm{stage}.' + tail] = value
            elif part == '1' and block is not None:
                prefix = f'block{stage}.{block}.'
                if tail.startswith('attn.attn.in_proj_'):
                    kind = tail.removeprefix('attn.attn.in_proj_')
                    channel = value.shape[0] // 3
                    converted[prefix + f'attn.q.{kind}'] = value[:channel]
                    converted[prefix + f'attn.kv.{kind}'] = value[channel:]
                else:
                    tail = tail.replace('attn.attn.out_proj.', 'attn.proj.')
                    tail = tail.replace('ffn.layers.0.', 'mlp.fc1.')
                    tail = tail.replace('ffn.layers.1.', 'mlp.dwconv.dwconv.')
                    tail = tail.replace('ffn.layers.4.', 'mlp.fc2.')
                    if tail.startswith(('mlp.fc1.', 'mlp.fc2.')) and value.ndim == 4:
                        value = value.flatten(1)
                    converted[prefix + tail] = value
        state = converted
    expected = encoder.backbone.state_dict()
    usable = {key: value for key, value in state.items() if key in expected and value.shape == expected[key].shape}
    if set(usable) != set(expected):
        missing = sorted(set(expected) - set(usable))
        raise ValueError(f'MiT-B2 checkpoint incompatible: {len(missing)} missing tensors, e.g. {missing[:5]}')
    encoder.backbone.load_state_dict(usable, strict=True)
