"""DGLE components: Real-ESRGAN, classwise seed fusion and DDP diffusion.

DDP equations/architecture ported from JiYuanFeng/DDP commit
5b7dd0267fc0f9da3990deaad15ca2ad9428980b (Apache-2.0 / OpenMMLab).
RRDB inference architecture follows Xintao Wang's BasicSR (Apache-2.0).
See references/dgle/LICENSES.md and REPRODUCE_DGLE_B2.md for provenance.
No MMCV extensions are needed: deformable attention uses PyTorch grid_sample.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def percentile_index(count, percentile):
    """Eq.8 interpreted as a one-based nearest rank, including the endpoints."""
    if count < 1 or not 0 <= percentile <= 1:
        raise ValueError('Nonempty class and percentile in [0,1] required')
    return min(count - 1, max(0, math.ceil(percentile * count) - 1))


def fuse_seeds(labels, confidence, sr_labels, sr_confidence, thresholds, sr_thresholds):
    valid = ((labels == sr_labels) & (confidence >= np.asarray(thresholds)[labels]) &
             (sr_confidence >= np.asarray(sr_thresholds)[sr_labels]))
    return np.where(valid, labels, 255).astype(np.uint8)


def sparse_ce(logits, labels):
    # All-ignore crops can occur with sparse seeds. Avoid NaN mean reduction.
    valid = labels != 255
    return F.cross_entropy(logits, labels, ignore_index=255, reduction='sum') / valid.sum().clamp_min(1)


def log_snr(t):
    return -torch.log(((torch.cos((t + .0002) / 1.00025 * math.pi / 2) ** -2) - 1).clamp_min(1e-5))


def alpha_sigma(t):
    level = log_snr(t)
    return level.sigmoid().sqrt(), (-level).sigmoid().sqrt()


def ddim_update(state, clean, now, next_time):
    alpha, sigma = alpha_sigma(now)
    alpha_next, sigma_next = alpha_sigma(next_time)
    reshape = lambda x: x.reshape(-1, 1, 1, 1)
    noise = (state - reshape(alpha) * clean) / reshape(sigma).clamp_min(1e-8)
    return reshape(alpha_next) * clean + reshape(sigma_next) * noise


class FourierTime(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, level):
        frequency = level[:, None] * self.weights[None] * 2 * math.pi
        return torch.cat((level[:, None], frequency.sin(), frequency.cos()), 1)


def sine_position(height, width, channels, device):
    # MMCV SinePositionalEncoding: normalized coordinates, offset=-0.5.
    y, x = torch.meshgrid(torch.arange(height, device=device) + .5,
                          torch.arange(width, device=device) + .5, indexing='ij')
    dim = 10000 ** (2 * (torch.arange(channels // 2, device=device) // 2) / (channels // 2))
    def encode(coordinate, extent):
        v = coordinate[..., None] / (extent + 1e-6) * (2 * math.pi) / dim
        return torch.stack((v[..., 0::2].sin(), v[..., 1::2].cos()), -1).flatten(-2)
    return torch.cat((encode(y, height), encode(x, width)), -1).reshape(1, height * width, channels)


class DeformableAttention(nn.Module):
    """Single-level, 8-head / 4-point DDP attention, equivalent to MMCV CPU op."""
    def __init__(self, channels=256, heads=8, points=4):
        super().__init__()
        self.heads, self.points = heads, points
        self.sampling_offsets = nn.Linear(channels, heads * points * 2)
        self.attention_weights = nn.Linear(channels, heads * points)
        self.value_proj = nn.Linear(channels, channels)
        self.output_proj = nn.Linear(channels, channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.sampling_offsets.weight)
        theta = torch.arange(self.heads) * (2 * math.pi / self.heads)
        grid = torch.stack((theta.cos(), theta.sin()), -1)
        grid = grid / grid.abs().amax(-1, keepdim=True)
        grid = grid[:, None].repeat(1, self.points, 1) * torch.arange(1, self.points + 1)[None, :, None]
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid.flatten())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        for module in (self.value_proj, self.output_proj):
            nn.init.xavier_uniform_(module.weight); nn.init.zeros_(module.bias)

    def forward(self, query, position, height, width):
        batch, count, channels = query.shape
        q = query + position
        offsets = self.sampling_offsets(q).view(batch, count, self.heads, self.points, 2)
        weights = self.attention_weights(q).view(batch, count, self.heads, self.points).softmax(-1)
        y, x = torch.meshgrid(torch.arange(height, device=q.device) + .5,
                              torch.arange(width, device=q.device) + .5, indexing='ij')
        reference = torch.stack((x / width, y / height), -1).reshape(1, count, 1, 1, 2)
        locations = reference + offsets / q.new_tensor([width, height])
        grid = (2 * locations - 1).transpose(1, 2).reshape(batch * self.heads, count, self.points, 2)
        value = self.value_proj(query).view(batch, count, self.heads, channels // self.heads)
        value = value.permute(0, 2, 3, 1).reshape(batch * self.heads, channels // self.heads, height, width)
        # grid_sample CPU and older CUDA versions require matching float32 dtypes.
        sampled = F.grid_sample(value.float(), grid.float(), align_corners=False, padding_mode='zeros')
        weights = weights.transpose(1, 2).reshape(batch * self.heads, 1, count, self.points)
        output = (sampled * weights).sum(-1).view(batch, self.heads, channels // self.heads, count)
        output = output.permute(0, 3, 1, 2).reshape(batch, count, channels)
        return query + self.output_proj(output.to(query.dtype))


class TimeLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = DeformableAttention(channels)
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels))
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(channels * 4, channels * 2))
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def forward(self, query, position, times, height, width):
        query = self.norm1(self.attn(query, position, height, width))
        query = self.norm2(query + self.ffn(query))
        scale, shift = self.time_mlp(times)[:, None].chunk(2, -1)
        return query * (1 + scale) + shift


class DDPSegmenter(nn.Module):
    def __init__(self, encoder, channels=256, layers=6, bit_scale=.01):
        super().__init__()
        self.encoder, self.bit_scale = encoder, bit_scale
        conv_gn = lambda a, b, k: nn.Sequential(nn.Conv2d(a, b, k, padding=k // 2, bias=False), nn.GroupNorm(32, b))
        self.laterals = nn.ModuleList(conv_gn(c, channels, 1) for c in (64, 128, 320, 512))
        self.fpn = nn.ModuleList(conv_gn(channels, channels, 3) for _ in range(4))
        self.merge = conv_gn(channels * 4, channels, 1)
        self.embedding = nn.Embedding(7, channels)  # class6 is the ignore embedding, NOT clutter.
        self.transform = nn.Conv2d(channels * 2, channels, 1)
        self.time_mlp = nn.Sequential(FourierTime(), nn.Linear(17, channels * 4), nn.GELU(),
                                      nn.Linear(channels * 4, channels * 4))
        self.layers = nn.ModuleList(TimeLayer(channels) for _ in range(layers))
        self.readout = nn.Conv2d(channels, 6, 1)
        # Match DeformableHeadWithTime Xavier and radial offset initialization.
        for module in (self.layers, self.readout):
            for parameter in module.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)
        for layer in self.layers:
            layer.attn.reset_parameters()
        # FPN/MultiStageMerging upstream use Xavier convolution initialization.
        for module in (self.laterals, self.fpn, self.merge, self.transform):
            for child in module.modules():
                if isinstance(child, nn.Conv2d):
                    nn.init.xavier_uniform_(child.weight)
                    if child.bias is not None:
                        nn.init.zeros_(child.bias)

    def condition(self, images):
        levels = [projection(x) for projection, x in zip(self.laterals, self.encoder(images)[1])]
        for i in range(3, 0, -1):
            levels[i - 1] = levels[i - 1] + F.interpolate(levels[i], size=levels[i - 1].shape[-2:], mode='nearest')
        levels = [conv(x) for conv, x in zip(self.fpn, levels)]
        return self.merge(torch.cat([F.interpolate(x, size=levels[0].shape[-2:], mode='bilinear', align_corners=False)
                                    for x in levels], 1))

    def encode_labels(self, labels, size):
        down = F.interpolate(labels[:, None].float(), size=size, mode='nearest')[:, 0].long()
        down = torch.where(down == 255, 6, down)
        return (self.embedding(down).permute(0, 3, 1, 2).sigmoid() * 2 - 1) * self.bit_scale

    def denoise(self, condition, state, times):
        features = self.transform(torch.cat((condition, state.to(condition.dtype)), 1))
        batch, channels, height, width = features.shape
        query = features.flatten(2).transpose(1, 2)
        position = sine_position(height, width, channels, features.device)
        time_features = self.time_mlp(log_snr(times))
        for layer in self.layers:
            query = layer(query, position, time_features, height, width)
        return self.readout(query.transpose(1, 2).reshape(batch, channels, height, width))

    def training_loss(self, images, seeds):
        condition = self.condition(images)
        clean = self.encode_labels(seeds, condition.shape[-2:])
        times = torch.rand(images.shape[0], device=images.device) * .999
        alpha, sigma = alpha_sigma(times)
        state = alpha[:, None, None, None] * clean + sigma[:, None, None, None] * torch.randn_like(clean)
        logits = self.denoise(condition, state, times)
        size = seeds.shape[-2:]
        loss = sparse_ce(F.interpolate(logits.float(), size=size, mode='bilinear', align_corners=False), seeds)
        # DGLE Eq.11 specifies only sparse CE on H. Do not silently include the
        # optional 0.4 FCN auxiliary head from the supervised DDP configs.
        return loss, {'diffusion_ce': loss.detach()}

    @torch.no_grad()
    def sample(self, images, seeds, steps=3, time_difference=1, noise=None):
        condition = self.condition(images)
        clean_seeds = self.encode_labels(seeds, condition.shape[-2:])
        valid = F.interpolate((seeds != 255)[:, None].float(), size=condition.shape[-2:], mode='nearest').bool()
        # DDP inference draws a contiguous BCHW map. randn_like on the NHWC
        # embedding's permuted strides changes the seeded noise realization.
        noise = torch.randn(clean_seeds.shape, device=clean_seeds.device, dtype=clean_seeds.dtype) if noise is None else noise
        # Published DGLE PDF (JSTARS2026) Eq.12: seed regions + noise elsewhere.
        state = torch.where(valid, clean_seeds, noise)
        for i in range(steps):
            now = images.new_full((images.shape[0],), 1 - i / steps)
            next_time = images.new_full((images.shape[0],), max(1 - (i + 1 + time_difference) / steps, 0))
            logits = self.denoise(condition, state, now)
            predicted_clean = self.encode_labels(logits.argmax(1), state.shape[-2:])
            state = ddim_update(state, predicted_clean, now, next_time)
        # Neither overwrite the final prediction with seeds nor average steps:
        # DGLE Eq.13 uses the final denoising prediction.
        return F.interpolate(logits.float(), size=images.shape[-2:], mode='bilinear', align_corners=False)


class ResidualDenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(1, 6):
            setattr(self, f'conv{i}', nn.Conv2d(64 + (i - 1) * 32, 32 if i < 5 else 64, 3, padding=1))
        self.lrelu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        values = [x]
        for i in range(1, 5):
            values.append(self.lrelu(getattr(self, f'conv{i}')(torch.cat(values, 1))))
        return x + .2 * self.conv5(torch.cat(values, 1))


class RRDB(nn.Module):
    def __init__(self):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = (ResidualDenseBlock() for _ in range(3))

    def forward(self, x):
        return x + .2 * self.rdb3(self.rdb2(self.rdb1(x)))


class RRDBNet(nn.Module):
    """Strictly compatible with official RealESRGAN_x4plus params_ema."""
    def __init__(self):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1)
        self.body = nn.Sequential(*(RRDB() for _ in range(23)))
        for name in ('conv_body', 'conv_up1', 'conv_up2', 'conv_hr'):
            setattr(self, name, nn.Conv2d(64, 64, 3, padding=1))
        self.conv_last = nn.Conv2d(64, 3, 3, padding=1)
        self.lrelu = nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


@torch.inference_mode()
def enhance(model, rgb, device, tile=128, pad=10):
    """Official tiled x4 inference, then Lanczos resize to original grid (Eq.2)."""
    import cv2
    image = torch.from_numpy(np.array(rgb, copy=True).transpose(2, 0, 1)).float()[None].to(device) / 255
    height, width = image.shape[-2:]
    output = image.new_zeros((1, 3, height * 4, width * 4))
    for y in range(0, height, tile):
        for x in range(0, width, tile):
            y1, x1 = min(height, y + tile), min(width, x + tile)
            py, px = max(0, y - pad), max(0, x - pad)
            py1, px1 = min(height, y1 + pad), min(width, x1 + pad)
            result = model(image[..., py:py1, px:px1])
            output[..., y * 4:y1 * 4, x * 4:x1 * 4] = result[..., (y - py) * 4:(y1 - py) * 4,
                                                                         (x - px) * 4:(x1 - px) * 4]
    output = output[0].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
    output = (output * 255).round().astype(np.uint8)
    return cv2.resize(output, (width, height), interpolation=cv2.INTER_LANCZOS4)
