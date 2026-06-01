# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""DySample dynamic upsampling module.

Adapted from "Learning to Upsample by Learning to Sample" (ICCV 2023).
The implementation is pure PyTorch and can replace nearest-neighbor upsample
layers in YOLO necks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normal_init(module, mean=0.0, std=1.0, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _constant_init(module, val, bias=0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """Lightweight dynamic upsampling by learned point sampling."""

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        if style not in {"lp", "pl"}:
            raise ValueError(f"DySample style must be 'lp' or 'pl', got {style!r}.")
        if style == "pl":
            if in_channels < scale**2 or in_channels % scale**2 != 0:
                raise ValueError("DySample style='pl' requires in_channels divisible by scale**2.")
        if in_channels < groups or in_channels % groups != 0:
            raise ValueError("DySample requires in_channels to be divisible by groups.")

        offset_channels_in = in_channels // scale**2 if style == "pl" else in_channels
        offset_channels_out = 2 * groups if style == "pl" else 2 * groups * scale**2
        self.offset = nn.Conv2d(offset_channels_in, offset_channels_out, 1)
        _normal_init(self.offset, std=0.001)

        if dyscope:
            self.scope = nn.Conv2d(offset_channels_in, offset_channels_out, 1, bias=False)
            _constant_init(self.scope, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid(h, h, indexing="ij")).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(b, -1, h, w), self.scale)
        coords = coords.reshape(b, 2, -1, self.scale * h, self.scale * w).permute(0, 2, 3, 4, 1)
        coords = coords.contiguous().flatten(0, 1)
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        return self.forward_pl(x) if self.style == "pl" else self.forward_lp(x)


class ZDDySample(DySample):
    """Zero-initialized detail-gated DySample.

    A zero-initialized residual gate keeps the module identical to DySample at
    the start of training, then lets the model learn a tiny local-detail bias.
    """

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False, gate_gain=0.05, gate_floor=1.0):
        super().__init__(in_channels, scale, style, groups, dyscope)
        self.gate_gain = float(gate_gain)
        self.gate_floor = float(gate_floor)
        self.gate_strength = nn.Parameter(torch.zeros(1))

    def _detail_delta(self, x, out_channels):
        detail = (x - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)).abs().mean(1, keepdim=True)
        detail = detail / (detail.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return torch.tanh(detail - 1.0).repeat(1, out_channels, 1, 1)

    def _modulate(self, x, residual):
        delta = self._detail_delta(x, residual.shape[1])
        strength = self.gate_gain * torch.tanh(self.gate_strength)
        return residual * (self.gate_floor + strength * delta)

    def forward_lp(self, x):
        if hasattr(self, "scope"):
            residual = self.offset(x) * self.scope(x).sigmoid() * 0.5
        else:
            residual = self.offset(x) * 0.25
        residual = self._modulate(x, residual)
        return self.sample(x, residual + self.init_pos)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, "scope"):
            residual = self.offset(x_) * self.scope(x_).sigmoid()
        else:
            residual = self.offset(x_)
        residual = self._modulate(x_, residual)
        scale = 0.5 if hasattr(self, "scope") else 0.25
        return self.sample(x, F.pixel_unshuffle(residual, self.scale) * scale + self.init_pos)


class SCBDySample(ZDDySample):
    """Backward-compatible name for the conservative classroom DySample variant."""
