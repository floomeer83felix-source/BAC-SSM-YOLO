# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""EfficientViM blocks for YOLO experiments.

Adapted from EfficientViM: Efficient Vision Mamba with Hidden State Mixer based
State Space Duality. This implementation keeps the module self-contained and
uses the original input spatial size instead of assuming square feature maps.
"""

import torch
import torch.nn as nn


class LayerNorm1D(nn.Module):
    """Channel-wise LayerNorm for tensors shaped as B,C,L."""

    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class ConvLayer2D(nn.Module):
    """Small Conv2d wrapper used by EfficientViM."""

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm=nn.BatchNorm2d,
        act_layer=nn.ReLU,
        bn_weight_init=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = norm(out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    """Small Conv1d wrapper used by EfficientViM."""

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm=nn.BatchNorm1d,
        act_layer=nn.ReLU,
        bn_weight_init=1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.norm = norm(out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm:
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class EfficientViMFFN(nn.Module):
    """Pointwise FFN used inside EfficientViMBlock."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = ConvLayer2D(dim, hidden_dim, 1)
        self.fc2 = ConvLayer2D(hidden_dim, dim, 1, act_layer=None, bn_weight_init=0)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class HSMSSD(nn.Module):
    """Hidden State Mixer based State Space Duality block."""

    def __init__(self, d_model, ssd_expand=1, a_init_range=(1, 16), state_dim=32):
        super().__init__()
        self.d_inner = int(ssd_expand * d_model)
        self.state_dim = state_dim

        self.bcdt_proj = ConvLayer1D(d_model, 3 * state_dim, 1, norm=None, act_layer=None)
        self.dw = ConvLayer2D(3 * state_dim, 3 * state_dim, 3, 1, 1, groups=3 * state_dim, norm=None, act_layer=None)
        self.hz_proj = ConvLayer1D(d_model, 2 * self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        self.a = nn.Parameter(torch.empty(state_dim, dtype=torch.float32).uniform_(*a_init_range))
        self.act = nn.SiLU()
        self.d = nn.Parameter(torch.ones(1))
        self.d._no_weight_decay = True

    def forward(self, x, h, w):
        b, _, length = x.shape
        if length != h * w:
            raise ValueError(f"EfficientViM expected length {h * w}, got {length}.")

        bcdt = self.dw(self.bcdt_proj(x).view(b, -1, h, w)).flatten(2)
        # Split returns multiple views. Clone them to avoid autograd errors if a
        # downstream op performs an in-place modification during backward.
        b_state, c_state, dt = [t.clone() for t in torch.split(bcdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)]
        a = (dt + self.a.view(1, -1, 1)).softmax(-1)

        hidden = x @ (a * b_state).transpose(-2, -1)
        hidden, gate = [t.clone() for t in torch.split(self.hz_proj(hidden), [self.d_inner, self.d_inner], dim=1)]
        hidden = self.out_proj(hidden * self.act(gate) + hidden * self.d)
        y = hidden @ c_state
        return y.view(b, -1, h, w).contiguous()


class EfficientViMBlock(nn.Module):
    """EfficientViM block that preserves the input channel count and spatial size."""

    def __init__(self, dim, mlp_ratio=2.0, ssd_expand=1, state_dim=32):
        super().__init__()
        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)
        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.ffn = EfficientViMFFN(dim, int(dim * mlp_ratio))
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)

    def forward(self, x):
        _, _, h, w = x.shape
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)
        shortcut = x
        x = self.mixer(self.norm(x.flatten(2)), h, w)
        x = (1 - alpha[1]) * shortcut + alpha[1] * x
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        return x
