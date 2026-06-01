"""Minimal Mamba-YOLO SS2D blocks for YOLO26 backbone replacement."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv
from .transformer import LayerNorm2d

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except Exception as e:
    selective_scan_fn = None
    selective_scan_ref = None
    MAMBA_IMPORT_ERROR = e
else:
    MAMBA_IMPORT_ERROR = None

__all__ = (
    "SS2D",
    "SS2DDALC",
    "SS2DSFLite",
    "SS1DSFMamba",
    "VSSBlock",
    "VSSBlockDALC",
    "VSSBlockSFLite",
    "SFMambaBlock",
    "C2PSADALCSS2D",
    "C2PSASS2D",
    "C3k2BAPCSS",
    "C3k2DALCSS2D",
    "C3k2SCBSS2D",
    "C3k2SS2D",
    "C3k2SS2DLite",
    "C3k2SS2DSFLite",
    "C3k2SFMamba",
)


class DropPath(nn.Module):
    """Drop paths per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * random_tensor.floor()


def cross_scan(x: torch.Tensor) -> torch.Tensor:
    """Build four scan directions from a 2D feature map."""
    x_hw = x.flatten(2)
    x_wh = x.transpose(2, 3).flatten(2)
    return torch.stack((x_hw, x_wh, x_hw.flip(-1), x_wh.flip(-1)), dim=1)


def cross_merge(ys: torch.Tensor) -> torch.Tensor:
    """Merge four scan directions back to the original 2D layout."""
    b, _, d, h, w = ys.shape
    ys = ys.view(b, 4, d, -1)
    y_hw = ys[:, 0] + ys[:, 2].flip(-1)
    y_wh = ys[:, 1] + ys[:, 3].flip(-1)
    return y_hw + y_wh.view(b, d, w, h).transpose(2, 3).reshape(b, d, -1)


def cross_merge_adaptive(ys: torch.Tensor, direction_weights: torch.Tensor) -> torch.Tensor:
    """Merge four scan directions with learned image-wise direction weights.

    The factor 4 keeps the uniform softmax initialization exactly equivalent to
    the original unweighted SS2D merge, which sums all four aligned directions.
    """
    b, _, d, h, w = ys.shape
    ys = ys.view(b, 4, d, -1)
    aligned = torch.stack(
        (
            ys[:, 0],
            ys[:, 2].flip(-1),
            ys[:, 1].view(b, d, w, h).transpose(2, 3).reshape(b, d, -1),
            ys[:, 3].flip(-1).view(b, d, w, h).transpose(2, 3).reshape(b, d, -1),
        ),
        dim=1,
    )
    weights = direction_weights.to(dtype=aligned.dtype).view(b, 4, 1, 1)
    return 4.0 * (aligned * weights).sum(dim=1)


def pair_swap(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Swap neighboring tokens along one spatial dimension with zero parameters."""
    y = x.movedim(dim, -1)
    size = y.shape[-1]
    pair = (size // 2) * 2
    if pair == 0:
        return x
    head = y[..., :pair].reshape(*y.shape[:-1], pair // 2, 2).flip(-1).reshape(*y.shape[:-1], pair)
    if pair < size:
        head = torch.cat((head, y[..., pair:]), dim=-1)
    return head.movedim(-1, dim)


def auxiliary_patch_swap(x: torch.Tensor) -> torch.Tensor:
    """Lightweight bidirectional compensation via local horizontal and vertical patch swapping."""
    return 0.5 * (pair_swap(x, -1) + pair_swap(x, -2))


def sf_scan(x: torch.Tensor, swap_ratio: float = 0.5) -> torch.Tensor:
    """Build two scan directions augmented with a lightweight patch-swapped context."""
    x_swap = auxiliary_patch_swap(x)
    x_hw = x.flatten(2)
    x_wh = x.transpose(2, 3).flatten(2)
    x_swap_hw = x_swap.flatten(2)
    x_swap_wh = x_swap.transpose(2, 3).flatten(2)
    return torch.stack((x_hw + swap_ratio * x_swap_hw, x_wh + swap_ratio * x_swap_wh), dim=1)


def sf_merge(ys: torch.Tensor) -> torch.Tensor:
    """Merge two scan directions back to the original 2D layout."""
    b, _, d, h, w = ys.shape
    y_hw = ys[:, 0].reshape(b, d, -1)
    y_wh = ys[:, 1].reshape(b, d, w, h).transpose(2, 3).reshape(b, d, -1)
    return y_hw + y_wh


def cross_selective_scan(
    x: torch.Tensor,
    x_proj_weight: torch.Tensor,
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    a_logs: torch.Tensor,
    ds: torch.Tensor,
    out_norm: nn.Module,
    delta_softplus: bool = True,
    force_fp32: bool = False,
) -> torch.Tensor:
    """Apply SS2D selective scan over four spatial directions."""
    if selective_scan_fn is None or selective_scan_ref is None:
        raise ImportError("SS2D requires mamba_ssm with selective scan support.") from MAMBA_IMPORT_ERROR

    b, _, h, w = x.shape
    _, d_state = a_logs.shape
    k, d_inner, dt_rank = dt_projs_weight.shape
    xs = cross_scan(x)
    x_dbl = torch.einsum("bkdl,knd->bknl", xs, x_proj_weight)
    dts, bs, cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
    dts = torch.einsum("bkrl,kdr->bkdl", dts, dt_projs_weight)

    l = h * w
    xs = xs.view(b, -1, l)
    dts = dts.contiguous().view(b, -1, l)
    bs = bs.contiguous()
    cs = cs.contiguous()
    a = -torch.exp(a_logs.float())
    d = ds.float()
    delta_bias = dt_projs_bias.view(-1).float()

    if force_fp32:
        xs = xs.float()
        dts = dts.float()
        bs = bs.float()
        cs = cs.float()

    # `selective_scan_fn` is CUDA-only in common mamba_ssm builds, while model parsing
    # and shape probing in Ultralytics may run on CPU before training starts.
    if xs.is_cuda:
        ys = selective_scan_fn(xs, dts, a, bs, cs, d, delta_bias=delta_bias, delta_softplus=delta_softplus)
    else:
        ys = selective_scan_ref(xs, dts, a, bs, cs, d, delta_bias=delta_bias, delta_softplus=delta_softplus)
    ys = ys.view(b, k, -1, h, w)
    y = cross_merge(ys).transpose(1, 2).contiguous()
    return out_norm(y).view(b, h, w, -1)


def cross_selective_scan_dalc(
    x: torch.Tensor,
    direction_weights: torch.Tensor,
    x_proj_weight: torch.Tensor,
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    a_logs: torch.Tensor,
    ds: torch.Tensor,
    out_norm: nn.Module,
    delta_softplus: bool = True,
    force_fp32: bool = False,
) -> torch.Tensor:
    """Apply SS2D and direction-adaptive local-compensated merging."""
    if selective_scan_fn is None or selective_scan_ref is None:
        raise ImportError("SS2D requires mamba_ssm with selective scan support.") from MAMBA_IMPORT_ERROR

    b, _, h, w = x.shape
    _, d_state = a_logs.shape
    k, d_inner, dt_rank = dt_projs_weight.shape
    xs = cross_scan(x)
    x_dbl = torch.einsum("bkdl,knd->bknl", xs, x_proj_weight)
    dts, bs, cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
    dts = torch.einsum("bkrl,kdr->bkdl", dts, dt_projs_weight)

    l = h * w
    xs = xs.view(b, -1, l)
    dts = dts.contiguous().view(b, -1, l)
    bs = bs.contiguous()
    cs = cs.contiguous()
    a = -torch.exp(a_logs.float())
    d = ds.float()
    delta_bias = dt_projs_bias.view(-1).float()

    if force_fp32:
        xs = xs.float()
        dts = dts.float()
        bs = bs.float()
        cs = cs.float()

    if xs.is_cuda:
        ys = selective_scan_fn(xs, dts, a, bs, cs, d, delta_bias=delta_bias, delta_softplus=delta_softplus)
    else:
        ys = selective_scan_ref(xs, dts, a, bs, cs, d, delta_bias=delta_bias, delta_softplus=delta_softplus)
    ys = ys.view(b, k, -1, h, w)
    y = cross_merge_adaptive(ys, direction_weights).transpose(1, 2).contiguous()
    return out_norm(y).view(b, h, w, -1)


def sf_selective_scan(
    x: torch.Tensor,
    x_proj_weight: torch.Tensor,
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    a_logs: torch.Tensor,
    ds: torch.Tensor,
    out_norm: nn.Module,
    swap_ratio: float = 0.5,
    delta_softplus: bool = True,
    force_fp32: bool = False,
) -> torch.Tensor:
    """Apply a lighter two-direction selective scan with SF-Mamba-inspired patch swapping."""
    if selective_scan_fn is None or selective_scan_ref is None:
        raise ImportError("SS2D requires mamba_ssm with selective scan support.") from MAMBA_IMPORT_ERROR

    b, _, h, w = x.shape
    _, d_state = a_logs.shape
    k, d_inner, dt_rank = dt_projs_weight.shape
    xs = sf_scan(x, swap_ratio=swap_ratio)
    x_dbl = torch.einsum("bkdl,knd->bknl", xs, x_proj_weight)
    dts, bs, cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
    dts = torch.einsum("bkrl,kdr->bkdl", dts, dt_projs_weight)

    l = h * w
    xs = xs.view(b, -1, l)
    dts = dts.contiguous().view(b, -1, l)
    bs = bs.contiguous()
    cs = cs.contiguous()
    a = -torch.exp(a_logs.float())
    d = ds.float()
    delta_bias = dt_projs_bias.view(-1).float()

    if force_fp32:
        xs = xs.float()
        dts = dts.float()
        bs = bs.float()
        cs = cs.float()

    if xs.is_cuda:
        ys = selective_scan_fn(xs, dts, a, bs, cs, d, delta_bias=delta_bias, delta_softplus=delta_softplus)
    else:
        ys = selective_scan_ref(xs, dts, a, bs, cs, d, delta_bias=delta_bias, delta_softplus=delta_softplus)
    ys = ys.view(b, k, -1, h, w)
    y = sf_merge(ys).transpose(1, 2).contiguous()
    return out_norm(y).view(b, h, w, -1)


class SS2D(nn.Module):
    """SS2D core block adapted from Mamba-YOLO."""

    def __init__(
        self,
        d_model: int,
        d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        d_conv: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if selective_scan_fn is None or selective_scan_ref is None:
            raise ImportError("SS2D requires mamba_ssm with selective scan support.") from MAMBA_IMPORT_ERROR

        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else int(d_state)
        self.d_conv = d_conv
        self.disable_z = False
        self.disable_z_act = False
        self.disable_force32 = False
        self.k = 4

        self.out_norm = nn.LayerNorm(d_inner)
        self.in_proj = nn.Conv2d(d_model, d_expand * 2, kernel_size=1, stride=1, bias=bias)
        self.act = nn.GELU()

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                d_expand,
                d_expand,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                groups=d_expand,
                bias=True,
            )
        else:
            self.conv2d = nn.Identity()

        self.ssm_low_rank = d_inner < d_expand
        if self.ssm_low_rank:
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False)

        proj_dim = self.dt_rank + self.d_state * 2
        self.x_proj_weight = nn.Parameter(torch.randn(self.k, proj_dim, d_inner) * (d_inner**-0.5))
        self.dt_projs_weight = nn.Parameter(torch.randn(self.k, d_inner, self.dt_rank) * (self.dt_rank**-0.5))
        self.dt_projs_bias = nn.Parameter(torch.stack([self._init_dt_bias(d_inner) for _ in range(self.k)], dim=0))
        self.a_logs = nn.Parameter(self._init_a_logs(d_inner).repeat(self.k, 1))
        self.ds = nn.Parameter(torch.ones(self.k * d_inner))

        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _init_dt_bias(d_inner: int, dt_min: float = 0.001, dt_max: float = 0.1) -> torch.Tensor:
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=1e-4)
        return dt + torch.log(-torch.expm1(-dt))

    def _init_a_logs(self, d_inner: int) -> torch.Tensor:
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(d_inner, 1).contiguous()
        return torch.log(a)

    def forward_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.ssm_low_rank:
            x = self.in_rank(x)
        y = cross_selective_scan(
            x,
            self.x_proj_weight,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.a_logs,
            self.ds,
            self.out_norm,
            delta_softplus=True,
            force_fp32=self.training and not self.disable_force32,
        )
        if self.ssm_low_rank:
            y = self.out_rank(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, z = self.in_proj(x).chunk(2, dim=1)
        z = self.act(z)
        x = self.act(self.conv2d(x))
        y = self.forward_core(x).permute(0, 3, 1, 2).contiguous()
        y = y * z
        return self.dropout(self.out_proj(y))


class SS2DDALC(SS2D):
    """Direction-Adaptive Local-Compensated SS2D.

    The direction gate is initialized to uniform weights and the local
    compensation scale starts at zero, so this block begins as vanilla SS2D.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        d_conv: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
        dir_hidden_ratio: float = 0.0625,
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            dropout=dropout,
            bias=bias,
        )
        d_inner = int(self.out_norm.normalized_shape[0])
        hidden = max(4, int(d_inner * dir_hidden_ratio))
        self.direction_mlp = nn.Sequential(
            nn.Conv2d(d_inner, hidden, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, 4, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.direction_mlp[-1].weight)
        nn.init.zeros_(self.direction_mlp[-1].bias)

        self.local_comp = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=True)
        nn.init.zeros_(self.local_comp.weight)
        nn.init.zeros_(self.local_comp.bias)
        with torch.no_grad():
            self.local_comp.weight[:, 0, 1, 1] = 1.0
        self.local_scale = nn.Parameter(torch.zeros(1))

    def forward_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.ssm_low_rank:
            x = self.in_rank(x)
        direction_weights = torch.softmax(self.direction_mlp(F.adaptive_avg_pool2d(x, 1)).flatten(1), dim=1)
        y = cross_selective_scan_dalc(
            x,
            direction_weights,
            self.x_proj_weight,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.a_logs,
            self.ds,
            self.out_norm,
            delta_softplus=True,
            force_fp32=self.training and not self.disable_force32,
        )
        local = self.local_comp(x).permute(0, 2, 3, 1).contiguous()
        y = y + self.local_scale.to(dtype=y.dtype) * local
        if self.ssm_low_rank:
            y = self.out_rank(y)
        return y


class SS2DSFLite(SS2D):
    """Lighter SS2D with two scan directions and auxiliary patch swapping inspired by SF-Mamba."""

    def __init__(
        self,
        d_model: int,
        d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        dt_rank: int | str = "auto",
        d_conv: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
        swap_ratio: float = 0.5,
    ):
        super().__init__(
            d_model=d_model,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            dropout=dropout,
            bias=bias,
        )
        self.k = 2
        self.swap_ratio = swap_ratio
        d_inner = self.out_norm.normalized_shape[0]
        proj_dim = self.dt_rank + self.d_state * 2
        self.x_proj_weight = nn.Parameter(torch.randn(self.k, proj_dim, d_inner) * (d_inner**-0.5))
        self.dt_projs_weight = nn.Parameter(torch.randn(self.k, d_inner, self.dt_rank) * (self.dt_rank**-0.5))
        self.dt_projs_bias = nn.Parameter(torch.stack([self._init_dt_bias(d_inner) for _ in range(self.k)], dim=0))
        self.a_logs = nn.Parameter(self._init_a_logs(d_inner).repeat(self.k, 1))
        self.ds = nn.Parameter(torch.ones(self.k * d_inner))

    def forward_core(self, x: torch.Tensor) -> torch.Tensor:
        if self.ssm_low_rank:
            x = self.in_rank(x)
        y = sf_selective_scan(
            x,
            self.x_proj_weight,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.a_logs,
            self.ds,
            self.out_norm,
            swap_ratio=self.swap_ratio,
            delta_softplus=True,
            force_fp32=self.training and not self.disable_force32,
        )
        if self.ssm_low_rank:
            y = self.out_rank(y)
        return y


class RGBlock(nn.Module):
    """Lightweight gated MLP branch used in VSSBlock."""

    def __init__(self, in_features: int, hidden_features: int, act_layer: type[nn.Module] = nn.GELU, drop: float = 0.0):
        super().__init__()
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class LSBlock(nn.Module):
    """Local spatial enhancement block used before SS2D."""

    def __init__(self, channels: int, act_layer: type[nn.Module] = nn.GELU, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.norm = nn.BatchNorm2d(channels)
        self.fc2 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.norm(y)
        y = self.fc2(y)
        y = self.act(y)
        y = self.fc3(y)
        return x + self.drop(y)


class VSSBlock(nn.Module):
    """Mamba-YOLO VSSBlock with SS2D branch."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.proj_conv = (
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
            )
            if in_channels != hidden_dim
            else nn.Identity()
        )
        self.norm = LayerNorm2d(hidden_dim, eps=1e-6)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            d_conv=ssm_conv,
        )
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = LayerNorm2d(hidden_dim, eps=1e-6)
            self.mlp = RGBlock(hidden_dim, int(hidden_dim * mlp_ratio), drop=mlp_drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_conv(x)
        x = x + self.drop_path(self.op(self.norm(self.lsblock(x))))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VSSBlockDALC(VSSBlock):
    """VSS block using DA-LC-SS2D as the state-space operator."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        drop_path: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.proj_conv = (
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
            )
            if in_channels != hidden_dim
            else nn.Identity()
        )
        self.norm = LayerNorm2d(hidden_dim, eps=1e-6)
        self.op = SS2DDALC(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            d_conv=ssm_conv,
        )
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = LayerNorm2d(hidden_dim, eps=1e-6)
            self.mlp = RGBlock(hidden_dim, int(hidden_dim * mlp_ratio), drop=mlp_drop_rate)


class VSSBlockSFLite(VSSBlock):
    """Lighter VSS block using the SF-Mamba-inspired SS2D-SF-Lite operator."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        drop_path: float = 0.0,
        swap_ratio: float = 0.5,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            mlp_ratio=mlp_ratio,
            mlp_drop_rate=mlp_drop_rate,
            drop_path=drop_path,
        )
        self.op = SS2DSFLite(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            d_conv=ssm_conv,
            swap_ratio=swap_ratio,
        )


class SCBSS2DEdgeEnhance(nn.Module):
    """Fixed Sobel edge branch for fine classroom behavior cues."""

    def __init__(self, c: int):
        super().__init__()
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(c, 1, 1, 1), persistent=False)
        self.register_buffer("sobel_y", sobel_y.repeat(c, 1, 1, 1), persistent=False)
        self.fuse = Conv(c, c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = F.conv2d(x, self.sobel_x.to(dtype=x.dtype), padding=1, groups=x.shape[1])
        gy = F.conv2d(x, self.sobel_y.to(dtype=x.dtype), padding=1, groups=x.shape[1])
        return self.fuse((gx.abs() + gy.abs()) * 0.5)


class SCBSS2DBottleneck(nn.Module):
    """SCB local detail branches fused with an optional SS2D global branch."""

    def __init__(
        self,
        c: int,
        shortcut: bool = True,
        edge: bool = False,
        large_kernel: bool = False,
        use_ss2d: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.local = Conv(c, c, 3, 1)
        self.dw3 = DWConv(c, c, 3, 1)
        self.dw5 = DWConv(c, c, 5, 1)
        self.asym = nn.Sequential(Conv(c, c, (1, 3), 1), Conv(c, c, (3, 1), 1))
        self.edge = SCBSS2DEdgeEnhance(c) if edge else nn.Identity()
        self.large = nn.Sequential(DWConv(c, c, (1, 7), 1), DWConv(c, c, (7, 1), 1)) if large_kernel else nn.Identity()
        self.ss2d = (
            VSSBlock(
                c,
                c,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_conv=ssm_conv,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
            )
            if use_ss2d
            else nn.Identity()
        )
        branch_count = 4 + int(edge) + int(large_kernel) + int(use_ss2d)
        self.fuse = Conv(c * branch_count, c, 1, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.local(x), self.dw3(x), self.dw5(x), self.asym(x)]
        if not isinstance(self.edge, nn.Identity):
            branches.append(self.edge(x))
        if not isinstance(self.large, nn.Identity):
            branches.append(self.large(x))
        if not isinstance(self.ss2d, nn.Identity):
            branches.append(self.ss2d(x))
        y = self.fuse(torch.cat(branches, 1))
        return x + y if self.add else y


class BAPCSSBottleneck(nn.Module):
    """Behavior-aware partial-channel SS2D bottleneck.

    A local BAC-style branch preserves fine behavior cues, while only a subset
    of channels enters SS2D. The local branch can also generate a behavior gate
    that modulates the SS2D branch before selective scanning.
    """

    def __init__(
        self,
        c: int,
        shortcut: bool = True,
        ss2d_ratio: float = 0.5,
        behavior_gate: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        if not 0.0 < ss2d_ratio < 1.0:
            raise ValueError(f"ss2d_ratio must be in (0, 1), got {ss2d_ratio}.")

        self.c_ss2d = max(1, min(c - 1, int(round(c * ss2d_ratio))))
        self.c_local = c - self.c_ss2d
        self.behavior_gate = bool(behavior_gate)

        self.local = Conv(self.c_local, self.c_local, 3, 1)
        self.dw3 = DWConv(self.c_local, self.c_local, 3, 1)
        self.dw5 = DWConv(self.c_local, self.c_local, 5, 1)
        self.asym = nn.Sequential(
            Conv(self.c_local, self.c_local, (1, 3), 1),
            Conv(self.c_local, self.c_local, (3, 1), 1),
        )
        self.local_fuse = Conv(self.c_local * 4, self.c_local, 1, 1)
        self.gate = nn.Sequential(Conv(self.c_local, self.c_ss2d, 1, 1), nn.Sigmoid()) if self.behavior_gate else None
        self.ss2d = VSSBlock(
            self.c_ss2d,
            self.c_ss2d,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )
        self.fuse = Conv(c, c, 1, 1)
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local, x_ss2d = torch.split(x, [self.c_local, self.c_ss2d], dim=1)
        local = self.local_fuse(torch.cat((self.local(x_local), self.dw3(x_local), self.dw5(x_local), self.asym(x_local)), 1))
        if self.gate is not None:
            x_ss2d = x_ss2d * self.gate(local)
        ss2d = self.ss2d(x_ss2d)
        y = self.fuse(torch.cat((local, ss2d), 1))
        return x + y if self.add else y


class C3k2BAPCSS(nn.Module):
    """C3k2-style Behavior-Aware Partial-Channel Selective Scan block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        ss2d_ratio: float = 0.5,
        behavior_gate: bool = True,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        _ = (attn, g)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                BAPCSSBottleneck(
                    self.c,
                    shortcut,
                    ss2d_ratio=ss2d_ratio,
                    behavior_gate=behavior_gate,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                ),
                BAPCSSBottleneck(
                    self.c,
                    shortcut,
                    ss2d_ratio=ss2d_ratio,
                    behavior_gate=behavior_gate,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                ),
            )
            if c3k
            else BAPCSSBottleneck(
                self.c,
                shortcut,
                ss2d_ratio=ss2d_ratio,
                behavior_gate=behavior_gate,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_conv=ssm_conv,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2SCBSS2D(nn.Module):
    """C3k2-style SCB block with edge/multi-scale convolution and SS2D context in one unit."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        edge: bool = False,
        large_kernel: bool = False,
        use_ss2d: bool = True,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        _ = (attn, g)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            nn.Sequential(
                SCBSS2DBottleneck(
                    self.c,
                    shortcut,
                    edge=edge,
                    large_kernel=large_kernel,
                    use_ss2d=use_ss2d,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                ),
                SCBSS2DBottleneck(
                    self.c,
                    shortcut,
                    edge=False,
                    large_kernel=large_kernel,
                    use_ss2d=use_ss2d,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                ),
            )
            if c3k
            else SCBSS2DBottleneck(
                self.c,
                shortcut,
                edge=edge,
                large_kernel=large_kernel,
                use_ss2d=use_ss2d,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_conv=ssm_conv,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2SS2D(nn.Module):
    """Backbone drop-in replacement for C3k2 using stacked VSSBlocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        _ = (c3k, e, attn, g, shortcut)
        self.blocks = nn.Sequential(
            *(
                VSSBlock(
                    c1 if i == 0 else c2,
                    c2,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
                for i in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class C3k2DALCSS2D(C3k2SS2D):
    """C3k2-style block using Direction-Adaptive Local-Compensated SS2D."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        nn.Module.__init__(self)
        _ = (c3k, e, attn, g, shortcut)
        self.blocks = nn.Sequential(
            *(
                VSSBlockDALC(
                    c1 if i == 0 else c2,
                    c2,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
                for i in range(n)
            )
        )


class C2PSASS2D(nn.Module):
    """C2PSA-style global context block with SS2D/VSS blocks replacing PSA attention."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(
                VSSBlock(
                    self.c,
                    self.c,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
                for _ in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSADALCSS2D(C2PSASS2D):
    """C2PSA-style global context block with DA-LC-SS2D operators."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        e: float = 0.5,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 2.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=0,
            e=e,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )
        self.m = nn.Sequential(
            *(
                VSSBlockDALC(
                    self.c,
                    self.c,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
                for _ in range(n)
            )
        )


class C3k2SS2DLite(C3k2SS2D):
    """Lighter SS2D block that keeps the same design but reduces scan and MLP width."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 1.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__(
            c1=c1,
            c2=c2,
            n=n,
            c3k=c3k,
            e=e,
            attn=attn,
            g=g,
            shortcut=shortcut,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            ssm_dt_rank=ssm_dt_rank,
            ssm_conv=ssm_conv,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
        )


class C3k2SS2DSFLite(C3k2SS2D):
    """SF-Mamba-inspired lighter SS2D replacement using two-direction scan plus patch swapping."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 1.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
        swap_ratio: float = 0.5,
    ):
        nn.Module.__init__(self)
        _ = (c3k, e, attn, g, shortcut)
        self.blocks = nn.Sequential(
            *(
                VSSBlockSFLite(
                    c1 if i == 0 else c2,
                    c2,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                    swap_ratio=swap_ratio,
                )
                for i in range(n)
            )
        )


def seq_selective_scan(
    x: torch.Tensor,
    x_proj_weight: torch.Tensor,
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    a_logs: torch.Tensor,
    ds: torch.Tensor,
    out_norm: nn.Module,
    delta_softplus: bool = True,
    force_fp32: bool = False,
) -> torch.Tensor:
    """Apply a single unidirectional selective scan on a sequence."""
    if selective_scan_fn is None or selective_scan_ref is None:
        raise ImportError("SS1DSFMamba requires mamba_ssm with selective scan support.") from MAMBA_IMPORT_ERROR

    b, d_inner, l = x.shape
    _, d_state = a_logs.shape
    dt_rank = dt_projs_weight.shape[-1]

    x_dbl = torch.einsum("bdl,nd->bnl", x, x_proj_weight)
    dts, bs, cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=1)
    dts = torch.einsum("brl,dr->bdl", dts, dt_projs_weight)

    a = -torch.exp(a_logs.float())
    d = ds.float()
    delta_bias = dt_projs_bias.float()

    if force_fp32:
        x = x.float()
        dts = dts.float()
        bs = bs.float()
        cs = cs.float()

    if x.is_cuda:
        y = selective_scan_fn(x, dts, a, bs, cs, d, delta_bias=delta_bias, delta_softplus=delta_softplus)
    else:
        y = selective_scan_ref(x, dts, a, bs, cs, d, delta_bias=delta_bias, delta_softplus=delta_softplus)
    return out_norm(y.transpose(1, 2)).contiguous()


class FFNSeq(nn.Module):
    """Lightweight sequence FFN for SFMamba blocks."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SS1DSFMamba(nn.Module):
    """Best-effort SF-Mamba-style unidirectional sequence Mamba with auxiliary token swapping."""

    def __init__(
        self,
        d_model: int,
        d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 1.0,
        dt_rank: int | str = "auto",
        d_conv: int = 3,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        if selective_scan_fn is None or selective_scan_ref is None:
            raise ImportError("SS1DSFMamba requires mamba_ssm with selective scan support.") from MAMBA_IMPORT_ERROR

        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else int(d_state)
        self.disable_force32 = False
        self.out_norm = nn.LayerNorm(d_inner)
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias)
        self.act = nn.GELU()
        self.conv1d = (
            nn.Conv1d(d_expand, d_expand, kernel_size=d_conv, padding=(d_conv - 1) // 2, groups=d_expand, bias=True)
            if d_conv > 1
            else nn.Identity()
        )
        self.ssm_low_rank = d_inner < d_expand
        if self.ssm_low_rank:
            self.in_rank = nn.Conv1d(d_expand, d_inner, kernel_size=1, bias=False)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False)

        proj_dim = self.dt_rank + self.d_state * 2
        self.x_proj_weight = nn.Parameter(torch.randn(proj_dim, d_inner) * (d_inner**-0.5))
        self.dt_projs_weight = nn.Parameter(torch.randn(d_inner, self.dt_rank) * (self.dt_rank**-0.5))
        self.dt_projs_bias = nn.Parameter(SS2D._init_dt_bias(d_inner))
        self.a_logs = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1, dtype=torch.float32)).unsqueeze(0).repeat(d_inner, 1))
        self.ds = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, z = self.in_proj(x).chunk(2, dim=-1)
        z = self.act(z)
        x = self.act(self.conv1d(x.transpose(1, 2)))
        if self.ssm_low_rank:
            x = self.in_rank(x)
        y = seq_selective_scan(
            x,
            self.x_proj_weight,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.a_logs,
            self.ds,
            self.out_norm,
            delta_softplus=True,
            force_fp32=self.training and not self.disable_force32,
        )
        if self.ssm_low_rank:
            y = self.out_rank(y)
        y = y * z
        return self.dropout(self.out_proj(y))


class SFMambaBlock(nn.Module):
    """Best-effort reproduction of the SF-Mamba data-flow idea for vision backbones."""

    def __init__(
        self,
        channels: int,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 1.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 2.0,
        mlp_drop_rate: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.lsblock = LSBlock(channels)
        self.norm = nn.LayerNorm(channels)
        self.op = SS1DSFMamba(
            d_model=channels,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            d_conv=ssm_conv,
        )
        self.drop_path = DropPath(drop_path)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = nn.LayerNorm(channels)
            self.mlp = FFNSeq(channels, int(channels * mlp_ratio), drop=mlp_drop_rate)

    def forward(
        self, x: torch.Tensor, aux_head: torch.Tensor | None = None, aux_tail: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = self.lsblock(x)
        seq = x.flatten(2).transpose(1, 2).contiguous()
        pooled = seq.mean(dim=1, keepdim=True)
        aux_head = pooled if aux_head is None else aux_head
        aux_tail = pooled if aux_tail is None else aux_tail
        seq_with_aux = torch.cat((aux_head, seq, aux_tail), dim=1)
        seq_with_aux = seq_with_aux + self.drop_path(self.op(self.norm(seq_with_aux)))
        if self.mlp_branch:
            seq_with_aux = seq_with_aux + self.drop_path(self.mlp(self.norm2(seq_with_aux)))
        aux_head_out = seq_with_aux[:, :1]
        seq_out = seq_with_aux[:, 1:-1]
        aux_tail_out = seq_with_aux[:, -1:]
        x = seq_out.transpose(1, 2).reshape(b, c, h, w).contiguous()
        return x, aux_tail_out, aux_head_out


class C3k2SFMamba(nn.Module):
    """Best-effort SF-Mamba stage wrapper with auxiliary token swapping across stacked blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        ssm_d_state: int | str = "auto",
        ssm_ratio: float = 2.0,
        ssm_rank_ratio: float = 1.0,
        ssm_dt_rank: int | str = "auto",
        ssm_conv: int = 3,
        mlp_ratio: float = 2.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        _ = (c3k, e, attn, g, shortcut)
        self.proj_conv = (
            nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(c2),
                nn.SiLU(),
            )
            if c1 != c2
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [
                SFMambaBlock(
                    c2,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    ssm_rank_ratio=ssm_rank_ratio,
                    ssm_dt_rank=ssm_dt_rank,
                    ssm_conv=ssm_conv,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path,
                )
                for _ in range(n)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj_conv(x)
        aux_head = None
        aux_tail = None
        for block in self.blocks:
            x, aux_head, aux_tail = block(x, aux_head, aux_tail)
        return x
