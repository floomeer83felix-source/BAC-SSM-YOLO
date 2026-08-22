"""Paper-faithful reimplementation blocks for classroom-specific baselines.

These are reimplementations from the published method descriptions of
PLA-YOLO11n and WAD-YOLOv8n. They are not claimed to be the authors' official
source code. Code-level assumptions are documented with the experiment files.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import C2f
from .conv import Conv


class PartialConv3(nn.Module):
    """Apply a 3x3 spatial convolution to one channel partition."""

    def __init__(self, c: int, n_div: int = 4):
        super().__init__()
        if c < n_div:
            raise ValueError(f"channels={c} must be >= n_div={n_div}")
        self.dim_conv = max(c // n_div, 1)
        self.dim_untouched = c - self.dim_conv
        self.partial_conv = nn.Sequential(
            nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dim_conv),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        return torch.cat((self.partial_conv(x1), x2), dim=1)


class PConvBottleneck(nn.Module):
    """PConv bottleneck used inside C3k2PConv."""

    def __init__(self, c: int, shortcut: bool = True, n_div: int = 4):
        super().__init__()
        self.spatial = PartialConv3(c, n_div=n_div)
        self.pw = Conv(c, c, 1, 1)
        self.add = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.spatial(x))
        return x + y if self.add else y


class C3k2PConv(C2f):
    """C3k2-style CSP block using PartialConv3 bottlenecks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        n_div: int = 4,
        shortcut: bool = True,
        g: int = 1,
    ):
        _ = (c3k, g)
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=1, e=e)
        self.constructor_args = {"c1": c1, "c2": c2, "n": n, "c3k": c3k, "e": e, "n_div": n_div}
        self.m = nn.ModuleList(PConvBottleneck(self.c, shortcut=shortcut, n_div=n_div) for _ in range(n))


class LSKA(nn.Module):
    """Large separable-kernel spatial attention."""

    def __init__(self, c: int, k: int = 7, dilation: int = 2):
        super().__init__()
        if k % 2 == 0:
            raise ValueError("LSKA kernel size k must be odd")
        self.constructor_args = {"c": c, "k": k, "dilation": dilation}
        p = dilation * (k - 1) // 2
        self.dw_h = nn.Conv2d(
            c, c, kernel_size=(1, k), padding=(0, p), dilation=(1, dilation), groups=c, bias=False
        )
        self.dw_v = nn.Conv2d(
            c, c, kernel_size=(k, 1), padding=(p, 0), dilation=(dilation, 1), groups=c, bias=False
        )
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = torch.sigmoid(self.pw(self.dw_v(self.dw_h(x))))
        return x * attention


class WADCA(nn.Module):
    """Multiscale channel-attention block described in WAD-YOLOv8."""

    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        if c % 2 != 0:
            raise ValueError("WADCA expects an even channel count")
        c_half = c // 2
        self.expand = Conv(c, 2 * c, 3, 1)
        self.b1 = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, padding=1, dilation=1, bias=False), nn.BatchNorm2d(c), nn.SiLU()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(c_half, c_half, 3, 1, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(c_half, c_half, 3, 1, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(c_half),
            nn.SiLU(),
        )
        self.fuse = Conv(2 * c, c, 1, 1)
        hidden = max(c // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.expand(x)
        c = x.shape[1]
        c_half = c // 2
        z1, z2, z3 = torch.split(z, [c, c_half, c_half], dim=1)
        z = self.fuse(torch.cat((self.b1(z1), self.b2(z2), self.b3(z3)), dim=1))
        return x + z * self.gate(z)


class C2fWADCA(C2f):
    """YOLOv8 C2f whose repeated modules are WADCA blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        reduction: int = 16,
    ):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.constructor_args = {
            "c1": c1,
            "c2": c2,
            "n": n,
            "shortcut": shortcut,
            "g": g,
            "e": e,
            "reduction": reduction,
        }
        self.m = nn.ModuleList(WADCA(self.c, reduction=reduction) for _ in range(n))


class TwoDPEMHA(nn.Module):
    """Two-dimensional positional-encoding multi-head attention."""

    def __init__(self, c: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if c % num_heads != 0:
            raise ValueError(f"channels={c} must be divisible by num_heads={num_heads}")
        self.constructor_args = {"c": c, "num_heads": num_heads, "dropout": dropout}
        self.norm1 = nn.LayerNorm(c)
        self.mha = nn.MultiheadAttention(c, num_heads, dropout=dropout, batch_first=True)
        self.conv_h = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.conv_w = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.norm2 = nn.LayerNorm(c)
        self.proj = nn.Linear(c, c)

    @staticmethod
    def _pe2d(h: int, w: int, c: int, device, dtype) -> torch.Tensor:
        pe = torch.zeros((1, c, h, w), device=device, dtype=torch.float32)
        even_idx = torch.arange(0, c, 2, device=device, dtype=torch.float32)
        odd_idx = torch.arange(1, c, 2, device=device, dtype=torch.float32)
        y = torch.arange(h, device=device, dtype=torch.float32).view(1, 1, h, 1)
        x = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, 1, w)
        if len(even_idx):
            f_even = torch.pow(10000.0, -even_idx / max(c, 1)).view(1, -1, 1, 1)
            pe[:, 0::2, :, :] = torch.sin(y * f_even).expand(-1, -1, -1, w)
        if len(odd_idx):
            f_odd = torch.pow(10000.0, -odd_idx / max(c, 1)).view(1, -1, 1, 1)
            pe[:, 1::2, :, :] = torch.cos(x * f_odd).expand(-1, -1, h, -1)
        return pe.to(dtype=dtype).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)
        q = self.norm1(seq)
        main, _ = self.mha(q, q, q, need_weights=False)
        ah = torch.sigmoid(self.conv_h(x))
        aw = torch.sigmoid(self.conv_w(x))
        spatial = (x * ah * aw).flatten(2).transpose(1, 2)
        y = seq + main + spatial + self._pe2d(h, w, c, x.device, x.dtype)
        y = self.proj(self.norm2(y))
        return y.transpose(1, 2).reshape(b, c, h, w).contiguous()
