# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA, RLE_WEIGHT
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast

from .metrics import bbox_iou, probiou
from .tal import bbox2dist, rbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al.

    Implements the Varifocal Loss function for addressing class imbalance in object detection by focusing on
    hard-to-classify examples and balancing positive/negative samples.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (float): The balancing factor used to address class imbalance.

    References:
        https://arxiv.org/abs/2008.13367
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        """Initialize the VarifocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute varifocal loss between predictions and ground truth."""
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5).

    Implements the Focal Loss function for addressing class imbalance by down-weighting easy examples and focusing on
    hard negatives during training.

    Attributes:
        gamma (float): The focusing parameter that controls how much the loss focuses on hard-to-classify examples.
        alpha (torch.Tensor): The balancing factor used to address class imbalance.
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.25):
        """Initialize FocalLoss class with focusing and balancing parameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)

    def forward(self, pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Calculate focal loss with modulating factors for class imbalance."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if (self.alpha > 0).any():
            self.alpha = self.alpha.to(device=pred.device, dtype=pred.dtype)
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class DFLoss(nn.Module):
    """Criterion class for computing Distribution Focal Loss (DFL)."""

    def __init__(self, reg_max: int = 16) -> None:
        """Initialize the DFL module with regularization maximum."""
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return sum of left and right DFL losses from https://ieeexplore.ieee.org/document/9792391."""
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


def dynamic_interpiou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    alpha_min: float = 0.2,
    alpha_max: float = 0.8,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute a Dynamic InterpIoU-inspired similarity for xyxy boxes.

    This keeps standard IoU behaviour on well-overlapped samples while injecting an interpolation bridge for
    low-IoU pairs so non-overlapping boxes still receive meaningful localization gradients.
    """
    base_iou = bbox_iou(box1, box2, xywh=False, eps=eps).clamp_(0.0, 1.0)
    detached_iou = base_iou.detach()
    interp_alpha = alpha_min + (alpha_max - alpha_min) * detached_iou
    interp_box = interp_alpha * box1 + (1.0 - interp_alpha) * box2
    interp_iou = bbox_iou(interp_box, box2, xywh=False, eps=eps).clamp_(0.0, 1.0)
    interp_weight = 1.0 - detached_iou
    return base_iou * (1.0 - interp_weight) + interp_iou * interp_weight


def focaler_ciou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    lower: float = 0.0,
    upper: float = 0.95,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Focaler-CIoU similarity for xyxy boxes.

    Focaler-IoU remaps the plain IoU into a target interval so training can
    focus more on useful regression samples while keeping CIoU's geometric
    penalty terms.
    """
    if upper <= lower:
        raise ValueError(f"focaler_iou_max must be greater than focaler_iou_min, got {upper} <= {lower}.")
    plain_iou = bbox_iou(box1, box2, xywh=False, eps=eps)
    ciou = bbox_iou(box1, box2, xywh=False, CIoU=True, eps=eps)
    focaler_iou = ((plain_iou - lower) / (upper - lower)).clamp(0.0, 1.0)
    # L_Focaler-CIoU = L_CIoU + IoU - IoU_focaler
    return ciou - plain_iou + focaler_iou


def bbox_nwd_xyxy(box1: torch.Tensor, box2: torch.Tensor, constant: float = 12.8, eps: float = 1e-7) -> torch.Tensor:
    """Compute Normalized Gaussian Wasserstein Distance similarity for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)

    b1_cx, b1_cy = (b1_x1 + b1_x2) * 0.5, (b1_y1 + b1_y2) * 0.5
    b2_cx, b2_cy = (b2_x1 + b2_x2) * 0.5, (b2_y1 + b2_y2) * 0.5
    b1_w, b1_h = (b1_x2 - b1_x1).clamp_min(eps), (b1_y2 - b1_y1).clamp_min(eps)
    b2_w, b2_h = (b2_x2 - b2_x1).clamp_min(eps), (b2_y2 - b2_y1).clamp_min(eps)

    wasserstein = (b1_cx - b2_cx).pow(2) + (b1_cy - b2_cy).pow(2)
    wasserstein += ((b1_w - b2_w).pow(2) + (b1_h - b2_h).pow(2)) * 0.25
    return torch.exp(-torch.sqrt(wasserstein.clamp_min(eps)) / constant).clamp(0.0, 1.0)


def mpdiou_xyxy(box1: torch.Tensor, box2: torch.Tensor, imgsz: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute MPDIoU similarity for xyxy boxes with image-scale normalization."""
    iou = bbox_iou(box1, box2, xywh=False, eps=eps).clamp(0.0, 1.0)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    d1 = (b1_x1 - b2_x1).pow(2) + (b1_y1 - b2_y1).pow(2)
    d2 = (b1_x2 - b2_x2).pow(2) + (b1_y2 - b2_y2).pow(2)
    normalizer = imgsz[0].pow(2) + imgsz[1].pow(2) + eps
    return (iou - (d1 + d2) / normalizer).clamp(-1.0, 1.0)


def inner_ciou_xyxy(box1: torch.Tensor, box2: torch.Tensor, ratio: float = 0.7, eps: float = 1e-7) -> torch.Tensor:
    """Compute Inner-CIoU similarity for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = (b1_x2 - b1_x1).clamp_min(eps), (b1_y2 - b1_y1).clamp_min(eps)
    w2, h2 = (b2_x2 - b2_x1).clamp_min(eps), (b2_y2 - b2_y1).clamp_min(eps)
    b1_cx, b1_cy = (b1_x1 + b1_x2) * 0.5, (b1_y1 + b1_y2) * 0.5
    b2_cx, b2_cy = (b2_x1 + b2_x2) * 0.5, (b2_y1 + b2_y2) * 0.5

    b1_hw, b1_hh = w1 * 0.5 * ratio, h1 * 0.5 * ratio
    b2_hw, b2_hh = w2 * 0.5 * ratio, h2 * 0.5 * ratio
    i1_x1, i1_x2, i1_y1, i1_y2 = b1_cx - b1_hw, b1_cx + b1_hw, b1_cy - b1_hh, b1_cy + b1_hh
    i2_x1, i2_x2, i2_y1, i2_y2 = b2_cx - b2_hw, b2_cx + b2_hw, b2_cy - b2_hh, b2_cy + b2_hh
    inter = (i1_x2.minimum(i2_x2) - i1_x1.maximum(i2_x1)).clamp(0) * (
        i1_y2.minimum(i2_y2) - i1_y1.maximum(i2_y1)
    ).clamp(0)
    union = (w1 * h1 + w2 * h2) * ratio**2 - inter + eps
    inner_iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = (b2_cx - b1_cx).pow(2) + (b2_cy - b1_cy).pow(2)
    iou = bbox_iou(box1, box2, xywh=False, eps=eps).clamp(0.0, 1.0)
    v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return (inner_iou - (rho2 / c2 + v * alpha)).clamp(-1.0, 1.0)


def shape_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, scale: float = 0.0, eps: float = 1e-7) -> torch.Tensor:
    """Compute Shape-IoU similarity for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = (b1_x2 - b1_x1).clamp_min(eps), (b1_y2 - b1_y1).clamp_min(eps)
    w2, h2 = (b2_x2 - b2_x1).clamp_min(eps), (b2_y2 - b2_y1).clamp_min(eps)
    iou = bbox_iou(box1, box2, xywh=False, eps=eps).clamp(0.0, 1.0)

    denom = w2.pow(scale) + h2.pow(scale) + eps
    ww = 2 * w2.pow(scale) / denom
    hh = 2 * h2.pow(scale) / denom
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    center_distance_x = (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) * 0.25
    center_distance_y = (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2) * 0.25
    distance = (hh * center_distance_x + ww * center_distance_y) / c2
    omega_w = hh * (w1 - w2).abs() / w1.maximum(w2)
    omega_h = ww * (h1 - h2).abs() / h1.maximum(h2)
    shape_cost = (1 - torch.exp(-omega_w)).pow(4) + (1 - torch.exp(-omega_h)).pow(4)
    return (iou - distance - 0.5 * shape_cost).clamp(-1.0, 1.0)


def eiou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute Efficient-IoU similarity for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = (b1_x2 - b1_x1).clamp_min(eps), (b1_y2 - b1_y1).clamp_min(eps)
    w2, h2 = (b2_x2 - b2_x1).clamp_min(eps), (b2_y2 - b2_y1).clamp_min(eps)
    iou = bbox_iou(box1, box2, xywh=False, eps=eps).clamp(0.0, 1.0)

    cw = (b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)).clamp_min(eps)
    ch = (b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)).clamp_min(eps)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) * 0.25
    wh_cost = (w1 - w2).pow(2) / cw.pow(2) + (h1 - h2).pow(2) / ch.pow(2)
    return (iou - rho2 / c2 - wh_cost).clamp(-1.0, 1.0)


def siou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute SIoU similarity for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = (b1_x2 - b1_x1).clamp_min(eps), (b1_y2 - b1_y1).clamp_min(eps)
    w2, h2 = (b2_x2 - b2_x1).clamp_min(eps), (b2_y2 - b2_y1).clamp_min(eps)
    iou = bbox_iou(box1, box2, xywh=False, eps=eps).clamp(0.0, 1.0)

    b1_cx, b1_cy = (b1_x1 + b1_x2) * 0.5, (b1_y1 + b1_y2) * 0.5
    b2_cx, b2_cy = (b2_x1 + b2_x2) * 0.5, (b2_y1 + b2_y2) * 0.5
    s_cw, s_ch = (b2_cx - b1_cx).abs(), (b2_cy - b1_cy).abs()
    sigma = torch.sqrt(s_cw.pow(2) + s_ch.pow(2)).clamp_min(eps)
    sin_alpha_1, sin_alpha_2 = s_cw / sigma, s_ch / sigma
    threshold = math.sqrt(2) / 2
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1).clamp(-1.0 + eps, 1.0 - eps)
    angle_cost = torch.cos(torch.asin(sin_alpha) * 2 - math.pi / 2)

    cw = (b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)).clamp_min(eps)
    ch = (b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)).clamp_min(eps)
    rho_x, rho_y = (s_cw / cw).pow(2), (s_ch / ch).pow(2)
    gamma = angle_cost - 2
    distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)

    omega_w = (w1 - w2).abs() / w1.maximum(w2).clamp_min(eps)
    omega_h = (h1 - h2).abs() / h1.maximum(h2).clamp_min(eps)
    shape_cost = (1 - torch.exp(-omega_w)).pow(4) + (1 - torch.exp(-omega_h)).pow(4)
    return (iou - 0.5 * (distance_cost + shape_cost)).clamp(-1.0, 1.0)


def nciou_xyxy(box1: torch.Tensor, box2: torch.Tensor, n: float = 5.0, eps: float = 1e-7) -> torch.Tensor:
    """Compute N-CIoU similarity by replacing the IoU term in CIoU with N-IoU."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = (b1_x2 - b1_x1).clamp_min(eps), (b1_y2 - b1_y1).clamp_min(eps)
    w2, h2 = (b2_x2 - b2_x1).clamp_min(eps), (b2_y2 - b2_y1).clamp_min(eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    niou = (1.0 + n) * inter / (union + n * inter + eps)

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) * 0.25
    plain_iou = inter / union
    v = (4 / math.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - plain_iou + (1 + eps))
    return (niou - (rho2 / c2 + v * alpha)).clamp(-1.0, 1.0)


def piou_v2_loss_xyxy(
    box1: torch.Tensor, box2: torch.Tensor, lambda_: float = 1.3, eps: float = 1e-7
) -> torch.Tensor:
    """Compute Powerful-IoU v2 loss for xyxy boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = (b1_x2 - b1_x1).clamp_min(eps), (b1_y2 - b1_y1).clamp_min(eps)
    w2, h2 = (b2_x2 - b2_x1).clamp_min(eps), (b2_y2 - b2_y1).clamp_min(eps)
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)
    iou = inter / (w1 * h1 + w2 * h2 - inter + eps)

    dw1 = (b1_x1.minimum(b1_x2) - b2_x1.minimum(b2_x2)).abs()
    dw2 = (b1_x1.maximum(b1_x2) - b2_x1.maximum(b2_x2)).abs()
    dh1 = (b1_y1.minimum(b1_y2) - b2_y1.minimum(b2_y2)).abs()
    dh2 = (b1_y1.maximum(b1_y2) - b2_y1.maximum(b2_y2)).abs()
    p = ((dw1 + dw2) / w2.abs().clamp_min(eps) + (dh1 + dh2) / h2.abs().clamp_min(eps)) * 0.25
    piou_loss = 2.0 - iou - torch.exp(-p.pow(2))
    q = torch.exp(-p)
    x = q * lambda_
    return 3.0 * x * torch.exp(-x.pow(2)) * piou_loss


def wiou_v3_loss_xyxy(
    box1: torch.Tensor,
    box2: torch.Tensor,
    iou_loss_mean: torch.Tensor,
    alpha: float = 1.7,
    delta: float = 2.7,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Compute Wise-IoU v3 loss for xyxy boxes with non-monotonic focusing."""
    iou = bbox_iou(box1, box2, xywh=False, eps=eps).clamp(0.0, 1.0)
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    c1x, c1y = (b1_x1 + b1_x2) * 0.5, (b1_y1 + b1_y2) * 0.5
    c2x, c2y = (b2_x1 + b2_x2) * 0.5, (b2_y1 + b2_y2) * 0.5
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    center_distance = (c1x - c2x).pow(2) + (c1y - c2y).pow(2)
    box_distance = (cw.pow(2) + ch.pow(2)).clamp_min(eps)
    iou_loss = 1.0 - iou
    wise_loss = torch.exp(center_distance / box_distance.detach()) * iou_loss
    beta = iou_loss.detach() / iou_loss_mean.clamp_min(eps)
    divisor = delta * torch.pow(torch.as_tensor(alpha, dtype=box1.dtype, device=box1.device), beta - delta)
    return wise_loss * beta / divisor.clamp_min(eps)


class BboxLoss(nn.Module):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(
        self,
        reg_max: int = 16,
        iou_loss: str = "ciou",
        interp_iou_min: float = 0.2,
        interp_iou_max: float = 0.8,
        focaler_iou_min: float = 0.0,
        focaler_iou_max: float = 0.95,
        nwd_alpha: float = 0.5,
        nwd_constant: float = 12.8,
        inner_iou_ratio: float = 0.7,
        shape_iou_scale: float = 0.0,
        niou_n: float = 5.0,
        alpha_iou_power: float = 1.2,
        piou_lambda: float = 1.3,
        wiou_alpha: float = 1.7,
        wiou_delta: float = 2.7,
        hybrid_iou_alpha: float = 0.9,
        aux_iou_gain: float = 0.05,
        wiou_focus_min: float = 0.5,
        wiou_focus_max: float = 2.0,
        sa_nwd_ref: float = 64.0,
        sa_nwd_gamma: float = 0.5,
        ltrb_loss: str = "l1",
        ltrb_beta: float = 0.03,
        ltrb_quality_gain: float = 0.5,
        ltrb_start_epoch: int = 0,
        ltrb_warm_epochs: int = 1,
    ):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.iou_loss = iou_loss.lower()
        self.interp_iou_min = interp_iou_min
        self.interp_iou_max = interp_iou_max
        self.focaler_iou_min = focaler_iou_min
        self.focaler_iou_max = focaler_iou_max
        self.nwd_alpha = nwd_alpha
        self.nwd_constant = nwd_constant
        self.inner_iou_ratio = inner_iou_ratio
        self.shape_iou_scale = shape_iou_scale
        self.niou_n = niou_n
        self.alpha_iou_power = alpha_iou_power
        self.piou_lambda = piou_lambda
        self.wiou_alpha = wiou_alpha
        self.wiou_delta = wiou_delta
        self.hybrid_iou_alpha = hybrid_iou_alpha
        self.aux_iou_gain = aux_iou_gain
        self.wiou_focus_min = wiou_focus_min
        self.wiou_focus_max = wiou_focus_max
        self.sa_nwd_ref = sa_nwd_ref
        self.sa_nwd_gamma = sa_nwd_gamma
        self.ltrb_loss = ltrb_loss.lower()
        self.ltrb_beta = ltrb_beta
        self.ltrb_quality_gain = ltrb_quality_gain
        self.ltrb_start_epoch = ltrb_start_epoch
        self.ltrb_warm_epochs = max(1, ltrb_warm_epochs)
        self.register_buffer("wiou_iou_loss_mean", torch.tensor(1.0))

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        loss_iou = None
        if self.iou_loss == "dynamic_interpiou":
            iou = dynamic_interpiou(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                alpha_min=self.interp_iou_min,
                alpha_max=self.interp_iou_max,
            )
        elif self.iou_loss == "focaler_ciou":
            iou = focaler_ciou(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                lower=self.focaler_iou_min,
                upper=self.focaler_iou_max,
            )
        elif self.iou_loss == "nwd_ciou":
            pred_boxes = pred_bboxes[fg_mask]
            target_boxes = target_bboxes[fg_mask]
            fg_stride = stride.unsqueeze(0).expand(fg_mask.shape[0], -1, -1)[fg_mask]
            ciou = bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True).clamp(0.0, 1.0)
            nwd = bbox_nwd_xyxy(pred_boxes * fg_stride, target_boxes * fg_stride, self.nwd_constant)
            iou = self.nwd_alpha * ciou + (1.0 - self.nwd_alpha) * nwd
        elif self.iou_loss == "ciou_nwd_aux":
            pred_boxes = pred_bboxes[fg_mask]
            target_boxes = target_bboxes[fg_mask]
            fg_stride = stride.unsqueeze(0).expand(fg_mask.shape[0], -1, -1)[fg_mask]
            ciou_loss = 1.0 - bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True)
            nwd_loss = 1.0 - bbox_nwd_xyxy(pred_boxes * fg_stride, target_boxes * fg_stride, self.nwd_constant)
            loss_iou = ((ciou_loss + self.aux_iou_gain * nwd_loss) * weight).sum() / target_scores_sum
        elif self.iou_loss == "sa_nwd_ciou":
            pred_boxes = pred_bboxes[fg_mask]
            target_boxes = target_bboxes[fg_mask]
            fg_stride = stride.unsqueeze(0).expand(fg_mask.shape[0], -1, -1)[fg_mask]
            pred_px = pred_boxes * fg_stride
            target_px = target_boxes * fg_stride
            ciou_loss = 1.0 - bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True)
            nwd_loss = 1.0 - bbox_nwd_xyxy(pred_px, target_px, self.nwd_constant)

            tw = (target_px[:, 2:3] - target_px[:, 0:1]).clamp_min(1e-7)
            th = (target_px[:, 3:4] - target_px[:, 1:2]).clamp_min(1e-7)
            obj_scale = torch.sqrt(tw * th)
            ref = torch.as_tensor(self.sa_nwd_ref, device=target_px.device, dtype=target_px.dtype).clamp_min(1e-7)
            scale_gate = (ref / (obj_scale + ref)).pow(self.sa_nwd_gamma).detach()
            loss_iou = ((ciou_loss + self.aux_iou_gain * scale_gate * nwd_loss) * weight).sum() / target_scores_sum
        elif self.iou_loss == "mpdiou":
            fg_stride = stride.unsqueeze(0).expand(fg_mask.shape[0], -1, -1)[fg_mask]
            iou = mpdiou_xyxy(pred_bboxes[fg_mask] * fg_stride, target_bboxes[fg_mask] * fg_stride, imgsz)
        elif self.iou_loss == "inner_ciou":
            iou = inner_ciou_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask], ratio=self.inner_iou_ratio)
        elif self.iou_loss == "shape_iou":
            iou = shape_iou_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask], scale=self.shape_iou_scale)
        elif self.iou_loss == "eiou":
            iou = eiou_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        elif self.iou_loss == "siou":
            iou = siou_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        elif self.iou_loss == "alpha_ciou":
            ciou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
            iou = ciou.clamp(0.0, 1.0).pow(self.alpha_iou_power)
        elif self.iou_loss == "nciou":
            iou = nciou_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask], n=self.niou_n)
        elif self.iou_loss == "piou_v2":
            iou = 1.0 - piou_v2_loss_xyxy(pred_bboxes[fg_mask], target_bboxes[fg_mask], lambda_=self.piou_lambda)
        elif self.iou_loss == "wiou_v3":
            pred_boxes = pred_bboxes[fg_mask]
            target_boxes = target_bboxes[fg_mask]
            plain_iou_loss = 1.0 - bbox_iou(pred_boxes, target_boxes, xywh=False).detach().clamp(0.0, 1.0)
            if self.training and plain_iou_loss.numel():
                with torch.no_grad():
                    self.wiou_iou_loss_mean.mul_(0.99).add_(plain_iou_loss.mean() * 0.01)
            iou = 1.0 - wiou_v3_loss_xyxy(
                pred_boxes,
                target_boxes,
                self.wiou_iou_loss_mean,
                alpha=self.wiou_alpha,
                delta=self.wiou_delta,
            )
        elif self.iou_loss == "ciou_nciou_mix":
            pred_boxes = pred_bboxes[fg_mask]
            target_boxes = target_bboxes[fg_mask]
            ciou = bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True)
            nciou = nciou_xyxy(pred_boxes, target_boxes, n=self.niou_n)
            iou = self.hybrid_iou_alpha * ciou + (1.0 - self.hybrid_iou_alpha) * nciou
        elif self.iou_loss == "ciou_piou_aux":
            pred_boxes = pred_bboxes[fg_mask]
            target_boxes = target_bboxes[fg_mask]
            ciou_loss = 1.0 - bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True)
            piou_loss = piou_v2_loss_xyxy(pred_boxes, target_boxes, lambda_=self.piou_lambda)
            loss_iou = ((ciou_loss + self.aux_iou_gain * piou_loss) * weight).sum() / target_scores_sum
        elif self.iou_loss == "ciou_wiou_focus":
            pred_boxes = pred_bboxes[fg_mask]
            target_boxes = target_bboxes[fg_mask]
            ciou_loss = 1.0 - bbox_iou(pred_boxes, target_boxes, xywh=False, CIoU=True)
            plain_iou_loss = 1.0 - bbox_iou(pred_boxes, target_boxes, xywh=False).detach().clamp(0.0, 1.0)
            if self.training and plain_iou_loss.numel():
                with torch.no_grad():
                    self.wiou_iou_loss_mean.mul_(0.99).add_(plain_iou_loss.mean() * 0.01)
            wiou_loss = wiou_v3_loss_xyxy(
                pred_boxes,
                target_boxes,
                self.wiou_iou_loss_mean,
                alpha=self.wiou_alpha,
                delta=self.wiou_delta,
            ).detach()
            focus = wiou_loss / plain_iou_loss.clamp_min(1e-7)
            focus = focus / focus.mean().clamp_min(1e-7)
            focus = focus.clamp(self.wiou_focus_min, self.wiou_focus_max)
            loss_iou = (ciou_loss * focus * weight).sum() / target_scores_sum
        else:
            iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        if loss_iou is None:
            loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            pred_ltrb = pred_dist[fg_mask]
            target_ltrb = target_ltrb[fg_mask]
            base_ltrb_loss = F.l1_loss(pred_ltrb, target_ltrb, reduction="none")
            epoch = int(getattr(self, "current_epoch", 0) or 0)
            schedule = (epoch + 1 - self.ltrb_start_epoch) / float(self.ltrb_warm_epochs)
            schedule = max(0.0, min(1.0, schedule))
            if schedule <= 0.0 or self.ltrb_loss == "l1":
                ltrb_loss = base_ltrb_loss
            elif self.ltrb_loss == "smooth_l1":
                ltrb_loss = F.smooth_l1_loss(pred_ltrb, target_ltrb, beta=self.ltrb_beta, reduction="none")
            elif self.ltrb_loss == "balanced_l1":
                diff = (pred_ltrb - target_ltrb).abs()
                alpha, gamma = 0.5, 1.5
                beta = max(float(self.ltrb_beta), 1e-7)
                b = math.exp(gamma / alpha) - 1.0
                ltrb_loss = torch.where(
                    diff < beta,
                    alpha / b * (b * diff + 1.0) * torch.log(b * diff / beta + 1.0) - alpha * diff,
                    gamma * diff + gamma / b - alpha * beta,
                )
            elif self.ltrb_loss == "quality_smooth_l1":
                ltrb_loss = F.smooth_l1_loss(pred_ltrb, target_ltrb, beta=self.ltrb_beta, reduction="none")
                quality = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True).detach()
                quality_weight = 1.0 + self.ltrb_quality_gain * (1.0 - quality).clamp(0.0, 1.0)
                ltrb_loss = ltrb_loss * quality_weight
            else:
                ltrb_loss = base_ltrb_loss
            if 0.0 < schedule < 1.0:
                ltrb_loss = base_ltrb_loss * (1.0 - schedule) + ltrb_loss * schedule
            loss_dfl = ltrb_loss.mean(-1, keepdim=True) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation Loss.

    Attributes:
        size_average (bool): Option to average the loss by the batch_size.
        use_target_weight (bool): Option to use weighted loss.
        residual (bool): Option to add L1 loss and let the flow learn the residual error distribution.

    References:
        https://arxiv.org/abs/2107.11291
        https://github.com/open-mmlab/mmpose/blob/main/mmpose/models/losses/regression_loss.py
    """

    def __init__(self, use_target_weight: bool = True, size_average: bool = True, residual: bool = True):
        """Initialize RLELoss with target weight and residual options.

        Args:
            use_target_weight (bool): Whether to use target weights for loss calculation.
            size_average (bool): Whether to average the loss over elements.
            residual (bool): Whether to include residual log-likelihood term.
        """
        super().__init__()
        self.size_average = size_average
        self.use_target_weight = use_target_weight
        self.residual = residual

    def forward(
        self, sigma: torch.Tensor, log_phi: torch.Tensor, error: torch.Tensor, target_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            sigma (torch.Tensor): Output sigma, shape (N, D).
            log_phi (torch.Tensor): Output log_phi, shape (N).
            error (torch.Tensor): Error, shape (N, D).
            target_weight (torch.Tensor): Weights across different joint types, shape (N).
        """
        log_sigma = torch.log(sigma)
        loss = log_sigma - log_phi.unsqueeze(1)

        if self.residual:
            loss += torch.log(sigma * 2) + torch.abs(error)

        if self.use_target_weight:
            assert target_weight is not None, "'target_weight' should not be None when 'use_target_weight' is True."
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss *= target_weight

        if self.size_average:
            loss /= len(loss)

        return loss.sum()


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses for rotated bounding boxes."""

    def __init__(self, reg_max: int):
        """Initialize the RotatedBboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for rotated bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = rbox2dist(
                target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5], reg_max=self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = rbox2dist(target_bboxes[..., :4], anchor_points, target_bboxes[..., 4:5])
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class MultiChannelDiceLoss(nn.Module):
    """Criterion class for computing multi-channel Dice losses."""

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        """Initialize MultiChannelDiceLoss with smoothing and reduction options.

        Args:
            smooth (float): Smoothing factor to avoid division by zero.
            reduction (str): Reduction method ('mean', 'sum', or 'none').
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate multi-channel Dice loss between predictions and targets."""
        assert pred.size() == target.size(), "the size of predict and target must be equal."

        pred = pred.sigmoid()
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice
        dice_loss = dice_loss.mean(dim=1)

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class BCEDiceLoss(nn.Module):
    """Criterion class for computing combined BCE and Dice losses."""

    def __init__(self, weight_bce: float = 0.5, weight_dice: float = 0.5):
        """Initialize BCEDiceLoss with BCE and Dice weight factors.

        Args:
            weight_bce (float): Weight factor for BCE loss component.
            weight_dice (float): Weight factor for Dice loss component.
        """
        super().__init__()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = MultiChannelDiceLoss(smooth=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined BCE and Dice loss between predictions and targets."""
        _, _, mask_h, mask_w = pred.shape
        if tuple(target.shape[-2:]) != (mask_h, mask_w):  # downsample to the same size as pred
            target = F.interpolate(target, (mask_h, mask_w), mode="nearest")
        return self.weight_bce * self.bce(pred, target) + self.weight_dice * self.dice(pred, target)


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas: torch.Tensor) -> None:
        """Initialize the KeypointLoss class with keypoint sigmas."""
        super().__init__()
        self.sigmas = sigmas

    def forward(
        self, pred_kpts: torch.Tensor, gt_kpts: torch.Tensor, kpt_mask: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Calculate keypoint loss factor and Euclidean distance loss for keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses for YOLOv8 object detection."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize v8DetectionLoss with model parameters and task-aligned assignment settings."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)
        self.cls_loss = getattr(h, "cls_loss", "bce").lower()
        self.aux_loss = getattr(h, "aux_loss", "none").lower()
        self.focal_gamma = getattr(h, "focal_gamma", 1.5)
        self.focal_alpha = getattr(h, "focal_alpha", 0.25)
        self.bafocal_gamma = getattr(h, "bafocal_gamma", 1.2)
        self.bafocal_gain = getattr(h, "bafocal_gain", 0.25)
        self.bafocal_neg_gain = getattr(h, "bafocal_neg_gain", 0.25)
        self.cbfocal_gamma = getattr(h, "cbfocal_gamma", 1.2)
        self.cbfocal_pos_gain = getattr(h, "cbfocal_pos_gain", 0.75)
        self.cbfocal_neg_relief = getattr(h, "cbfocal_neg_relief", 0.25)
        self.cbfocal_hard_gain = getattr(h, "cbfocal_hard_gain", 0.15)
        self.scb_read_class = int(getattr(h, "scb_read_class", 1))
        self.scb_write_class = int(getattr(h, "scb_write_class", 2))
        self.scb_rw_margin = getattr(h, "scb_rw_margin", 0.12)
        self.scb_rw_gain = getattr(h, "scb_rw_gain", 0.06)
        self.scb_write_gain = getattr(h, "scb_write_gain", 0.03)
        self.scb_quality_thr = getattr(h, "scb_quality_thr", 0.0)
        self.scb_start_epoch = int(getattr(h, "scb_start_epoch", 0))
        self.scb_warm_epochs = max(int(getattr(h, "scb_warm_epochs", 1)), 1)
        self.scb_logit_tau = getattr(h, "scb_logit_tau", 0.5)
        self.scb_cals_eps = getattr(h, "scb_cals_eps", 0.03)
        self.scb_cals_pos_gain = getattr(h, "scb_cals_pos_gain", 0.30)
        self.scb_cals_power = getattr(h, "scb_cals_power", 0.5)
        self.scb_qfocal_gamma = getattr(h, "scb_qfocal_gamma", 1.0)
        self.scb_qfocal_pos_gain = getattr(h, "scb_qfocal_pos_gain", 0.12)
        self.scb_qfocal_neg_gain = getattr(h, "scb_qfocal_neg_gain", 0.03)
        self.scb_qfocal_class_gain = getattr(h, "scb_qfocal_class_gain", 0.12)
        self.scb_qfocal_class_cap = getattr(h, "scb_qfocal_class_cap", 1.25)
        self.scb_qfocal_power = getattr(h, "scb_qfocal_power", 0.5)
        self.scb_qm_margin = getattr(h, "scb_qm_margin", 0.05)
        self.scb_qm_gain = getattr(h, "scb_qm_gain", 0.015)
        self.scb_qm_quality_thr = getattr(h, "scb_qm_quality_thr", 0.25)
        self.scb_qm_start_epoch = int(getattr(h, "scb_qm_start_epoch", 30))
        self.scb_qm_warm_epochs = max(int(getattr(h, "scb_qm_warm_epochs", 20)), 1)
        self.scb_aux_gain = getattr(h, "scb_aux_gain", 0.0)
        self.scb_aux_write_gain = getattr(h, "scb_aux_write_gain", 1.0)
        self.scb_aux_rw_gain = getattr(h, "scb_aux_rw_gain", 1.0)
        self.scb_aux_margin = getattr(h, "scb_aux_margin", 0.02)
        self.scb_aux_quality_thr = getattr(h, "scb_aux_quality_thr", 0.30)
        self.scb_aux_start_epoch = int(getattr(h, "scb_aux_start_epoch", 25))
        self.scb_aux_warm_epochs = max(int(getattr(h, "scb_aux_warm_epochs", 15)), 1)
        self.scb_qalign_gain = getattr(h, "scb_qalign_gain", 0.0)
        self.scb_qalign_quality_thr = getattr(h, "scb_qalign_quality_thr", 0.20)
        self.scb_qalign_start_epoch = int(getattr(h, "scb_qalign_start_epoch", 30))
        self.scb_qalign_warm_epochs = max(int(getattr(h, "scb_qalign_warm_epochs", 15)), 1)
        self.scb_qalign_class_power = getattr(h, "scb_qalign_class_power", 0.25)
        self.scb_qalign_class_cap = getattr(h, "scb_qalign_class_cap", 1.5)
        self.scb_class_counts = torch.tensor(
            [
                float(getattr(h, "scb_count_hand", 8897.0)),
                float(getattr(h, "scb_count_read", 8408.0)),
                float(getattr(h, "scb_count_write", 3078.0)),
            ],
            device=device,
        )
        self.vfl = VarifocalLoss()

        assigner_topk = getattr(h, "tal_topk", None)
        assigner_topk = tal_topk if assigner_topk is None else int(assigner_topk)
        assigner_topk2 = getattr(h, "tal_topk2", tal_topk2)
        assigner_topk2 = None if assigner_topk2 is None else int(assigner_topk2)
        self.assigner = TaskAlignedAssigner(
            topk=assigner_topk,
            num_classes=self.nc,
            alpha=float(getattr(h, "tal_alpha", 0.5)),
            beta=float(getattr(h, "tal_beta", 6.0)),
            stride=self.stride.tolist(),
            topk2=assigner_topk2,
        )
        self.bbox_loss = BboxLoss(
            m.reg_max,
            iou_loss=getattr(h, "iou_loss", "ciou"),
            interp_iou_min=getattr(h, "interp_iou_min", 0.2),
            interp_iou_max=getattr(h, "interp_iou_max", 0.8),
            focaler_iou_min=getattr(h, "focaler_iou_min", 0.0),
            focaler_iou_max=getattr(h, "focaler_iou_max", 0.95),
            nwd_alpha=getattr(h, "nwd_alpha", 0.5),
            nwd_constant=getattr(h, "nwd_constant", 12.8),
            inner_iou_ratio=getattr(h, "inner_iou_ratio", 0.7),
            shape_iou_scale=getattr(h, "shape_iou_scale", 0.0),
            niou_n=getattr(h, "niou_n", 5.0),
            alpha_iou_power=getattr(h, "alpha_iou_power", 1.2),
            piou_lambda=getattr(h, "piou_lambda", 1.3),
            wiou_alpha=getattr(h, "wiou_alpha", 1.7),
            wiou_delta=getattr(h, "wiou_delta", 2.7),
            hybrid_iou_alpha=getattr(h, "hybrid_iou_alpha", 0.9),
            aux_iou_gain=getattr(h, "aux_iou_gain", 0.05),
            wiou_focus_min=getattr(h, "wiou_focus_min", 0.5),
            wiou_focus_max=getattr(h, "wiou_focus_max", 2.0),
            sa_nwd_ref=getattr(h, "sa_nwd_ref", 64.0),
            sa_nwd_gamma=getattr(h, "sa_nwd_gamma", 0.5),
            ltrb_loss=getattr(h, "ltrb_loss", "l1"),
            ltrb_beta=getattr(h, "ltrb_beta", 0.03),
            ltrb_quality_gain=getattr(h, "ltrb_quality_gain", 0.5),
            ltrb_start_epoch=getattr(h, "ltrb_start_epoch", 0),
            ltrb_warm_epochs=getattr(h, "ltrb_warm_epochs", 1),
        ).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def scb_classification_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        gt_labels: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dataset-tailored classification loss for SCB read/write behavior confusion."""
        target_scores_float = target_scores.to(dtype)
        loss_cls = self.bce(pred_scores, target_scores_float).sum() / target_scores_sum

        if self.nc <= max(self.scb_read_class, self.scb_write_class) or not fg_mask.any():
            return loss_cls

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        schedule = (epoch + 1 - self.scb_start_epoch) / float(self.scb_warm_epochs)
        schedule = max(0.0, min(1.0, schedule))
        if schedule <= 0.0:
            return loss_cls

        safe_gt_idx = target_gt_idx.clamp(min=0, max=gt_labels.shape[1] - 1)
        assigned_labels = gt_labels.long().squeeze(-1).gather(1, safe_gt_idx).clamp(0, self.nc - 1)
        pos_quality = target_scores_float.sum(-1).detach()
        pos_mask = fg_mask & pos_quality.gt(self.scb_quality_thr)
        if not pos_mask.any():
            return loss_cls

        pos_logits = pred_scores[pos_mask]
        pos_labels = assigned_labels[pos_mask]
        pos_weight = pos_quality[pos_mask].clamp_min(1e-4)
        correct_logits = pos_logits.gather(1, pos_labels.unsqueeze(1)).squeeze(1)

        write_mask = pos_labels.eq(self.scb_write_class)
        if self.scb_write_gain > 0 and write_mask.any():
            write_loss = F.softplus(-correct_logits[write_mask])
            loss_cls += schedule * self.scb_write_gain * (write_loss * pos_weight[write_mask]).sum() / pos_weight[
                write_mask
            ].sum().clamp_min(1.0)

        rw_mask = pos_labels.eq(self.scb_read_class) | pos_labels.eq(self.scb_write_class)
        if self.scb_rw_gain > 0 and rw_mask.any():
            confuse_labels = torch.where(
                pos_labels[rw_mask].eq(self.scb_read_class),
                torch.full_like(pos_labels[rw_mask], self.scb_write_class),
                torch.full_like(pos_labels[rw_mask], self.scb_read_class),
            )
            confuse_logits = pos_logits[rw_mask].gather(1, confuse_labels.unsqueeze(1)).squeeze(1)
            rw_margin_loss = F.softplus(confuse_logits - correct_logits[rw_mask] + self.scb_rw_margin)
            loss_cls += schedule * self.scb_rw_gain * (rw_margin_loss * pos_weight[rw_mask]).sum() / pos_weight[
                rw_mask
            ].sum().clamp_min(1.0)

        return loss_cls

    def scb_logit_adjust_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Class-prior-aware BCE for SCB long-tailed classroom behavior categories."""
        if self.nc != 3 or self.scb_logit_tau <= 0:
            return self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        priors = (self.scb_class_counts / self.scb_class_counts.sum()).to(device=pred_scores.device, dtype=dtype)
        adjustment = self.scb_logit_tau * priors.clamp_min(1e-6).log().view(1, 1, -1)
        adjusted_scores = pred_scores + adjustment
        return self.bce(adjusted_scores, target_scores.to(dtype)).sum() / target_scores_sum

    def scb_cals_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        gt_labels: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Confusion-aware label smoothing with mild minority-positive compensation for SCB."""
        target_scores_float = target_scores.to(dtype)
        if self.nc != 3 or not fg_mask.any():
            return self.bce(pred_scores, target_scores_float).sum() / target_scores_sum

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        schedule = (epoch + 1 - self.scb_start_epoch) / float(self.scb_warm_epochs)
        schedule = max(0.0, min(1.0, schedule))
        if schedule <= 0.0:
            return self.bce(pred_scores, target_scores_float).sum() / target_scores_sum

        safe_gt_idx = target_gt_idx.clamp(min=0, max=gt_labels.shape[1] - 1)
        assigned_labels = gt_labels.long().squeeze(-1).gather(1, safe_gt_idx).clamp(0, self.nc - 1)
        pos_quality = target_scores_float.sum(-1).detach()
        pos_mask = fg_mask & pos_quality.gt(0)
        if not pos_mask.any():
            return self.bce(pred_scores, target_scores_float).sum() / target_scores_sum

        soft_targets = target_scores_float.clone()
        pos_labels = assigned_labels[pos_mask]

        eps = float(max(0.0, min(self.scb_cals_eps * schedule, 0.15)))
        rw_mask = pos_labels.eq(self.scb_read_class) | pos_labels.eq(self.scb_write_class)
        if eps > 0 and rw_mask.any():
            pos_b, pos_a = pos_mask.nonzero(as_tuple=True)
            rw_b = pos_b[rw_mask]
            rw_a = pos_a[rw_mask]
            rw_labels = pos_labels[rw_mask]
            sibling = torch.where(
                rw_labels.eq(self.scb_read_class),
                torch.full_like(rw_labels, self.scb_write_class),
                torch.full_like(rw_labels, self.scb_read_class),
            )
            quality = pos_quality[rw_b, rw_a].to(dtype)
            soft_targets[rw_b, rw_a, rw_labels] = target_scores_float[rw_b, rw_a, rw_labels] * (1.0 - eps)
            soft_targets[rw_b, rw_a, sibling] = torch.maximum(soft_targets[rw_b, rw_a, sibling], quality * eps)

        bce_loss = self.bce(pred_scores, soft_targets)
        if self.scb_cals_pos_gain > 0:
            ratios = (self.scb_class_counts.max() / self.scb_class_counts.clamp_min(1.0)).pow(self.scb_cals_power)
            pos_weights = 1.0 + schedule * self.scb_cals_pos_gain * (ratios - 1.0).clamp_min(0.0)
            pos_weights = pos_weights.to(device=pred_scores.device, dtype=dtype).view(1, 1, -1)
            true_pos = target_scores_float.gt(0)
            bce_loss = torch.where(true_pos, bce_loss * pos_weights, bce_loss)

        return bce_loss.sum() / target_scores_sum

    def scb_qfocal_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Quality-aware focal BCE with mild SCB class-frequency compensation."""
        target_scores_float = target_scores.to(dtype)
        bce_loss = self.bce(pred_scores, target_scores_float)

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        schedule = (epoch + 1 - self.scb_start_epoch) / float(self.scb_warm_epochs)
        schedule = max(0.0, min(1.0, schedule))
        if schedule <= 0.0:
            return bce_loss.sum() / target_scores_sum

        pred_prob = pred_scores.sigmoid().detach()
        pos_mask = target_scores_float.gt(0)
        pos_focus = (target_scores_float - pred_prob).abs().clamp(0.0, 1.0).pow(self.scb_qfocal_gamma)
        neg_focus = pred_prob.clamp(0.0, 1.0).pow(self.scb_qfocal_gamma)
        pos_weight = 1.0 + schedule * self.scb_qfocal_pos_gain * pos_focus
        neg_weight = 1.0 + schedule * self.scb_qfocal_neg_gain * neg_focus
        bce_loss *= torch.where(pos_mask, pos_weight, neg_weight)

        if self.nc == 3 and self.scb_qfocal_class_gain > 0:
            ratios = (self.scb_class_counts.max() / self.scb_class_counts.clamp_min(1.0)).pow(
                self.scb_qfocal_power
            )
            ratios = (ratios - 1.0).clamp_min(0.0).clamp_max(self.scb_qfocal_class_cap)
            class_weights = 1.0 + schedule * self.scb_qfocal_class_gain * ratios
            class_weights = class_weights.to(device=pred_scores.device, dtype=dtype).view(1, 1, -1)
            bce_loss = torch.where(pos_mask, bce_loss * class_weights, bce_loss)

        return bce_loss.sum() / target_scores_sum

    def scb_qm_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        gt_labels: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        target_scores_sum: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """SCB-QFocal plus delayed high-quality read/write margin regularization."""
        loss_cls = self.scb_qfocal_loss(pred_scores, target_scores, target_scores_sum, dtype)
        if self.nc <= max(self.scb_read_class, self.scb_write_class) or not fg_mask.any() or self.scb_qm_gain <= 0:
            return loss_cls

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        schedule = (epoch + 1 - self.scb_qm_start_epoch) / float(self.scb_qm_warm_epochs)
        schedule = max(0.0, min(1.0, schedule))
        if schedule <= 0.0:
            return loss_cls

        target_scores_float = target_scores.to(dtype)
        safe_gt_idx = target_gt_idx.clamp(min=0, max=gt_labels.shape[1] - 1)
        assigned_labels = gt_labels.long().squeeze(-1).gather(1, safe_gt_idx).clamp(0, self.nc - 1)
        pos_quality = target_scores_float.sum(-1).detach()
        pos_mask = fg_mask & pos_quality.gt(self.scb_qm_quality_thr)
        if not pos_mask.any():
            return loss_cls

        pos_logits = pred_scores[pos_mask]
        pos_labels = assigned_labels[pos_mask]
        rw_mask = pos_labels.eq(self.scb_read_class) | pos_labels.eq(self.scb_write_class)
        if not rw_mask.any():
            return loss_cls

        correct_logits = pos_logits.gather(1, pos_labels.unsqueeze(1)).squeeze(1)
        confuse_labels = torch.where(
            pos_labels[rw_mask].eq(self.scb_read_class),
            torch.full_like(pos_labels[rw_mask], self.scb_write_class),
            torch.full_like(pos_labels[rw_mask], self.scb_read_class),
        )
        confuse_logits = pos_logits[rw_mask].gather(1, confuse_labels.unsqueeze(1)).squeeze(1)
        margin_loss = F.softplus(confuse_logits - correct_logits[rw_mask] + self.scb_qm_margin)
        pos_weight = pos_quality[pos_mask][rw_mask].clamp_min(1e-4)
        loss_cls += schedule * self.scb_qm_gain * (margin_loss * pos_weight).sum() / pos_weight.sum().clamp_min(1.0)
        return loss_cls

    def scb_auxiliary_loss(
        self,
        pred_scores: torch.Tensor,
        target_scores: torch.Tensor,
        gt_labels: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Training-only SCB auxiliary loss added on top of the unchanged YOLO classification loss."""
        if (
            self.aux_loss not in {"scb_write", "scb_rw"}
            or self.scb_aux_gain <= 0
            or self.nc <= max(self.scb_read_class, self.scb_write_class)
            or not fg_mask.any()
        ):
            return pred_scores.new_zeros(())

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        schedule = (epoch + 1 - self.scb_aux_start_epoch) / float(self.scb_aux_warm_epochs)
        schedule = max(0.0, min(1.0, schedule))
        if schedule <= 0.0:
            return pred_scores.new_zeros(())

        target_scores_float = target_scores.to(dtype)
        safe_gt_idx = target_gt_idx.clamp(min=0, max=gt_labels.shape[1] - 1)
        assigned_labels = gt_labels.long().squeeze(-1).gather(1, safe_gt_idx).clamp(0, self.nc - 1)
        pos_quality = target_scores_float.sum(-1).detach()
        pos_mask = fg_mask & pos_quality.gt(self.scb_aux_quality_thr)
        if not pos_mask.any():
            return pred_scores.new_zeros(())

        pos_logits = pred_scores[pos_mask]
        pos_labels = assigned_labels[pos_mask]
        pos_weight = pos_quality[pos_mask].clamp_min(1e-4)
        correct_logits = pos_logits.gather(1, pos_labels.unsqueeze(1)).squeeze(1)

        aux = pred_scores.new_zeros(())
        write_mask = pos_labels.eq(self.scb_write_class)
        if self.scb_aux_write_gain > 0 and write_mask.any():
            write_loss = F.softplus(-correct_logits[write_mask])
            aux = aux + self.scb_aux_write_gain * (write_loss * pos_weight[write_mask]).sum() / pos_weight[
                write_mask
            ].sum().clamp_min(1.0)

        rw_mask = pos_labels.eq(self.scb_read_class) | pos_labels.eq(self.scb_write_class)
        if self.aux_loss == "scb_rw" and self.scb_aux_rw_gain > 0 and rw_mask.any():
            confuse_labels = torch.where(
                pos_labels[rw_mask].eq(self.scb_read_class),
                torch.full_like(pos_labels[rw_mask], self.scb_write_class),
                torch.full_like(pos_labels[rw_mask], self.scb_read_class),
            )
            confuse_logits = pos_logits[rw_mask].gather(1, confuse_labels.unsqueeze(1)).squeeze(1)
            margin_loss = F.softplus(confuse_logits - correct_logits[rw_mask] + self.scb_aux_margin)
            aux = aux + self.scb_aux_rw_gain * (margin_loss * pos_weight[rw_mask]).sum() / pos_weight[
                rw_mask
            ].sum().clamp_min(1.0)

        return schedule * self.scb_aux_gain * aux

    def scb_quality_alignment_loss(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        gt_labels: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
        stride_tensor: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Align behavior confidence with localization quality for reliable positive samples only."""
        if self.aux_loss != "scb_qalign" or self.scb_qalign_gain <= 0 or not fg_mask.any():
            return pred_scores.new_zeros(())

        epoch = int(getattr(self, "current_epoch", 0) or 0)
        schedule = (epoch + 1 - self.scb_qalign_start_epoch) / float(self.scb_qalign_warm_epochs)
        schedule = max(0.0, min(1.0, schedule))
        if schedule <= 0.0:
            return pred_scores.new_zeros(())

        target_scores_float = target_scores.to(dtype)
        pos_quality = target_scores_float.sum(-1).detach()
        pos_mask = fg_mask & pos_quality.gt(self.scb_qalign_quality_thr)
        if not pos_mask.any():
            return pred_scores.new_zeros(())

        pred_boxes_px = (pred_bboxes * stride_tensor)[pos_mask].detach()
        target_boxes_px = target_bboxes[pos_mask].detach()
        quality_target = bbox_iou(pred_boxes_px, target_boxes_px, xywh=False).detach().clamp(0.0, 1.0).view(-1)
        if not quality_target.numel():
            return pred_scores.new_zeros(())

        safe_gt_idx = target_gt_idx.clamp(min=0, max=gt_labels.shape[1] - 1)
        assigned_labels = gt_labels.long().squeeze(-1).gather(1, safe_gt_idx).clamp(0, self.nc - 1)
        pos_labels = assigned_labels[pos_mask]
        correct_logits = pred_scores[pos_mask].gather(1, pos_labels.unsqueeze(1)).squeeze(1)

        sample_weight = pos_quality[pos_mask].to(dtype).clamp_min(1e-4)
        if self.nc == 3 and self.scb_qalign_class_power > 0:
            counts = self.scb_class_counts.to(device=pred_scores.device, dtype=dtype).clamp_min(1.0)
            class_weights = (counts.max() / counts).pow(self.scb_qalign_class_power)
            class_weights = class_weights.clamp(1.0, self.scb_qalign_class_cap)
            sample_weight = sample_weight * class_weights[pos_labels]

        quality_target = quality_target.to(dtype)
        qalign = F.binary_cross_entropy_with_logits(correct_logits, quality_target, reduction="none")
        qalign = (qalign * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
        return schedule * self.scb_qalign_gain * qalign

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets by converting to tensor format and scaling coordinates."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def get_assigned_targets_and_loss(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and
        target indices.
        """
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss with optional class weighting
        if self.cls_loss == "varifocal":
            target_labels = target_scores.gt(0).to(dtype)
            loss[1] = self.vfl(pred_scores, target_scores.to(dtype), target_labels) / target_scores_sum
        elif self.cls_loss == "bafocal":
            target_scores_float = target_scores.to(dtype)
            bce_loss = self.bce(pred_scores, target_scores_float)
            pred_prob = pred_scores.sigmoid()
            difficulty = (target_scores_float - pred_prob).abs().clamp(0.0, 1.0).pow(self.bafocal_gamma)
            pos_mask = target_scores_float.gt(0).to(dtype)
            focus = self.bafocal_neg_gain + (1.0 - self.bafocal_neg_gain) * pos_mask
            bce_loss *= 1.0 + self.bafocal_gain * difficulty * focus
            if self.class_weights is not None:
                bce_loss *= self.class_weights
            loss[1] = bce_loss.sum() / target_scores_sum
        elif self.cls_loss == "cbfocal":
            target_scores_float = target_scores.to(dtype)
            bce_loss = self.bce(pred_scores, target_scores_float)
            pred_prob = pred_scores.sigmoid()
            pos_mask = target_scores_float.gt(0)

            if self.class_weights is not None:
                rare = (self.class_weights - 1.0).clamp_min(0.0)
                rare_norm = rare / rare.max().clamp_min(1e-6)
                pos_weight = 1.0 + self.cbfocal_pos_gain * rare
                neg_weight = (1.0 - self.cbfocal_neg_relief * rare_norm).clamp_min(0.05)
                bce_loss *= torch.where(pos_mask, pos_weight, neg_weight)

            # Focus only positive hard samples to avoid over-amplifying the many easy background negatives.
            hard = (target_scores_float - pred_prob).abs().clamp(0.0, 1.0).pow(self.cbfocal_gamma)
            bce_loss *= torch.where(pos_mask, 1.0 + self.cbfocal_hard_gain * hard, torch.ones_like(hard))
            loss[1] = bce_loss.sum() / target_scores_sum
        elif self.cls_loss == "scb":
            loss[1] = self.scb_classification_loss(
                pred_scores, target_scores, gt_labels, target_gt_idx, fg_mask, target_scores_sum, dtype
            )
        elif self.cls_loss == "scb_logitadj":
            loss[1] = self.scb_logit_adjust_loss(pred_scores, target_scores, target_scores_sum, dtype)
        elif self.cls_loss == "scb_cals":
            loss[1] = self.scb_cals_loss(
                pred_scores, target_scores, gt_labels, target_gt_idx, fg_mask, target_scores_sum, dtype
            )
        elif self.cls_loss == "scb_qfocal":
            loss[1] = self.scb_qfocal_loss(pred_scores, target_scores, target_scores_sum, dtype)
        elif self.cls_loss == "scb_qm":
            loss[1] = self.scb_qm_loss(
                pred_scores, target_scores, gt_labels, target_gt_idx, fg_mask, target_scores_sum, dtype
            )
        elif self.cls_loss == "focal":
            bce_loss = F.binary_cross_entropy_with_logits(pred_scores, target_scores.to(dtype), reduction="none")
            pred_prob = pred_scores.sigmoid()
            p_t = target_scores.to(dtype) * pred_prob + (1 - target_scores.to(dtype)) * (1 - pred_prob)
            bce_loss *= (1.0 - p_t) ** self.focal_gamma
            if self.class_weights is not None:
                bce_loss *= self.class_weights
            loss[1] = bce_loss.sum() / target_scores_sum
        else:
            bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
            if self.class_weights is not None:
                bce_loss *= self.class_weights
            loss[1] = bce_loss.sum() / target_scores_sum  # BCE

        if self.aux_loss == "scb_qalign":
            loss[1] += self.scb_quality_alignment_loss(
                pred_scores,
                pred_bboxes,
                target_bboxes,
                target_scores,
                gt_labels,
                target_gt_idx,
                fg_mask,
                stride_tensor,
                dtype,
            )
        elif self.aux_loss != "none":
            loss[1] += self.scb_auxiliary_loss(
                pred_scores, target_scores, gt_labels, target_gt_idx, fg_mask, dtype
            )

        # Bbox loss
        if fg_mask.sum():
            self.bbox_loss.current_epoch = getattr(self, "current_epoch", 0)
            self.bbox_loss.total_epochs = getattr(self, "total_epochs", None)
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            loss.detach(),
        )  # loss(box, cls, dfl)

    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Parse model predictions to extract features."""
        return preds[1] if isinstance(preds, tuple) else preds

    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss using assigned targets."""
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach


class CBRDetectLoss(v8DetectionLoss):
    """Detection loss with training-only classroom behavior auxiliary supervision."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):
        """Initialize the classroom behavior loss on top of standard detection loss."""
        super().__init__(model, tal_topk, tal_topk2)
        self.cbr_aux = getattr(self.hyp, "cbr_aux", 0.25)
        self.cbr_proto = getattr(self.hyp, "cbr_proto", 0.1)
        self.cbr_temp = getattr(self.hyp, "cbr_temp", 0.2)
        self.cbr_margin = getattr(self.hyp, "cbr_margin", 0.25)
        self.cbr_aux_type = getattr(self.hyp, "cbr_aux_type", "margin")
        self.cbr_center_momentum = getattr(self.hyp, "cbr_center_momentum", 0.9)
        self.cbr_read_class = int(getattr(self.hyp, "scb_read_class", 1))
        self.cbr_write_class = int(getattr(self.hyp, "scb_write_class", 2))
        self.cbr_quality_thr = float(getattr(self.hyp, "scb_quality_thr", 0.0))
        self.cbr_start_epoch = int(getattr(self.hyp, "scb_start_epoch", 0))
        self.cbr_warm_epochs = max(int(getattr(self.hyp, "scb_warm_epochs", 1)), 1)
        self.cbr_class_power = float(getattr(self.hyp, "cbr_class_power", 0.25))
        self.cbr_class_cap = float(getattr(self.hyp, "cbr_class_cap", 1.5))
        self.cbr_class_counts = torch.tensor(
            [
                float(getattr(self.hyp, "scb_count_hand", 8897.0)),
                float(getattr(self.hyp, "scb_count_read", 8408.0)),
                float(getattr(self.hyp, "scb_count_write", 3078.0)),
            ],
            device=self.device,
        )
        self.cbr_level_weights = (1.0, 0.5, 0.0)
        self.cbr_centers = None
        self.cbr_center_ready = None

    def _cbr_schedule(self) -> float:
        """Ramp dataset-tailored CBR auxiliary terms after the detector has a stable decision boundary."""
        epoch = int(getattr(self, "current_epoch", 0) or 0)
        schedule = (epoch + 1 - self.cbr_start_epoch) / float(self.cbr_warm_epochs)
        return max(0.0, min(1.0, schedule))

    def _cbr_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Return mild class-frequency weights for the SCB three-class setting."""
        if self.nc != 3 or labels.numel() == 0:
            return torch.ones_like(labels, dtype=torch.float, device=self.device)
        counts = self.cbr_class_counts.to(device=self.device, dtype=torch.float).clamp_min(1.0)
        weights = (counts.max() / counts).pow(self.cbr_class_power).clamp(1.0, self.cbr_class_cap)
        return weights[labels.long().clamp(0, self.nc - 1)]

    def _legacy_aux_loss(
        self, preds: dict[str, torch.Tensor], target_scores: torch.Tensor, target_scores_sum: torch.Tensor, dtype
    ) -> torch.Tensor:
        """Compute the original fine-scale auxiliary BCE loss for ablation/backward compatibility."""
        start = 0
        aux_loss = torch.zeros(1, device=self.device)
        for i, level_scores in enumerate(preds["level_scores"]):
            num_anchors = level_scores.shape[-1]
            weight = self.cbr_level_weights[i] if i < len(self.cbr_level_weights) else 0.0
            if weight > 0:
                level_target = target_scores[:, start : start + num_anchors, :].to(dtype)
                level_loss = self.bce(level_scores.permute(0, 2, 1).contiguous(), level_target)
                if self.class_weights is not None:
                    level_loss *= self.class_weights
                aux_loss += weight * level_loss.sum()
            start += num_anchors
        return aux_loss.squeeze(0) / target_scores_sum

    def _margin_aux_loss(
        self,
        preds: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        assigned_labels: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage positive behavior anchors to separate their class logit from the hardest wrong class."""
        if self.nc <= 1 or not fg_mask.any():
            return torch.zeros((), device=self.device)

        logits = torch.cat(preds["level_scores"], dim=-1).permute(0, 2, 1).contiguous()
        pos_logits = logits[fg_mask]
        pos_labels = assigned_labels[fg_mask].long().clamp(0, self.nc - 1)
        pos_quality = target_scores.sum(-1)[fg_mask].detach().clamp_min(1e-4)

        correct_logits = pos_logits.gather(1, pos_labels.unsqueeze(1)).squeeze(1)
        wrong_logits = pos_logits.masked_fill(F.one_hot(pos_labels, self.nc).bool(), -torch.inf).amax(dim=1)
        margin_loss = F.softplus(wrong_logits - correct_logits + self.cbr_margin)
        return (margin_loss * pos_quality).sum() / pos_quality.sum().clamp_min(1.0)

    def _scb_cal_aux_loss(
        self,
        preds: dict[str, torch.Tensor],
        fg_mask: torch.Tensor,
        assigned_labels: torch.Tensor,
        target_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Dataset-aware positive calibration for SCB without suppressing confusing negative classes."""
        if self.nc <= 1 or not fg_mask.any():
            return torch.zeros((), device=self.device)

        pos_quality_all = target_scores.sum(-1).detach()
        pos_mask = fg_mask & pos_quality_all.gt(self.cbr_quality_thr)
        if not pos_mask.any():
            return torch.zeros((), device=self.device)

        logits = torch.cat(preds["level_scores"], dim=-1).permute(0, 2, 1).contiguous()
        pos_logits = logits[pos_mask]
        pos_labels = assigned_labels[pos_mask].long().clamp(0, self.nc - 1)
        pos_quality = pos_quality_all[pos_mask].clamp_min(1e-4)

        correct_logits = pos_logits.gather(1, pos_labels.unsqueeze(1)).squeeze(1)
        class_weights = self._cbr_class_weights(pos_labels)
        cal_loss = F.softplus(-correct_logits) * pos_quality * class_weights
        return cal_loss.sum() / (pos_quality * class_weights).sum().clamp_min(1.0)

    def _center_proto_loss(self, pos_features: torch.Tensor, pos_labels: torch.Tensor) -> torch.Tensor:
        """Align behavior features with class-wise EMA prototypes to provide stable cross-batch supervision."""
        if pos_features.shape[0] <= 1 or self.nc <= 1:
            return torch.zeros((), device=self.device)

        features = F.normalize(pos_features.float(), dim=-1)
        labels = pos_labels.long().clamp(0, self.nc - 1)
        feat_dim = features.shape[-1]
        if (
            self.cbr_centers is None
            or self.cbr_centers.shape != (self.nc, feat_dim)
            or self.cbr_centers.device != self.device
        ):
            self.cbr_centers = torch.zeros(self.nc, feat_dim, device=self.device, dtype=features.dtype)
            self.cbr_center_ready = torch.zeros(self.nc, device=self.device, dtype=torch.bool)

        with torch.no_grad():
            for cls_id in labels.unique(sorted=True):
                cls_mask = labels == cls_id
                cls_center = features[cls_mask].mean(0)
                idx = int(cls_id.item())
                if bool(self.cbr_center_ready[idx]):
                    updated = self.cbr_centers[idx] * self.cbr_center_momentum + cls_center * (
                        1.0 - self.cbr_center_momentum
                    )
                    self.cbr_centers[idx] = F.normalize(updated, dim=0)
                else:
                    self.cbr_centers[idx] = cls_center
                    self.cbr_center_ready[idx] = True

        valid = self.cbr_center_ready
        if valid is None or int(valid.sum().item()) <= 1:
            return torch.zeros((), device=self.device)

        centers = F.normalize(self.cbr_centers[valid], dim=-1)
        remap = torch.full((self.nc,), -1, dtype=torch.long, device=self.device)
        remap[valid] = torch.arange(int(valid.sum().item()), device=self.device)
        proto_targets = remap[labels]
        valid_pos = proto_targets.ge(0)
        if not valid_pos.any():
            return torch.zeros((), device=self.device)

        proto_logits = features[valid_pos] @ centers.T
        ce_weight = self.class_weights.view(-1)[valid].float() if self.class_weights is not None else None
        return F.cross_entropy(proto_logits / self.cbr_temp, proto_targets[valid_pos], weight=ce_weight)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate detection loss with margin-based behavior supervision and EMA prototype alignment."""
        batch_size = preds["boxes"].shape[0]
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, aux, proto
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = max(target_scores.sum(), 1)

        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        assigned_labels = (
            gt_labels.long().squeeze(-1).gather(1, target_gt_idx.clamp(min=0)) if fg_mask.any() else None
        )

        cbr_schedule = self._cbr_schedule() if self.cbr_aux_type == "scb_cal" else 1.0

        if "level_scores" in preds and self.cbr_aux > 0 and cbr_schedule > 0:
            if self.cbr_aux_type == "legacy_bce":
                loss[3] = self._legacy_aux_loss(preds, target_scores, target_scores_sum, dtype)
            elif self.cbr_aux_type == "scb_cal" and assigned_labels is not None:
                loss[3] = self._scb_cal_aux_loss(preds, fg_mask, assigned_labels, target_scores)
            elif assigned_labels is not None:
                loss[3] = self._margin_aux_loss(preds, fg_mask, assigned_labels, target_scores)

        if "behavior_feats" in preds and self.cbr_proto > 0 and assigned_labels is not None and cbr_schedule > 0:
            feature_bank = torch.cat(preds["behavior_feats"], dim=-1).permute(0, 2, 1).contiguous()
            proto_mask = fg_mask
            if self.cbr_aux_type == "scb_cal":
                proto_mask = fg_mask & target_scores.sum(-1).detach().gt(self.cbr_quality_thr)
            pos_features = feature_bank[proto_mask]
            pos_labels = assigned_labels[proto_mask]
            loss[4] = self._center_proto_loss(pos_features, pos_labels)

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.cbr_aux * cbr_schedule
        loss[4] *= self.cbr_proto * cbr_schedule
        return loss * batch_size, loss.detach()


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 segmentation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize the v8SegmentationLoss class with model parameters and mask overlap setting."""
        super().__init__(model, tal_topk, tal_topk2)
        self.overlap = model.args.overlap_mask
        self.bcedice_loss = BCEDiceLoss(weight_bce=0.5, weight_dice=0.5)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the combined loss for detection and segmentation."""
        pred_masks, proto = preds["mask_coefficient"].permute(0, 2, 1).contiguous(), preds["proto"]
        loss = torch.zeros(5, device=self.device)  # box, seg, cls, dfl, semseg
        if isinstance(proto, tuple) and len(proto) == 2:
            proto, pred_semseg = proto
        else:
            pred_semseg = None
        (fg_mask, target_gt_idx, target_bboxes, _, _), det_loss, _ = self.get_assigned_targets_and_loss(preds, batch)
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[2], loss[3] = det_loss[0], det_loss[1], det_loss[2]

        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        if fg_mask.sum():
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                # masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]
                proto = F.interpolate(proto, masks.shape[-2:], mode="bilinear", align_corners=False)

            imgsz = (
                torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_masks.dtype) * self.stride[0]
            )
            loss[1] = self.calculate_segmentation_loss(
                fg_mask,
                masks,
                target_gt_idx,
                target_bboxes,
                batch["batch_idx"].view(-1, 1),
                proto,
                pred_masks,
                imgsz,
            )
            if pred_semseg is not None:
                sem_masks = batch["sem_masks"].to(self.device)  # NxHxW
                sem_masks = F.one_hot(sem_masks.long(), num_classes=self.nc).permute(0, 3, 1, 2).float()  # NxCxHxW

                if self.overlap:
                    mask_zero = masks == 0  # NxHxW
                    sem_masks[mask_zero.unsqueeze(1).expand_as(sem_masks)] = 0
                else:
                    batch_idx = batch["batch_idx"].view(-1)  # [total_instances]
                    for i in range(batch_size):
                        instance_mask_i = masks[batch_idx == i]  # [num_instances_i, H, W]
                        if len(instance_mask_i) == 0:
                            continue
                        sem_masks[i, :, instance_mask_i.sum(dim=0) == 0] = 0

                loss[4] = self.bcedice_loss(pred_semseg, sem_masks)
                loss[4] *= self.hyp.box  # seg gain

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss
            if pred_semseg is not None:
                loss[4] += (pred_semseg * 0).sum()

        loss[1] *= self.hyp.box  # seg gain
        return loss * batch_size, loss.detach()  # loss(box, seg, cls, dfl, semseg)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (N, H, W), where N is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (N, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (N, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (N,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if self.overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int = 10):  # model must be de-paralleled
        """Initialize v8PoseLoss with model parameters and keypoint-specific loss functions."""
        super().__init__(model, tal_topk, tal_topk2)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(5, device=self.device)  # box, kpt_location, kpt_visibility, cls, dfl
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        # Pboxes
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain

        return loss * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def _select_target_keypoints(
        self,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        target_gt_idx: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Select target keypoints for each anchor based on batch index and target ground truth index.

        Args:
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).

        Returns:
            (torch.Tensor): Selected keypoints tensor, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # Vectorized fill: compute within-batch position for each keypoint using cumulative offsets
        batch_idx_long = batch_idx.long()
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx_long + 1, torch.ones_like(batch_idx_long))
        offsets = offsets.cumsum(0)
        within_idx = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx_long]
        batched_keypoints[batch_idx_long, within_idx] = keypoints

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        return selected_keypoints

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        # Select target keypoints using helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class PoseLoss26(v8PoseLoss):
    """Criterion class for computing training losses for YOLOv8 pose estimation with RLE loss support."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None):  # model must be de-paralleled
        """Initialize PoseLoss26 with model parameters and keypoint-specific loss functions including RLE loss."""
        super().__init__(model, tal_topk, tal_topk2)
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        self.rle_loss = None
        self.flow_model = model.model[-1].flow_model if hasattr(model.model[-1], "flow_model") else None
        if self.flow_model is not None:
            self.rle_loss = RLELoss(use_target_weight=True).to(self.device)
            self.target_weights = (
                torch.from_numpy(RLE_WEIGHT).to(self.device) if is_pose else torch.ones(nkpt, device=self.device)
            )

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the total loss and detach it for pose estimation."""
        pred_kpts = preds["kpts"].permute(0, 2, 1).contiguous()
        loss = torch.zeros(
            6 if self.rle_loss else 5, device=self.device
        )  # box, kpt_location, kpt_visibility, cls, dfl[, rle]
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), det_loss, _ = (
            self.get_assigned_targets_and_loss(preds, batch)
        )
        # NOTE: re-assign index for consistency for now. Need to be removed in the future.
        loss[0], loss[3], loss[4] = det_loss[0], det_loss[1], det_loss[2]

        batch_size = pred_kpts.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=pred_kpts.dtype) * self.stride[0]

        pred_kpts = pred_kpts.view(batch_size, -1, *self.kpt_shape)  # (b, h*w, 17, 3)

        if self.rle_loss and preds.get("kpts_sigma", None) is not None:
            pred_sigma = preds["kpts_sigma"].permute(0, 2, 1).contiguous()
            pred_sigma = pred_sigma.view(batch_size, -1, self.kpt_shape[0], 2)  # (b, h*w, 17, 2)
            pred_kpts = torch.cat([pred_kpts, pred_sigma], dim=-1)  # (b, h*w, 17, 5)

        pred_kpts = self.kpts_decode(anchor_points, pred_kpts)

        # Keypoint loss
        if fg_mask.sum():
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            keypoints_loss = self.calculate_keypoints_loss(
                fg_mask,
                target_gt_idx,
                keypoints,
                batch["batch_idx"].view(-1, 1),
                stride_tensor,
                target_bboxes,
                pred_kpts,
            )
            loss[1] = keypoints_loss[0]
            loss[2] = keypoints_loss[1]
            if self.rle_loss is not None:
                loss[5] = keypoints_loss[2]

        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        if self.rle_loss is not None:
            loss[5] *= self.hyp.rle  # rle gain

        return loss * batch_size, loss.detach()  # loss(box, kpt_location, kpt_visibility, cls, dfl[, rle])

    @staticmethod
    def kpts_decode(anchor_points: torch.Tensor, pred_kpts: torch.Tensor) -> torch.Tensor:
        """Decode predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., 0] += anchor_points[:, [0]]
        y[..., 1] += anchor_points[:, [1]]
        return y

    def calculate_rle_loss(self, pred_kpt: torch.Tensor, gt_kpt: torch.Tensor, kpt_mask: torch.Tensor) -> torch.Tensor:
        """Calculate the RLE (Residual Log-likelihood Estimation) loss for keypoints.

        Args:
            pred_kpt (torch.Tensor): Predicted kpts with sigma, shape (N, num_keypoints, kpts_dim) where kpts_dim >= 4.
            gt_kpt (torch.Tensor): Ground truth keypoints, shape (N, num_keypoints, kpts_dim).
            kpt_mask (torch.Tensor): Mask for valid keypoints, shape (N, num_keypoints).

        Returns:
            (torch.Tensor): The RLE loss.
        """
        pred_kpt_visible = pred_kpt[kpt_mask]
        gt_kpt_visible = gt_kpt[kpt_mask]
        pred_coords = pred_kpt_visible[:, 0:2]
        pred_sigma = pred_kpt_visible[:, -2:]
        gt_coords = gt_kpt_visible[:, 0:2]

        target_weights = self.target_weights.unsqueeze(0).repeat(kpt_mask.shape[0], 1)
        target_weights = target_weights[kpt_mask]

        pred_sigma = pred_sigma.sigmoid()
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)

        # Filter out NaN and Inf values to prevent MultivariateNormal validation errors
        valid_mask = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_kpt.device)

        error = error[valid_mask]
        error = error.clamp(-100, 100)  # Prevent numerical instability
        pred_sigma = pred_sigma[valid_mask]
        target_weights = target_weights[valid_mask]

        log_phi = self.flow_model.log_prob(error)

        return self.rle_loss(pred_sigma, log_phi, error, target_weights)

    def calculate_keypoints_loss(
        self,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        keypoints: torch.Tensor,
        batch_idx: torch.Tensor,
        stride_tensor: torch.Tensor,
        target_bboxes: torch.Tensor,
        pred_kpts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            kpts_loss (torch.Tensor): The keypoints loss.
            kpts_obj_loss (torch.Tensor): The keypoints object loss.
            rle_loss (torch.Tensor): The RLE loss.
        """
        # Select target keypoints using inherited helper method
        selected_keypoints = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)

        # Divide coordinates by stride
        selected_keypoints[..., :2] /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0
        rle_loss = 0

        if masks.any():
            target_bboxes /= stride_tensor
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if self.rle_loss is not None and (pred_kpt.shape[-1] == 4 or pred_kpt.shape[-1] == 5):
                rle_loss = self.calculate_rle_loss(pred_kpt, gt_kpt, kpt_mask)
                rle_loss = rle_loss.clamp(min=0)
            if pred_kpt.shape[-1] == 3 or pred_kpt.shape[-1] == 5:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss, rle_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses for classification."""

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the classification loss between predictions and true labels."""
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for object detection, classification, and box distribution in rotated YOLO models."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize v8OBBLoss with model, assigner, and rotated bbox loss; model must be de-paralleled."""
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)

    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        """Preprocess targets for oriented bounding box detection."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            packed_targets = targets[:, 1:].clone()
            packed_targets[:, 1:5].mul_(scale_tensor)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(len(targets), device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = packed_targets
        return out

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate and return the loss for oriented bounding box detection."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, angle
        pred_distri, pred_scores, pred_angle = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
            preds["angle"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        batch_size = pred_angle.shape[0]  # batch size

        dtype = pred_scores.dtype
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * float(imgsz[1]), targets[:, 5] * float(imgsz[0])
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolo26n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
            weight = target_scores.sum(-1)[fg_mask]
            loss[3] = self.calculate_angle_loss(
                pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum
            )  # angle loss
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        loss[3] *= self.hyp.angle  # angle gain

        return loss * batch_size, loss.detach()  # loss(box, cls, dfl, angle)

    def bbox_decode(
        self, anchor_points: torch.Tensor, pred_dist: torch.Tensor, pred_angle: torch.Tensor
    ) -> torch.Tensor:
        """Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def calculate_angle_loss(self, pred_bboxes, target_bboxes, fg_mask, weight, target_scores_sum, lambda_val=3):
        """Calculate oriented angle loss.

        Args:
            pred_bboxes (torch.Tensor): Predicted bounding boxes with shape [N, 5] (x, y, w, h, theta).
            target_bboxes (torch.Tensor): Target bounding boxes with shape [N, 5] (x, y, w, h, theta).
            fg_mask (torch.Tensor): Foreground mask indicating valid predictions.
            weight (torch.Tensor): Loss weights for each prediction.
            target_scores_sum (torch.Tensor): Sum of target scores for normalization.
            lambda_val (int): Controls the sensitivity to aspect ratio.

        Returns:
            (torch.Tensor): The calculated angle loss.
        """
        w_gt = target_bboxes[..., 2]
        h_gt = target_bboxes[..., 3]
        pred_theta = pred_bboxes[..., 4]
        target_theta = target_bboxes[..., 4]

        log_ar = torch.log((w_gt + 1e-9) / (h_gt + 1e-9))
        scale_weight = torch.exp(-(log_ar**2) / (lambda_val**2))

        delta_theta = pred_theta - target_theta
        delta_theta_wrapped = delta_theta - torch.round(delta_theta / math.pi) * math.pi
        ang_loss = torch.sin(2 * delta_theta_wrapped[fg_mask]) ** 2

        ang_loss = scale_weight[fg_mask] * ang_loss
        ang_loss = ang_loss * weight

        return ang_loss.sum() / target_scores_sum


class E2EDetectLoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model):
        """Initialize E2EDetectLoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)
        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]


class E2ELoss:
    """Criterion class for computing training losses for end-to-end detection."""

    def __init__(self, model, loss_fn=v8DetectionLoss):
        """Initialize E2ELoss with one-to-many and one-to-one detection losses using the provided model."""
        self.one2many = loss_fn(model, tal_topk=10)
        self.one2one = loss_fn(model, tal_topk=1)
        self.updates = 0

    def _sync_inner_state(self) -> None:
        """Keep wrapped one-to-many/one-to-one losses aligned with trainer state."""
        for criterion in (self.one2many, self.one2one):
            criterion.current_epoch = getattr(self, "current_epoch", 0)
            criterion.total_epochs = getattr(self, "total_epochs", None)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        self._sync_inner_state()
        preds = self.one2many.parse_output(preds)
        one2many, one2one = preds["one2many"], preds["one2one"]
        loss_one2many = self.one2many.loss(one2many, batch)
        loss_one2one = self.one2one.loss(one2one, batch)
        return loss_one2many[0] + loss_one2one[0], loss_one2many[1] + loss_one2one[1]

    def update(self) -> None:
        """Keep a compatible no-op hook for the trainer."""
        self.updates += 1


class TVPDetectLoss:
    """Criterion class for computing training losses for text-visual prompt detection."""

    def __init__(self, model, tal_topk=10, tal_topk2: int | None = None):
        """Initialize TVPDetectLoss with task-prompt and visual-prompt criteria using the provided model."""
        self.vp_criterion = v8DetectionLoss(model, tal_topk, tal_topk2)
        # NOTE: store following info as it's changeable in __call__
        self.hyp = self.vp_criterion.hyp
        self.ori_nc = self.vp_criterion.nc
        self.ori_no = self.vp_criterion.no
        self.ori_reg_max = self.vp_criterion.reg_max

    def parse_output(self, preds) -> dict[str, torch.Tensor]:
        """Parse model predictions to extract features."""
        return self.vp_criterion.parse_output(preds)

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt detection."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(3, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        box_loss = vp_loss[0][1]
        return box_loss, vp_loss[1]

    def _get_vp_features(self, preds: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        """Extract visual-prompt features from the model output."""
        scores = preds["scores"]
        vnc = scores.shape[1]

        self.vp_criterion.nc = vnc
        self.vp_criterion.no = vnc + self.vp_criterion.reg_max * 4
        self.vp_criterion.assigner.num_classes = vnc
        return scores


class TVPSegmentLoss(TVPDetectLoss):
    """Criterion class for computing training losses for text-visual prompt segmentation."""

    def __init__(self, model, tal_topk=10):
        """Initialize TVPSegmentLoss with task-prompt and visual-prompt criteria using the provided model."""
        super().__init__(model)
        self.vp_criterion = v8SegmentationLoss(model, tal_topk)
        self.hyp = self.vp_criterion.hyp

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        return self.loss(self.parse_output(preds), batch)

    def loss(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for text-visual prompt segmentation."""
        if self.ori_nc == preds["scores"].shape[1]:
            loss = torch.zeros(4, device=self.vp_criterion.device, requires_grad=True)
            return loss, loss.detach()

        preds["scores"] = self._get_vp_features(preds)
        vp_loss = self.vp_criterion(preds, batch)
        cls_loss = vp_loss[0][2]
        return cls_loss, vp_loss[1]
