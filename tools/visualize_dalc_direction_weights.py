from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


if sys.platform.startswith("linux"):
    DATASET = Path("/mnt/d/datasets/SCB/scb2")
else:
    DATASET = Path(r"D:\datasets\SCB\scb2")

DEFAULT_WEIGHTS = ROOT / "runs/detect/ss2d-improvement-scb2-mamba/yolo26-da-lc-ss2d-scbconv-ca/weights/best.pt"
DEFAULT_OUT = ROOT / "the_visual_computer_submission_project/paper_artifacts/dalc_direction_weights"

DIRECTIONS = ["LR", "RL", "TB", "BT"]
SELECTED = [
    ("hand-raising", "1292004"),
    ("reading", "0014029"),
    ("writing", "1433016"),
]


def image_path(stem: str) -> Path:
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        path = DATASET / "images" / "val" / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Image not found for stem {stem}")


def collect_weights(model: YOLO, image: Path, imgsz: int) -> np.ndarray:
    collected: list[torch.Tensor] = []
    handles = []

    def hook(_module, _inputs, output):
        logits = output.flatten(1)
        weights = torch.softmax(logits, dim=1).detach().float().cpu()
        collected.append(weights)

    for module in model.model.modules():
        if module.__class__.__name__ == "SS2DDALC" and hasattr(module, "direction_mlp"):
            handles.append(module.direction_mlp.register_forward_hook(hook))

    try:
        model.predict(str(image), imgsz=imgsz, conf=0.25, iou=0.7, verbose=False)
    finally:
        for handle in handles:
            handle.remove()

    if not collected:
        raise RuntimeError("No DA-LC direction weights were captured. Check whether the model contains SS2DDALC modules.")
    stacked = torch.cat(collected, dim=0)
    return stacked.mean(dim=0).numpy()


def make_figure(rows: list[tuple[str, Path, np.ndarray]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = np.stack([r[2] for r in rows], axis=0)

    fig = plt.figure(figsize=(10.8, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, len(rows), height_ratios=[1.15, 1.0])

    for idx, (label, path, _w) in enumerate(rows):
        ax = fig.add_subplot(gs[0, idx])
        img = Image.open(path).convert("RGB")
        img.thumbnail((420, 260), Image.Resampling.LANCZOS)
        ax.imshow(img)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=6)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(weights, cmap="YlGnBu", vmin=0.0, vmax=max(0.45, float(weights.max()) + 0.03), aspect="auto")
    ax.set_xticks(np.arange(len(DIRECTIONS)), DIRECTIONS, fontsize=12)
    ax.set_yticks(np.arange(len(rows)), [r[0] for r in rows], fontsize=12)
    ax.set_xlabel("Directional scan branch", fontsize=12)
    ax.set_title("Image-wise DA-LC direction weights", fontsize=13, fontweight="bold", pad=8)

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            ax.text(j, i, f"{weights[i, j]:.3f}", ha="center", va="center", fontsize=11, color="black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("Softmax weight", rotation=90, fontsize=11)

    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_dir / f"dalc_direction_weights.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize DA-LC-SS2D direction-adaptive weights.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    rows = []
    for label, stem in SELECTED:
        path = image_path(stem)
        weights = collect_weights(model, path, args.imgsz)
        rows.append((label, path, weights))

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "dalc_direction_weights.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_hint", *DIRECTIONS])
        for label, path, weights in rows:
            writer.writerow([path.name, label, *[f"{float(x):.6f}" for x in weights]])

    make_figure(rows, args.out)
    print(f"Saved DA-LC direction-weight visualization to {args.out}")


if __name__ == "__main__":
    main()
