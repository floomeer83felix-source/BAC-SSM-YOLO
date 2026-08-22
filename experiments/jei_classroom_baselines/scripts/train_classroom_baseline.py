#!/usr/bin/env python3
"""Train a paper-faithful PLA-YOLO11n or WAD-YOLOv8n reimplementation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402

CONFIGS = {
    "pla": EXPERIMENT_ROOT / "configs/pla_yolo11n_scb2.yaml",
    "wad": EXPERIMENT_ROOT / "configs/wad_yolov8n_scb2.yaml",
}
PRETRAINED = {"pla": "yolo11n.pt", "wad": "yolov8n.pt"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pla", "wad"], required=True)
    parser.add_argument("--recipe", choices=["matched", "paper"], default="matched")
    parser.add_argument("--data", required=True, help="Use the exact SCB2 split used by BAC-SSM-YOLO")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/jei_classroom_baselines")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--optimizer", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    model = YOLO(str(CONFIGS[args.model]), task="detect")
    if not args.no_pretrained:
        print(f"[INFO] Partially loading compatible weights from {PRETRAINED[args.model]}")
        model.load(PRETRAINED[args.model])

    if args.recipe == "matched":
        epochs = args.epochs or 100
        batch = args.batch or 16
        optimizer = args.optimizer or "auto"
        extra = {}
    elif args.model == "pla":
        epochs = args.epochs or 400
        batch = args.batch or 8
        optimizer = args.optimizer or "SGD"
        extra = {"lr0": 0.01, "momentum": 0.937, "weight_decay": 5e-4, "warmup_epochs": 3.0, "mosaic": 1.0}
    else:
        epochs = args.epochs or 200
        batch = args.batch or 4
        optimizer = args.optimizer or "SGD"
        extra = {"lr0": 0.01, "momentum": 0.937, "weight_decay": 5e-4}

    name = args.name or f"{args.model}_{args.recipe}_seed{args.seed}"
    print("[INFO] Paper-faithful reimplementation; not official author source code.")
    print(f"[INFO] model={args.model}, recipe={args.recipe}, epochs={epochs}, batch={batch}, optimizer={optimizer}")
    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        workers=args.workers,
        optimizer=optimizer,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        project=args.project,
        name=name,
        **extra,
    )


if __name__ == "__main__":
    main()
