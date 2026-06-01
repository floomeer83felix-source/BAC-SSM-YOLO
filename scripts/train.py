from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BAC-SSM-YOLO.")
    parser.add_argument("--model", default=str(ROOT / "configs/models/yolo26-da-lc-ss2d-scbconv-ca.yaml"))
    parser.add_argument("--data", default=str(ROOT / "configs/datasets/scb2.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(ROOT / "runs/train"))
    parser.add_argument("--name", default="BAC-SSM-YOLO")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "optimizer": args.optimizer,
    }
    train_kwargs["lr0"] = args.lr0
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
