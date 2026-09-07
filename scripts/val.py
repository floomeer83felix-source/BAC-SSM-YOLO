from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BAC-SSM-YOLO with the locked SCB-Dataset2 protocol.")
    parser.add_argument("--weights", default=str(ROOT / "weights/BAC-SSM-YOLO_scb2_best.pt"))
    parser.add_argument("--data", default=str(ROOT / "configs/datasets/scb2.yaml"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        rect=True,
        conf=0.001,
        iou=0.7,
        augment=False,
        half=False,
    )


if __name__ == "__main__":
    main()
