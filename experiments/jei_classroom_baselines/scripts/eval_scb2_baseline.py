#!/usr/bin/env python3
"""Evaluate a classroom baseline with the locked SCB-Dataset2 protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model = YOLO(args.weights)
    result = model.val(
        data=args.data,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        half=False,
        rect=True,
        conf=0.001,
        iou=0.7,
        plots=False,
        verbose=True,
    )
    payload = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "protocol": {"imgsz": 640, "fp32": True, "rect": True, "conf": 0.001, "iou": 0.7, "batch": args.batch},
        "P": float(result.box.mp),
        "R": float(result.box.mr),
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
        "per_class": {},
    }
    ap50 = getattr(result.box, "ap50", None)
    maps = getattr(result.box, "maps", None)
    if ap50 is not None and maps is not None:
        for i, name in model.names.items():
            payload["per_class"][str(name)] = {"AP50": float(ap50[i]), "AP50_95": float(maps[i])}

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
