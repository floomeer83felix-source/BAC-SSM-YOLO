#!/usr/bin/env python3
"""Train one BAC-RT candidate using the locked placement-ablation recipe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    model = YOLO(args.config)
    model.train(
        data=str(ROOT / "scb2-wsl.yaml"),
        imgsz=640,
        batch=16,
        epochs=100,
        device="cuda:0",
        workers=8,
        project=args.project,
        name=args.name,
        exist_ok=True,
        optimizer="auto",
        lr0=0.01,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
