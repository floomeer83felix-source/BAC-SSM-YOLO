#!/usr/bin/env python3
"""Resume the controlled final BAC-RT training run from an existing checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise FileNotFoundError(f"Invalid resume checkpoint: {checkpoint}")

    model = YOLO(str(checkpoint))
    model.train(resume=True, workers=8)


if __name__ == "__main__":
    main()
