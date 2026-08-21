#!/usr/bin/env python3
"""Evaluate one BAC-RT checkpoint with the reproduced manuscript protocol."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO, __version__ as ultralytics_version  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

from end_to_end_latency import file_sha256, get_driver_version, get_git_commit, restore_legacy_dalc_flags  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--scan-stages", required=True)
    parser.add_argument("--num-dalc-blocks", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    weights = Path(args.weights).resolve()
    data = Path(args.data).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    restored = restore_legacy_dalc_flags(model.model)
    params_m = sum(parameter.numel() for parameter in model.model.parameters()) / 1e6
    gflops = float(get_flops(model.model, imgsz=640))
    metrics = model.val(
        data=str(data),
        split="val",
        imgsz=640,
        batch=args.batch,
        rect=True,
        conf=None,
        iou=0.7,
        augment=False,
        half=False,
        device=args.device,
        workers=args.workers,
        plots=False,
        save_json=False,
        verbose=True,
        project=str(outdir / "ultralytics"),
        name="val",
        exist_ok=True,
    )
    per_class = {}
    for index, class_name in metrics.names.items():
        values = metrics.box.class_result(index)
        per_class[class_name] = {
            "precision": float(values[0]),
            "recall": float(values[1]),
            "ap50": float(values[2]),
            "ap50_95": float(values[3]),
        }
    summary = {
        "model": args.model_name,
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "dataset": str(data),
        "protocol": {
            "split": "val",
            "imgsz": 640,
            "batch": args.batch,
            "rect": True,
            "conf": 0.001,
            "iou": 0.7,
            "augment": False,
            "half": False,
            "device": args.device,
        },
        "seed": args.seed,
        "params_m": params_m,
        "gflops_640": gflops,
        "scan_stages": args.scan_stages,
        "num_DA_LC_blocks": args.num_dalc_blocks,
        "metrics": {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        },
        "per_class": per_class,
        "legacy_dalc_compatibility": restored,
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "driver": get_driver_version(),
            "cuda": torch.version.cuda,
            "pytorch": torch.__version__,
            "ultralytics": ultralytics_version,
            "python": platform.python_version(),
            "git_commit": get_git_commit(ROOT),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    target = outdir / "accuracy_summary.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
