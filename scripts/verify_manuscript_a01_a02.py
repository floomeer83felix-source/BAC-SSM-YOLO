#!/usr/bin/env python3
"""Audit the released BAC-SSM-YOLO checkpoint without changing source artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    return str(value)


def runtime_data_yaml(source: Path, outdir: Path) -> Path:
    """Translate Windows drive paths for WSL while preserving dataset semantics."""
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if os.name != "nt":
        for key in ("path", "train", "val", "test"):
            value = payload.get(key)
            if not isinstance(value, str):
                continue
            match = re.match(r"^([A-Za-z]):[/\\](.*)$", value)
            if match:
                payload[key] = f"/mnt/{match.group(1).lower()}/{match.group(2).replace(chr(92), '/')}"
    output = outdir / "runtime_data.yaml"
    output.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return output


def checkpoint_metadata(weights: Path, model: YOLO) -> dict[str, Any]:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    embedded = checkpoint.get("model") or checkpoint.get("ema")
    embedded_yaml = getattr(embedded, "yaml", None)
    modules = []
    for index, module in enumerate(model.model.model.modules()):
        modules.append(
            {
                "index": index,
                "class": f"{module.__class__.__module__}.{module.__class__.__name__}",
                "parameters": sum(parameter.numel() for parameter in module.parameters(recurse=False)),
            }
        )
    return {
        "checkpoint": str(weights),
        "sha256": sha256(weights),
        "size_bytes": weights.stat().st_size,
        "checkpoint_keys": sorted(checkpoint.keys()),
        "epoch": json_safe(checkpoint.get("epoch")),
        "best_fitness": json_safe(checkpoint.get("best_fitness")),
        "date": json_safe(checkpoint.get("date")),
        "version": json_safe(checkpoint.get("version")),
        "license": json_safe(checkpoint.get("license")),
        "docs": json_safe(checkpoint.get("docs")),
        "train_args": json_safe(checkpoint.get("train_args") or checkpoint.get("args")),
        "embedded_yaml": json_safe(embedded_yaml),
        "model_names": json_safe(model.names),
        "model_task": model.task,
        "model_end2end": bool(getattr(model.model, "end2end", False)),
        "module_structure": modules,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--outdir", default="manuscript_verification/a01_a02")
    args = parser.parse_args()

    weights = (ROOT / args.weights).resolve() if not Path(args.weights).is_absolute() else Path(args.weights).resolve()
    data = (ROOT / args.data).resolve() if not Path(args.data).is_absolute() else Path(args.data).resolve()
    outdir = (ROOT / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir).resolve()
    locked_eval = outdir / "locked_eval"
    outdir.mkdir(parents=True, exist_ok=True)
    locked_eval.mkdir(parents=True, exist_ok=True)
    if not weights.is_file() or not data.is_file():
        raise FileNotFoundError(f"Missing weights or data YAML: {weights}, {data}")

    runtime_yaml = runtime_data_yaml(data, outdir)
    model = YOLO(str(weights))
    metadata = checkpoint_metadata(weights, model)
    metadata["source_data_yaml"] = str(data)
    metadata["source_data_yaml_sha256"] = sha256(data)
    metadata["runtime_data_yaml"] = str(runtime_yaml)
    (outdir / "checkpoint_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    result = model.val(
        data=str(runtime_yaml),
        imgsz=640,
        batch=16,
        device=args.device,
        half=False,
        rect=True,
        conf=0.001,
        iou=0.7,
        augment=False,
        plots=True,
        project=str(outdir),
        name="locked_eval",
        exist_ok=True,
        verbose=True,
    )

    per_class = []
    for index, name in model.names.items():
        per_class.append(
            {
                "class_id": int(index),
                "class_name": str(name),
                "AP50": float(result.box.ap50[index]),
                "AP50_95": float(result.box.maps[index]),
            }
        )
    mean_ap50 = sum(row["AP50"] for row in per_class) / len(per_class)
    mean_ap5095 = sum(row["AP50_95"] for row in per_class) / len(per_class)
    metrics = {
        "protocol": {
            "imgsz": 640,
            "batch": 16,
            "precision": "FP32",
            "rect": True,
            "conf": 0.001,
            "iou": 0.7,
            "augment": False,
            "plots": True,
        },
        "overall": {
            "P": float(result.box.mp),
            "R": float(result.box.mr),
            "mAP50": float(result.box.map50),
            "mAP50_95": float(result.box.map),
        },
        "per_class": per_class,
        "per_class_means": {"mAP50": mean_ap50, "mAP50_95": mean_ap5095},
        "consistency": {
            "mAP50_absolute_error": abs(mean_ap50 - float(result.box.map50)),
            "mAP50_95_absolute_error": abs(mean_ap5095 - float(result.box.map)),
            "floating_point_consistent": (
                abs(mean_ap50 - float(result.box.map50)) <= 1e-12
                and abs(mean_ap5095 - float(result.box.map)) <= 1e-12
            ),
        },
        "save_dir": str(result.save_dir),
    }
    (outdir / "metrics_locked_eval.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(outdir / "overall_metrics.csv", [{"metric": key, "value": value} for key, value in metrics["overall"].items()])
    write_csv(outdir / "per_class_metrics.csv", per_class)

    aliases = {
        "BoxPR_curve.png": "PR_curve.png",
        "BoxF1_curve.png": "F1_curve.png",
        "BoxP_curve.png": "P_curve.png",
        "BoxR_curve.png": "R_curve.png",
    }
    for source_name, target_name in aliases.items():
        source = locked_eval / source_name
        if source.is_file():
            shutil.copy2(source, locked_eval / target_name)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
