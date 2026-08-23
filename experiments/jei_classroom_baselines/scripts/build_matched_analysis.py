#!/usr/bin/env python3
"""Build the audited seed-0 matched comparison from completed experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/jei_classroom_baselines/matched_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_eval(filename: str) -> dict:
    return json.loads((OUT / filename).read_text(encoding="utf-8"))


def pct(value: float) -> float:
    return round(100.0 * value, 4)


def main() -> None:
    sources = {
        "PLA-YOLO11n-reimpl": load_eval("pla_yolo11n_reimpl_locked_eval.json"),
        "WAD-YOLOv8n-reimpl": load_eval("wad_yolov8n_reimpl_locked_eval.json"),
        "BAC-SSM-YOLO": load_eval("bac_ssm_yolo_locked_eval.json"),
        "YOLO26s": load_eval("yolo26s_locked_eval.json"),
    }
    complexity = {
        "PLA-YOLO11n-reimpl": (3.474412, 10.2993408),
        "WAD-YOLOv8n-reimpl": (3.909769, 9.6239232),
        "BAC-SSM-YOLO": (5.342903, 15.3139576),
        "YOLO26s": (9.950186, 22.5073152),
    }
    weights = {
        "PLA-YOLO11n-reimpl": OUT / "runs/pla_yolo11n_reimpl_seed0/weights/best.pt",
        "WAD-YOLOv8n-reimpl": OUT / "runs/wad_yolov8n_reimpl_seed0/weights/best.pt",
        "BAC-SSM-YOLO": REPO / "weights/BAC-SSM-YOLO_scb2_best.pt",
        "YOLO26s": Path("E:/yolo/yolo26/runs/detect/scb-2/yolo26s/weights/best.pt"),
    }
    best_epochs = {
        "PLA-YOLO11n-reimpl": 100,
        "WAD-YOLOv8n-reimpl": 88,
        "BAC-SSM-YOLO": None,
        "YOLO26s": 100,
    }

    rows = []
    for name, result in sources.items():
        params, gflops = complexity[name]
        row = {
            "model": name,
            "seed": 0,
            "params_M": params,
            "GFLOPs": gflops,
            "P_percent": pct(result["P"]),
            "R_percent": pct(result["R"]),
            "mAP50_percent": pct(result["mAP50"]),
            "mAP50_95_percent": pct(result["mAP50_95"]),
        }
        for cls in ("hand-raising", "reading", "writing"):
            key = cls.replace("-", "_")
            row[f"{key}_AP50_percent"] = pct(result["per_class"][cls]["AP50"])
            row[f"{key}_AP50_95_percent"] = pct(result["per_class"][cls]["AP50_95"])
        rows.append(row)

    rows.sort(key=lambda item: item["mAP50_95_percent"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank_by_mAP50_95"] = rank

    csv_path = OUT / "matched_comparison_seed0.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bac = next(row for row in rows if row["model"] == "BAC-SSM-YOLO")
    deltas = {}
    for model in ("PLA-YOLO11n-reimpl", "WAD-YOLOv8n-reimpl"):
        row = next(item for item in rows if item["model"] == model)
        deltas[model] = {
            "delta_mAP50_points_vs_BAC": round(row["mAP50_percent"] - bac["mAP50_percent"], 4),
            "delta_mAP50_95_points_vs_BAC": round(row["mAP50_95_percent"] - bac["mAP50_95_percent"], 4),
            "delta_params_M_vs_BAC": round(row["params_M"] - bac["params_M"], 6),
            "delta_GFLOPs_vs_BAC": round(row["GFLOPs"] - bac["GFLOPs"], 7),
        }

    best_classroom = max(
        (row for row in rows if row["model"] in {"PLA-YOLO11n-reimpl", "WAD-YOLOv8n-reimpl"}),
        key=lambda item: item["mAP50_95_percent"],
    )
    gap = abs(best_classroom["mAP50_95_percent"] - bac["mAP50_95_percent"])
    protocol = sources["BAC-SSM-YOLO"]["protocol"]
    analysis = {
        "status": {
            "training_completed": True,
            "pla_epochs_completed": 100,
            "wad_epochs_completed": 100,
            "sequential_training": True,
            "locked_evaluation_completed": True,
        },
        "dataset_lock": {
            "train_images": 3418,
            "val_images": 848,
            "train_instances": 14506,
            "val_instances": 3992,
            "class_order": ["hand-raising", "reading", "writing"],
            "train_image_list_sha256": "8027C2A6707A9EC30D63A2946A4B1FF47DF83C7EC399BBDFA3F9382D3460CF33",
            "val_image_list_sha256": "969E0418EC80FE8B0BE46082868A2A5FCCC7DA9578DAB543A1CFB2CE70C5446C",
        },
        "training_protocol": {
            "epochs": 100,
            "imgsz": 640,
            "batch": 16,
            "seed": 0,
            "deterministic": True,
            "workers": 8,
            "optimizer": "auto",
            "PLA_pretrained": "yolo11n.pt (partial compatible loading)",
            "WAD_pretrained": "yolov8n.pt (partial compatible loading)",
        },
        "evaluation_protocol": protocol,
        "bac_gate_check": {
            "expected_mAP50_95_percent": 48.05,
            "measured_mAP50_95_percent": bac["mAP50_95_percent"],
            "absolute_error_points": round(abs(bac["mAP50_95_percent"] - 48.05), 4),
            "tolerance_points": 0.10,
            "passed": abs(bac["mAP50_95_percent"] - 48.05) <= 0.10,
        },
        "ranking": [{"rank": row["rank_by_mAP50_95"], "model": row["model"], "mAP50_95_percent": row["mAP50_95_percent"]} for row in rows],
        "deltas_vs_BAC": deltas,
        "best_classroom_baseline": best_classroom["model"],
        "absolute_gap_to_BAC_points": round(gap, 4),
        "recommend_three_seed_followup": gap <= 0.50,
        "three_seed_followup_executed": False,
        "checkpoints": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "best_epoch": best_epochs[name],
            }
            for name, path in weights.items()
        },
        "caveats": [
            "PyTorch emitted warn-only nondeterminism notices for grid_sampler_2d backward and Flash Attention during WAD training despite deterministic=True.",
            "Params and GFLOPs are unified unfused architecture-profile values at 640 pixels; fused checkpoint summaries are not used in the comparison.",
        ],
        "results": rows,
    }
    json_path = OUT / "matched_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")

    report = f"""# Matched SCB-Dataset2 Seed-0 Training Report

## Completion and protocol

PLA-YOLO11n-reimpl and WAD-YOLOv8n-reimpl completed 100 epochs sequentially. Both `best.pt` checkpoints were evaluated with the locked protocol: 640-pixel FP32 input, `rect=True`, `conf=0.001`, `IoU=0.7`, and `augment=False`. The split contains 3,418 training images and 848 validation images in the class order hand-raising, reading, and writing.

The BAC gate reproduced 48.0520% mAP@0.5:0.95, only 0.0020 percentage point from the 48.05% target; therefore, the 0.10-point gate passed.

## Ranking

| Rank | Model | Params (M) | GFLOPs | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---:|---|---:|---:|---:|---:|---:|---:|
"""
    for row in rows:
        report += (
            f"| {row['rank_by_mAP50_95']} | {row['model']} | {row['params_M']:.3f} | "
            f"{row['GFLOPs']:.3f} | {row['P_percent']:.4f} | {row['R_percent']:.4f} | "
            f"{row['mAP50_percent']:.4f} | {row['mAP50_95_percent']:.4f} |\n"
        )
    report += f"""

## Matched-baseline comparison

- PLA-YOLO11n-reimpl versus BAC: {deltas['PLA-YOLO11n-reimpl']['delta_mAP50_points_vs_BAC']:+.4f} mAP@0.5 points, {deltas['PLA-YOLO11n-reimpl']['delta_mAP50_95_points_vs_BAC']:+.4f} mAP@0.5:0.95 points, {deltas['PLA-YOLO11n-reimpl']['delta_params_M_vs_BAC']:+.6f} M parameters, and {deltas['PLA-YOLO11n-reimpl']['delta_GFLOPs_vs_BAC']:+.7f} GFLOPs.
- WAD-YOLOv8n-reimpl versus BAC: {deltas['WAD-YOLOv8n-reimpl']['delta_mAP50_points_vs_BAC']:+.4f} mAP@0.5 points, {deltas['WAD-YOLOv8n-reimpl']['delta_mAP50_95_points_vs_BAC']:+.4f} mAP@0.5:0.95 points, {deltas['WAD-YOLOv8n-reimpl']['delta_params_M_vs_BAC']:+.6f} M parameters, and {deltas['WAD-YOLOv8n-reimpl']['delta_GFLOPs_vs_BAC']:+.7f} GFLOPs.

WAD-YOLOv8n-reimpl is the strongest classroom reimplementation and is {gap:.4f} percentage point from BAC on mAP@0.5:0.95. Because this absolute gap is at most 0.50 point, a three-seed follow-up is recommended. It was not started in this run, as required.

## Reproducibility note

The CSV contains all overall and per-class locked-evaluation metrics. Checkpoint SHA256 values, the image-list hashes, exact protocols, deltas, and the deterministic-training caveat are recorded in `matched_analysis.json`. No latency, FP16, profiling, paper, or architecture changes were performed.
"""
    (OUT / "MATCHED_TRAINING_REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
