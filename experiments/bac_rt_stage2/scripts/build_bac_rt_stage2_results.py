#!/usr/bin/env python3
"""Build the stage-2 BAC real-time comparison from measured artifacts."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/bac_rt_stage2"
REFERENCE_CSV = ROOT / "runs/bac_rt/bac_rt_comparison.csv"
FIELDNAMES = [
    "model", "seed", "scan_stages", "num_DA_LC_blocks", "params_m", "gflops", "P", "R", "mAP50",
    "mAP50_95", "mean_latency_ms", "latency_sd_ms", "P95_ms", "P99_ms", "processing_fps",
    "deadline_miss_rate", "inference_ms", "peak_memory_allocated_mib", "peak_memory_reserved_mib",
]
CANDIDATES = {
    "BAC-SSM-YOLO-RT-NoP2": RUN_ROOT / "nop2",
    "BAC-SSM-YOLO-RT-NoEarly": RUN_ROOT / "noearly",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def reference_rows() -> list[dict]:
    wanted = {"YOLO26s", "BAC-SSM-YOLO full", "BAC-SSM-YOLO-RT-P45"}
    with REFERENCE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [{key: number(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    selected = [row for row in rows if row["model"] in wanted]
    if {row["model"] for row in selected} != wanted:
        raise RuntimeError("Missing required stage-1 references")
    return selected


def aggregate_latency(base: Path) -> dict:
    summaries = [load(base / f"run{index}/latency_summary.json") for index in (1, 2, 3)]
    hashes = {item["image_list_sha256"] for item in summaries}
    protocols = {
        (item["precision"], item["batch_size"], item["imgsz"], item["rect"], item["conf"], item["iou"])
        for item in summaries
    }
    if len(hashes) != 1 or len(protocols) != 1:
        raise RuntimeError(f"Latency protocol mismatch for {base}")
    means = [item["processing_total_ms"]["mean"] for item in summaries]
    misses = sum(item["soft_deadline"]["deadline_miss_count"] for item in summaries)
    images = sum(item["num_images"] for item in summaries)
    return {
        "mean_latency_ms": statistics.mean(means),
        "latency_sd_ms": statistics.stdev(means),
        "P95_ms": statistics.mean(item["processing_total_ms"]["p95"] for item in summaries),
        "P99_ms": statistics.mean(item["processing_total_ms"]["p99"] for item in summaries),
        "processing_fps": 1000.0 / statistics.mean(means),
        "deadline_miss_rate": misses / images * 100.0,
        "inference_ms": statistics.mean(item["stage_latency_ms"]["inference_ms"]["mean"] for item in summaries),
        "peak_memory_allocated_mib": statistics.mean(item["peak_gpu_memory_allocated_mib"] for item in summaries),
        "peak_memory_reserved_mib": statistics.mean(item["peak_gpu_memory_reserved_mib"] for item in summaries),
        "image_list_sha256": hashes.pop(),
    }


def candidate_row(name: str, base: Path) -> tuple[dict, dict]:
    accuracy = load(base / "accuracy/accuracy_summary.json")
    latency = aggregate_latency(base / "latency")
    metrics = accuracy["metrics"]
    row = {
        "model": name,
        "seed": accuracy["seed"],
        "scan_stages": accuracy["scan_stages"],
        "num_DA_LC_blocks": accuracy["num_DA_LC_blocks"],
        "params_m": accuracy["params_m"],
        "gflops": accuracy["gflops_640"],
        "P": metrics["precision"],
        "R": metrics["recall"],
        "mAP50": metrics["map50"],
        "mAP50_95": metrics["map50_95"],
        **{key: latency[key] for key in FIELDNAMES if key in latency},
    }
    return row, accuracy["per_class"]


def tier(row: dict) -> str:
    if row["mAP50_95"] >= 0.4760 and row["mean_latency_ms"] <= 14.5:
        return "A"
    if row["mAP50_95"] >= 0.4740 and row["mean_latency_ms"] <= 15.0:
        return "B"
    return "C"


def pareto(rows: list[dict]) -> list[str]:
    return [
        row["model"]
        for row in rows
        if not any(
            other["mean_latency_ms"] <= row["mean_latency_ms"]
            and other["mAP50_95"] >= row["mAP50_95"]
            and (other["mean_latency_ms"] < row["mean_latency_ms"] or other["mAP50_95"] > row["mAP50_95"])
            for other in rows
            if other is not row
        )
    ]


def deltas(row: dict, reference: dict) -> dict:
    return {
        "ap_difference_points": (row["mAP50_95"] - reference["mAP50_95"]) * 100.0,
        "latency_difference_ms": row["mean_latency_ms"] - reference["mean_latency_ms"],
        "latency_reduction_percent": (1.0 - row["mean_latency_ms"] / reference["mean_latency_ms"]) * 100.0,
        "fps_gain_percent": (row["processing_fps"] / reference["processing_fps"] - 1.0) * 100.0,
        "memory_reduction_percent": (
            1.0 - row["peak_memory_allocated_mib"] / reference["peak_memory_allocated_mib"]
        ) * 100.0,
    }


def main() -> None:
    rows = reference_rows()
    per_class = {}
    candidate_rows = []
    for name, base in CANDIDATES.items():
        row, class_metrics = candidate_row(name, base)
        rows.append(row)
        candidate_rows.append(row)
        per_class[name] = class_metrics

    with (RUN_ROOT / "bac_rt_stage2_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    full = next(row for row in rows if row["model"] == "BAC-SSM-YOLO full")
    p45 = next(row for row in rows if row["model"] == "BAC-SSM-YOLO-RT-P45")
    yolo = next(row for row in rows if row["model"] == "YOLO26s")
    front = pareto(rows)
    analysis = {}
    qualifying = []
    for row in candidate_rows:
        grade = tier(row)
        item = {
            "tier": grade,
            "relative_to_full": deltas(row, full),
            "relative_to_rt_p45": deltas(row, p45),
            "relative_to_yolo26s": deltas(row, yolo),
            "pareto_optimal": row["model"] in front,
        }
        analysis[row["model"]] = item
        if grade in {"A", "B"}:
            qualifying.append(row)

    rank = {"A": 2, "B": 1, "C": 0}
    best = None
    if qualifying:
        best = max(
            qualifying,
            key=lambda row: (
                rank[tier(row)],
                row["model"] in front,
                row["mAP50_95"] / row["mean_latency_ms"],
            ),
        )["model"]
    report = {
        "reference_source": str(REFERENCE_CSV),
        "rows": rows,
        "per_class": per_class,
        "candidate_analysis": analysis,
        "pareto_front": front,
        "best_qualifying_candidate": best,
        "tier_thresholds": {
            "A": {"mAP50_95_min": 0.4760, "mean_latency_ms_max": 14.5},
            "B": {"mAP50_95_min": 0.4740, "mean_latency_ms_max": 15.0},
        },
        "all_values_finite": all(
            math.isfinite(float(row[key]))
            for row in rows
            for key in FIELDNAMES
            if key not in {"model", "scan_stages"}
        ),
    }
    (RUN_ROOT / "stage2_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
