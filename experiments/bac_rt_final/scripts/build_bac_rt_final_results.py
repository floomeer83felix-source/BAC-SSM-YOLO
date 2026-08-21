#!/usr/bin/env python3
"""Build the final ConvEarly comparison and screening report from measured artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/bac_rt_final"
STAGE2 = ROOT / "runs/bac_rt_stage2"
FIELDS = [
    "model", "seed", "scan_stages", "num_DA_LC_blocks", "num_standard_SS2D_blocks",
    "num_early_conv_blocks", "params_m", "gflops", "P", "R", "mAP50", "mAP50_95",
    "mean_latency_ms", "latency_sd_ms", "P50_ms", "P90_ms", "P95_ms", "P99_ms", "processing_fps",
    "deadline_miss_rate", "inference_ms", "peak_memory_allocated_mib", "peak_memory_reserved_mib",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse(value: str):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def references() -> list[dict]:
    wanted = {"YOLO26s", "BAC-SSM-YOLO full", "BAC-SSM-YOLO-RT-P45", "BAC-SSM-YOLO-RT-NoEarly"}
    with (STAGE2 / "bac_rt_stage2_comparison.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [{key: parse(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    selected = [row for row in rows if row["model"] in wanted]
    if {row["model"] for row in selected} != wanted:
        raise RuntimeError("Required references are incomplete")
    for row in selected:
        row["num_standard_SS2D_blocks"] = 0 if row["model"] in {"YOLO26s", "BAC-SSM-YOLO full"} else "not_reported"
        row["num_early_conv_blocks"] = 4 if row["model"] == "YOLO26s" else 0
        row.setdefault("P50_ms", "")
        row.setdefault("P90_ms", "")
    return selected


def aggregate_latency() -> dict:
    summaries = [load(OUT / f"latency/run{index}/latency_summary.json") for index in (1, 2, 3)]
    if len({item["image_list_sha256"] for item in summaries}) != 1:
        raise RuntimeError("Latency image-list hashes differ")
    protocols = {
        (item["precision"], item["batch_size"], item["imgsz"], item["rect"], item["conf"], item["iou"])
        for item in summaries
    }
    if protocols != {("FP32", 1, 640, False, 0.25, 0.7)}:
        raise RuntimeError(f"Unexpected latency protocol: {protocols}")
    means = [item["processing_total_ms"]["mean"] for item in summaries]
    total_images = sum(item["num_images"] for item in summaries)
    misses = sum(item["soft_deadline"]["deadline_miss_count"] for item in summaries)
    return {
        "mean_latency_ms": statistics.mean(means),
        "latency_sd_ms": statistics.stdev(means),
        "P50_ms": statistics.mean(item["processing_total_ms"]["p50"] for item in summaries),
        "P90_ms": statistics.mean(item["processing_total_ms"]["p90"] for item in summaries),
        "P95_ms": statistics.mean(item["processing_total_ms"]["p95"] for item in summaries),
        "P99_ms": statistics.mean(item["processing_total_ms"]["p99"] for item in summaries),
        "processing_fps": 1000.0 / statistics.mean(means),
        "deadline_miss_rate": misses / total_images * 100.0,
        "inference_ms": statistics.mean(item["stage_latency_ms"]["inference_ms"]["mean"] for item in summaries),
        "peak_memory_allocated_mib": statistics.mean(item["peak_gpu_memory_allocated_mib"] for item in summaries),
        "peak_memory_reserved_mib": statistics.mean(item["peak_gpu_memory_reserved_mib"] for item in summaries),
        "image_list_sha256": summaries[0]["image_list_sha256"],
    }


def candidate() -> tuple[dict, dict]:
    accuracy = load(OUT / "accuracy/accuracy_summary.json")
    metrics = accuracy["metrics"]
    latency = aggregate_latency()
    row = {
        "model": "BAC-SSM-YOLO-RT-ConvEarly",
        "seed": 0,
        "scan_stages": "P4/C4,P5/C5,P5-context,P4-neck",
        "num_DA_LC_blocks": 4,
        "num_standard_SS2D_blocks": 0,
        "num_early_conv_blocks": 2,
        "params_m": accuracy["params_m"],
        "gflops": accuracy["gflops_640"],
        "P": metrics["precision"],
        "R": metrics["recall"],
        "mAP50": metrics["map50"],
        "mAP50_95": metrics["map50_95"],
        **{key: value for key, value in latency.items() if key in FIELDS},
    }
    return row, accuracy["per_class"]


def reduction(candidate_row: dict, reference: dict) -> dict:
    return {
        "ap_difference_points": (candidate_row["mAP50_95"] - reference["mAP50_95"]) * 100.0,
        "latency_reduction_ms": reference["mean_latency_ms"] - candidate_row["mean_latency_ms"],
        "latency_reduction_percent": (1.0 - candidate_row["mean_latency_ms"] / reference["mean_latency_ms"]) * 100.0,
        "fps_increase": candidate_row["processing_fps"] - reference["processing_fps"],
        "fps_increase_percent": (candidate_row["processing_fps"] / reference["processing_fps"] - 1.0) * 100.0,
        "P95_reduction_ms": reference["P95_ms"] - candidate_row["P95_ms"],
        "P99_reduction_ms": reference["P99_ms"] - candidate_row["P99_ms"],
        "deadline_miss_reduction_points": reference["deadline_miss_rate"] - candidate_row["deadline_miss_rate"],
        "memory_reduction_mib": reference["peak_memory_allocated_mib"] - candidate_row["peak_memory_allocated_mib"],
        "memory_reduction_percent": (
            1.0 - candidate_row["peak_memory_allocated_mib"] / reference["peak_memory_allocated_mib"]
        ) * 100.0,
    }


def main() -> None:
    rows = references()
    conv, per_class = candidate()
    rows.append(conv)
    with (OUT / "bac_rt_final_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_name = {row["model"]: row for row in rows}
    noearly = by_name["BAC-SSM-YOLO-RT-NoEarly"]
    full = by_name["BAC-SSM-YOLO full"]
    yolo = by_name["YOLO26s"]
    labels = {
        "strong": conv["mAP50_95"] >= 0.4740 and conv["mean_latency_ms"] <= 14.5,
        "very_useful": conv["mAP50_95"] >= 0.4720 and conv["mean_latency_ms"] <= 14.0,
        "exceptional": conv["mAP50_95"] >= yolo["mAP50_95"] and conv["mean_latency_ms"] <= 13.0,
    }
    trigger = conv["mAP50_95"] >= 0.4720 and conv["mean_latency_ms"] <= 14.5
    noearly_delta = reduction(conv, noearly)
    if noearly_delta["latency_reduction_percent"] >= 5.0:
        interpretation = (
            "ConvEarly is measurably faster than NoEarly under the locked protocol, supporting the conclusion that "
            "the early SS2D core, rather than only DA/LC, contributes materially to runtime overhead."
        )
    else:
        interpretation = "Removing early SS2D did not materially improve measured latency under the evaluated implementation."
    report = {
        "architecture": {
            "P2": "native YOLO26 C3k2 [256, False, 0.25]",
            "P3": "native YOLO26 C3k2 [512, False, 0.25]",
            "P4": "C3k2DALCSS2D",
            "P5": "C3k2DALCSS2D",
            "P5_context": "C2PSADALCSS2D",
            "P4_neck": "C3k2DALCSS2D",
        },
        "candidate": conv,
        "per_class": per_class,
        "NoEarly_vs_ConvEarly": noearly_delta,
        "full_BAC_vs_ConvEarly": reduction(conv, full),
        "YOLO26s_vs_ConvEarly": reduction(conv, yolo),
        "screening": {**labels, "follow_up_trigger": trigger},
        "scientific_interpretation": interpretation,
        "three_seed_triggered": trigger,
        "profiler_triggered": trigger,
        "further_architecture_search_recommended": False,
        "paper_modified": False,
    }
    seed_path = OUT / "accuracy/seed_stability.json"
    profile_path = OUT / "profiler/profiler_comparison.json"
    if seed_path.is_file():
        report["seed_stability"] = load(seed_path)
    if profile_path.is_file():
        report["profiler_comparison"] = load(profile_path)
    (OUT / "final_analysis.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    readme = f"""# BAC-SSM-YOLO Final Controlled Real-Time Experiment

ConvEarly restores the exact native YOLO26 convolutional blocks at P2 and P3 and retains DA-LC-SS2D at P4, P5, P5 context, and P4 neck. No other full-model layer changes.

## Result

| Params (M) | GFLOPs | P | R | mAP50 | mAP50-95 | Latency (ms) | FPS | P95 (ms) | P99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| {conv['params_m']:.6f} | {conv['gflops']:.6f} | {conv['P']:.6f} | {conv['R']:.6f} | {conv['mAP50']:.6f} | {conv['mAP50_95']:.6f} | {conv['mean_latency_ms']:.6f} | {conv['processing_fps']:.6f} | {conv['P95_ms']:.6f} | {conv['P99_ms']:.6f} |

Follow-up triggered: **{'YES' if trigger else 'NO'}**. Further architecture search: **NO**.

## Interpretation

{interpretation}

All values come from the locked seed-0 accuracy protocol and three independent decoded-frame FP32 latency runs on the RTX 3090. Previous experiments and the manuscript were not modified.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
