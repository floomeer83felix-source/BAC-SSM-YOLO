#!/usr/bin/env python3
"""Run the locked BAC-RT stage-2 queue with conditional seed/profiler expansion."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
RUN_ROOT = ROOT / "runs/bac_rt_stage2"
DATA = ROOT / "scb2-wsl.yaml"
SOURCE = Path("/mnt/d/datasets/SCB/scb2/images/val")
CANDIDATES = {
    "BAC-SSM-YOLO-RT-NoP2": {
        "slug": "nop2",
        "config": ROOT / "ultralytics/cfg/models/26/bac_rt_stage2/bac_ssm_yolo_rt_nop2.yaml",
        "scan_stages": "P3/C3,P4/C4,P5/C5,P5-context,P4-neck",
        "num_dalc": 5,
    },
    "BAC-SSM-YOLO-RT-NoEarly": {
        "slug": "noearly",
        "config": ROOT / "ultralytics/cfg/models/26/bac_rt_stage2/bac_ssm_yolo_rt_noearly.yaml",
        "scan_stages": "P4/C4,P5/C5,P5-context,P4-neck",
        "num_dalc": 4,
    },
}


def log(message: str) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    with (RUN_ROOT / "coordinator.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], logfile: Path | None = None) -> None:
    log("RUN " + " ".join(map(str, command)))
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as handle:
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, cwd=ROOT, check=True)


def training_complete(run_dir: Path) -> bool:
    results = run_dir / "results.csv"
    best = run_dir / "weights/best.pt"
    if not results.is_file() or not best.is_file():
        return False
    with results.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle)) >= 100


def train(name: str, info: dict, seed: int, seed_root: Path) -> Path:
    run_name = f"{name}-seed{seed}"
    run_dir = seed_root / run_name
    if not training_complete(run_dir):
        run(
            [
                str(PYTHON), "scripts/train_bac_rt_candidate.py", "--config", str(info["config"]),
                "--name", run_name, "--seed", str(seed), "--project", str(seed_root),
            ],
            RUN_ROOT / "logs" / f"train_{run_name}.log",
        )
    if not training_complete(run_dir):
        raise RuntimeError(f"Training did not complete: {run_dir}")
    return run_dir / "weights/best.pt"


def evaluate(name: str, info: dict, weights: Path, seed: int, outdir: Path) -> None:
    summary = outdir / "accuracy_summary.json"
    if summary.is_file():
        return
    run(
        [
            str(PYTHON), "scripts/evaluate_bac_rt_candidate.py", "--weights", str(weights), "--data", str(DATA),
            "--model-name", name, "--scan-stages", info["scan_stages"], "--num-dalc-blocks",
            str(info["num_dalc"]), "--seed", str(seed), "--outdir", str(outdir), "--device", "0",
            "--batch", "16", "--workers", "8",
        ],
        RUN_ROOT / "logs" / f"evaluate_{name}_seed{seed}.log",
    )


def latency(name: str, weights: Path, base: Path) -> None:
    for index in (1, 2, 3):
        outdir = base / f"run{index}"
        if (outdir / "latency_summary.json").is_file():
            continue
        run(
            [
                str(PYTHON), "scripts/in_memory_latency.py", "--weights", str(weights), "--source", str(SOURCE),
                "--imgsz", "640", "--device", "cuda:0", "--warmup", "50", "--conf", "0.25", "--iou",
                "0.7", "--expected-images", "848", "--model-name", name, "--outdir", str(outdir),
            ],
            RUN_ROOT / "logs" / f"latency_{name}_run{index}.log",
        )


def seed_stability(name: str, info: dict, seed0_accuracy: Path) -> None:
    summaries = [json.loads(seed0_accuracy.read_text(encoding="utf-8"))]
    seed_root = RUN_ROOT / "seed_stability/training"
    for seed in (1, 2):
        weights = train(name, info, seed, seed_root)
        outdir = RUN_ROOT / f"seed_stability/{info['slug']}/seed{seed}/accuracy"
        evaluate(name, info, weights, seed, outdir)
        summaries.append(json.loads((outdir / "accuracy_summary.json").read_text(encoding="utf-8")))
    rows = [
        {"seed": item["seed"], "mAP50": item["metrics"]["map50"], "mAP50_95": item["metrics"]["map50_95"]}
        for item in summaries
    ]
    report = {
        "model": name,
        "seeds": rows,
        "mAP50_mean": statistics.mean(row["mAP50"] for row in rows),
        "mAP50_sd": statistics.stdev(row["mAP50"] for row in rows),
        "mAP50_95_mean": statistics.mean(row["mAP50_95"] for row in rows),
        "mAP50_95_sd": statistics.stdev(row["mAP50_95"] for row in rows),
    }
    (RUN_ROOT / "seed_stability.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def profiler(name: str, weights: Path) -> None:
    outdir = RUN_ROOT / "best_profiler"
    if not (outdir / "profile_summary.json").is_file():
        run(
            [
                str(PYTHON), "scripts/profile_runtime.py", "--weights", str(weights), "--source", str(SOURCE),
                "--imgsz", "640", "--device", "cuda:0", "--conf", "0.25", "--iou", "0.7", "--warmup",
                "50", "--profile-steps", "50", "--outdir", str(outdir), "--model-name", name,
            ],
            RUN_ROOT / "logs" / "best_profiler.log",
        )
    full_path = ROOT / "runs/profiler/bac_640/profile_summary.json"
    full = json.loads(full_path.read_text(encoding="utf-8"))
    rt = json.loads((outdir / "profile_summary.json").read_text(encoding="utf-8"))
    def extract(item: dict) -> dict:
        return {
            "dalc_cuda_ms": item["module_profile"]["dalc_ss2d_cuda_ms"],
            "tensor_rearrangement_ms": item["forward_profile"]["groups"]["Tensor rearrangement"]["cuda_time_ms"],
            "cuda_kernel_launches_per_step": item["forward_profile"]["cuda_kernel_launches_per_step"],
            "peak_memory_allocated_mib": item["pipeline_profile"].get("peak_memory_allocated_mib"),
            "peak_memory_reserved_mib": item["pipeline_profile"].get("peak_memory_reserved_mib"),
        }
    report = {"full": extract(full), "best_candidate": extract(rt)}
    report["reduction_percent"] = {
        key: (1.0 - report["best_candidate"][key] / report["full"][key]) * 100.0
        for key in report["full"]
        if report["full"][key] not in {None, 0} and report["best_candidate"][key] is not None
    }
    (RUN_ROOT / "profiler_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    run([str(PYTHON), "scripts/validate_bac_rt_stage2_configs.py"], RUN_ROOT / "logs/config_validation.log")
    weights_by_name = {}
    for name, info in CANDIDATES.items():
        log(f"Starting locked seed-0 workflow for {name}")
        weights = train(name, info, 0, RUN_ROOT / "training")
        weights_by_name[name] = weights
        base = RUN_ROOT / info["slug"]
        evaluate(name, info, weights, 0, base / "accuracy")
        latency(name, weights, base / "latency")
    run([str(PYTHON), "scripts/build_bac_rt_stage2_results.py"], RUN_ROOT / "logs/build_results.log")

    analysis = json.loads((RUN_ROOT / "stage2_analysis.json").read_text(encoding="utf-8"))
    best = analysis["best_qualifying_candidate"]
    if best is None:
        log("Neither candidate reached Tier B; stopping architecture search as required.")
        (RUN_ROOT / "STAGE2_COMPLETE").write_text("NO_TIER_A_OR_B\n", encoding="utf-8")
        return
    info = CANDIDATES[best]
    log(f"Selected {best}; running only its seed1/2 stability and profiler validation.")
    seed_stability(best, info, RUN_ROOT / info["slug"] / "accuracy/accuracy_summary.json")
    profiler(best, weights_by_name[best])
    (RUN_ROOT / "STAGE2_COMPLETE").write_text(f"BEST={best}\n", encoding="utf-8")
    log("All required conditional stage-2 work completed.")


if __name__ == "__main__":
    main()
