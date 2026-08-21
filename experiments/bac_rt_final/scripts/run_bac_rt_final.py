#!/usr/bin/env python3
"""Run the single final ConvEarly experiment and its threshold-gated follow-up."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
OUT = ROOT / "experiments/bac_rt_final"
CONFIG = OUT / "config/bac_ssm_yolo_rt_convearly.yaml"
DATA = ROOT / "scb2-wsl.yaml"
SOURCE = Path("/mnt/d/datasets/SCB/scb2/images/val")
NAME = "BAC-SSM-YOLO-RT-ConvEarly"


def log(message: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
    print(line, flush=True)
    with (OUT / "coordinator.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], logfile: Path | None = None) -> None:
    log("RUN " + " ".join(map(str, command)))
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a", encoding="utf-8") as handle:
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
    else:
        subprocess.run(command, cwd=ROOT, check=True)


def completed(run_dir: Path) -> bool:
    if not (run_dir / "weights/best.pt").is_file() or not (run_dir / "results.csv").is_file():
        return False
    with (run_dir / "results.csv").open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle)) >= 100


def valid_checkpoint(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train(seed: int) -> Path:
    run_name = f"{NAME}-seed{seed}"
    run_dir = OUT / "training" / run_name
    if not completed(run_dir):
        last = run_dir / "weights/last.pt"
        best = run_dir / "weights/best.pt"
        resume_checkpoint = last if valid_checkpoint(last) else best if valid_checkpoint(best) else None
        if resume_checkpoint is not None:
            run(
                [str(PYTHON), "scripts/resume_bac_rt_training.py", "--checkpoint", str(resume_checkpoint)],
                OUT / f"logs/train_seed{seed}_resume.log",
            )
        else:
            run(
                [str(PYTHON), "scripts/train_bac_rt_candidate.py", "--config", str(CONFIG), "--name", run_name,
                 "--seed", str(seed), "--project", str(OUT / "training")],
                OUT / f"logs/train_seed{seed}.log",
            )
    if not completed(run_dir):
        raise RuntimeError(f"Training incomplete: {run_dir}")
    return run_dir / "weights/best.pt"


def publish_weight(source: Path) -> Path:
    target = OUT / "weights/bac_ssm_yolo_rt_convearly_seed0_best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != sha256(source):
            raise RuntimeError("Refusing to overwrite a different existing ConvEarly checkpoint")
    else:
        shutil.copy2(source, target)
    return target


def evaluate(weights: Path, seed: int, outdir: Path) -> None:
    if (outdir / "accuracy_summary.json").is_file():
        return
    run(
        [str(PYTHON), "scripts/evaluate_bac_rt_candidate.py", "--weights", str(weights), "--data", str(DATA),
         "--model-name", NAME, "--scan-stages", "P4/C4,P5/C5,P5-context,P4-neck", "--num-dalc-blocks",
         "4", "--seed", str(seed), "--outdir", str(outdir), "--device", "0", "--batch", "16", "--workers", "8"],
        OUT / f"logs/evaluate_seed{seed}.log",
    )


def latency(weights: Path) -> None:
    for index in (1, 2, 3):
        outdir = OUT / f"latency/run{index}"
        if (outdir / "latency_summary.json").is_file():
            continue
        run(
            [str(PYTHON), "scripts/in_memory_latency.py", "--weights", str(weights), "--source", str(SOURCE),
             "--imgsz", "640", "--device", "cuda:0", "--warmup", "50", "--conf", "0.25", "--iou", "0.7",
             "--expected-images", "848", "--model-name", NAME, "--outdir", str(outdir)],
            OUT / f"logs/latency_run{index}.log",
        )


def seed_follow_up(seed0: dict) -> None:
    summaries = [seed0]
    for seed in (1, 2):
        weights = train(seed)
        outdir = OUT / f"accuracy/seed{seed}"
        evaluate(weights, seed, outdir)
        summaries.append(json.loads((outdir / "accuracy_summary.json").read_text(encoding="utf-8")))
    rows = [{"seed": item["seed"], **item["metrics"]} for item in summaries]
    report = {
        "model": NAME,
        "seeds": rows,
        "mAP50_mean": statistics.mean(row["map50"] for row in rows),
        "mAP50_sd": statistics.stdev(row["map50"] for row in rows),
        "mAP50_95_mean": statistics.mean(row["map50_95"] for row in rows),
        "mAP50_95_sd": statistics.stdev(row["map50_95"] for row in rows),
    }
    (OUT / "accuracy/seed_stability.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def profiler(weights: Path) -> None:
    outdir = OUT / "profiler"
    if not (outdir / "profile_summary.json").is_file():
        run(
            [str(PYTHON), "scripts/profile_runtime.py", "--weights", str(weights), "--source", str(SOURCE),
             "--imgsz", "640", "--device", "cuda:0", "--conf", "0.25", "--iou", "0.7", "--warmup", "50",
             "--profile-steps", "50", "--outdir", str(outdir), "--model-name", NAME],
            OUT / "logs/profiler.log",
        )
    full = json.loads((ROOT / "runs/profiler/bac_640/profile_summary.json").read_text(encoding="utf-8"))
    conv = json.loads((outdir / "profile_summary.json").read_text(encoding="utf-8"))
    def extract(item: dict) -> dict:
        return {
            "selective_scan_cuda_ms": item["module_profile"]["dalc_ss2d_cuda_ms"],
            "tensor_rearrangement_ms": item["forward_profile"]["groups"]["Tensor rearrangement"]["cuda_time_ms"],
            "cuda_kernel_launches_per_step": item["forward_profile"]["cuda_kernel_launches_per_step"],
            "peak_memory_allocated_mib": item["pipeline_profile"]["peak_memory_allocated_mib"],
            "peak_memory_reserved_mib": item["pipeline_profile"]["peak_memory_reserved_mib"],
        }
    report = {"full_BAC": extract(full), "ConvEarly": extract(conv)}
    report["reduction_percent"] = {
        key: (1.0 - report["ConvEarly"][key] / report["full_BAC"][key]) * 100.0
        for key in report["full_BAC"] if report["full_BAC"][key]
    }
    (outdir / "profiler_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    run([str(PYTHON), "scripts/validate_bac_rt_final_config.py"], OUT / "logs/config_validation.log")
    seed0_training_weight = train(0)
    weight = publish_weight(seed0_training_weight)
    evaluate(weight, 0, OUT / "accuracy")
    latency(weight)
    run([str(PYTHON), "scripts/build_bac_rt_final_results.py"], OUT / "logs/build_results.log")
    analysis = json.loads((OUT / "final_analysis.json").read_text(encoding="utf-8"))
    trigger = analysis["screening"]["follow_up_trigger"]
    if trigger:
        log("ConvEarly met the follow-up threshold; running only seed1/2 and one profiler validation.")
        seed0 = json.loads((OUT / "accuracy/accuracy_summary.json").read_text(encoding="utf-8"))
        seed_follow_up(seed0)
        profiler(weight)
        run([str(PYTHON), "scripts/build_bac_rt_final_results.py"], OUT / "logs/build_results.log")
        status = "SUCCESS_FOLLOW_UP_COMPLETE"
    else:
        log("ConvEarly failed the locked threshold; stopping final architecture search.")
        status = "NEGATIVE_RESULT_STOPPED"
    (OUT / "FINAL_EXPERIMENT_COMPLETE").write_text(status + "\n", encoding="utf-8")
    log("Final controlled experiment complete. Paper modified: NO.")


if __name__ == "__main__":
    main()
