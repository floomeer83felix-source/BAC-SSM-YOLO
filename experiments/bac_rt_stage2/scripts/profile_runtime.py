from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import math
import platform
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import __version__ as ultralytics_version  # noqa: E402
from ultralytics.models.yolo.detect import DetectionPredictor  # noqa: E402

from end_to_end_latency import (  # noqa: E402
    EXPECTED_SCB2_VAL_IMAGES,
    collect_images,
    cpu_preprocess,
    file_sha256,
    get_driver_version,
    get_git_commit,
    image_list_hash,
    restore_legacy_dalc_flags,
    synchronize,
)


GROUPS = (
    "Selective scan / state-space",
    "Convolution",
    "Tensor rearrangement",
    "Elementwise / gating",
    "Normalization",
    "Memory transfer",
    "Postprocessing",
    "Other / Unclassified",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward and decoded-frame runtime profiling.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--profile-steps", type=int, default=50)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--expected-images", type=int, default=EXPECTED_SCB2_VAL_IMAGES)
    return parser.parse_args()


def configure_logging(outdir: Path) -> logging.Logger:
    logger = logging.getLogger("profile_runtime")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    disk = logging.FileHandler(outdir / "run.log", encoding="utf-8")
    disk.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(disk)
    return logger


def preload_images(paths: list[Path]) -> list[tuple[str, Any]]:
    decoded = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode validation image: {path}")
        decoded.append((str(path), image))
    return decoded


def prepare_input(predictor: DetectionPredictor, image: np.ndarray, imgsz: int) -> torch.Tensor:
    host = cpu_preprocess(predictor, image)
    expected = (1, 3, imgsz, imgsz)
    if tuple(host.shape) != expected:
        raise RuntimeError(f"Actual model input shape {tuple(host.shape)} != {expected}")
    tensor = host.to(predictor.device).float()
    tensor /= 255.0
    return tensor


def run_forward(predictor: DetectionPredictor, tensor: torch.Tensor) -> Any:
    with torch.inference_mode():
        return predictor.inference(tensor)


def run_pipeline(
    predictor: DetectionPredictor, filename: str, original: np.ndarray, imgsz: int
) -> Any:
    tensor = prepare_input(predictor, original, imgsz)
    with torch.inference_mode():
        predictions = predictor.inference(tensor)
    predictor.batch = ([filename], [original], [""])
    with torch.inference_mode():
        results = predictor.postprocess(predictions, tensor, [original])
    return [result.boxes.data.detach().cpu().numpy() for result in results]


def cuda_value(event: Any, name: str) -> float:
    # PyTorch 2.x transitioned from cuda_time_* to device_time_* naming.
    values = []
    for candidate in (name, name.replace("cuda", "device")):
        value = getattr(event, candidate, None)
        if value is not None:
            values.append(float(value))
    return max(values, default=0.0)


def memory_value(event: Any, name: str) -> int:
    values = []
    for candidate in (name, name.replace("cuda", "device")):
        value = getattr(event, candidate, None)
        if value is not None:
            values.append(int(value))
    return max(values, default=0)


def classify_operator(operator: str) -> str:
    name = operator.lower().replace(" ", "")
    if any(token in name for token in ("selective_scan", "selectivescan", "cross_scan", "crossscan", "cross_merge", "crossmerge", "mamba", "ss2d")):
        return "Selective scan / state-space"
    if any(token in name for token in ("conv", "cudnn", "depthwise")):
        return "Convolution"
    if any(token in name for token in ("permute", "transpose", "contiguous", "reshape", "view", "flatten", "cat", "stack", "chunk", "split", "flip", "clone")):
        return "Tensor rearrangement"
    if any(token in name for token in ("batch_norm", "batchnorm", "layer_norm", "layernorm", "group_norm", "groupnorm", "instance_norm", "instancenorm")):
        return "Normalization"
    if any(token in name for token in ("cudamemcpy", "memcpy", "aten::to", "aten::_to_copy", "copy_", "copy")):
        return "Memory transfer"
    if any(token in name for token in ("non_max_suppression", "torchvision::nms", "aten::nms", "nms", "box_iou")):
        return "Postprocessing"
    if any(token in name for token in ("mul", "add", "sub", "div", "sigmoid", "silu", "softmax", "exp", "relu", "gelu", "tanh", "sum", "mean", "where", "clamp")):
        return "Elementwise / gating"
    return "Other / Unclassified"


def export_trace(profiler: Any, destination: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        raw = Path(tmpdir) / "trace.json"
        profiler.export_chrome_trace(str(raw))
        with raw.open("rb") as source, gzip.open(destination, "wb", compresslevel=6) as target:
            shutil.copyfileobj(source, target)


def analyze_profiler(profiler: Any, steps: int, prefix: str, outdir: Path) -> dict[str, Any]:
    records = []
    total_self_cuda_us = 0.0
    total_cuda_calls = 0
    for event in profiler.key_averages(group_by_input_shape=False):
        self_cuda_us = cuda_value(event, "self_cuda_time_total")
        cuda_total_us = cuda_value(event, "cuda_time_total")
        calls = int(getattr(event, "count", 0))
        if self_cuda_us > 0:
            total_self_cuda_us += self_cuda_us
            total_cuda_calls += calls
        records.append(
            {
                "operator": str(event.key),
                "self_cpu_time_ms": float(getattr(event, "self_cpu_time_total", 0.0)) / 1000.0 / steps,
                "cpu_time_total_ms": float(getattr(event, "cpu_time_total", 0.0)) / 1000.0 / steps,
                "self_cuda_time_ms": self_cuda_us / 1000.0 / steps,
                "cuda_time_total_ms": cuda_total_us / 1000.0 / steps,
                "cuda_time_percent": 0.0,
                "calls": calls / steps,
                "cuda_time_per_call_us": self_cuda_us / calls if calls else 0.0,
                "cpu_memory": int(getattr(event, "cpu_memory_usage", 0)) / steps,
                "cuda_memory": memory_value(event, "cuda_memory_usage") / steps,
                "group": classify_operator(str(event.key)),
            }
        )
    denominator = total_self_cuda_us / 1000.0 / steps
    for record in records:
        record["cuda_time_percent"] = record["self_cuda_time_ms"] / denominator * 100.0 if denominator else 0.0
    records.sort(key=lambda item: item["self_cuda_time_ms"], reverse=True)

    operator_path = outdir / f"{prefix}operator_profile.csv"
    with operator_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records[:50])

    grouped: dict[str, dict[str, float]] = {
        name: {"cuda_time_ms": 0.0, "calls": 0.0} for name in GROUPS
    }
    for record in records:
        if record["self_cuda_time_ms"] <= 0:
            continue
        group = grouped[record["group"]]
        group["cuda_time_ms"] += record["self_cuda_time_ms"]
        group["calls"] += record["calls"]
    group_rows = []
    for name in GROUPS:
        item = grouped[name]
        group_rows.append(
            {
                "group": name,
                "cuda_time_ms": item["cuda_time_ms"],
                "cuda_time_percent": item["cuda_time_ms"] / denominator * 100.0 if denominator else 0.0,
                "calls": item["calls"],
            }
        )
    group_path = outdir / f"{prefix}operator_groups.csv"
    with group_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)

    top_path = outdir / f"{prefix}top20_cuda_operators.md"
    lines = [
        "| Rank | Operator | Self CUDA (ms) | Total CUDA (ms) | CUDA (%) | Calls/step | Time/call (us) |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, record in enumerate(records[:20], start=1):
        lines.append(
            f'| {rank} | `{record["operator"]}` | {record["self_cuda_time_ms"]:.6f} | '
            f'{record["cuda_time_total_ms"]:.6f} | {record["cuda_time_percent"]:.3f} | '
            f'{record["calls"]:.2f} | {record["cuda_time_per_call_us"]:.3f} |'
        )
    top_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "self_cuda_operator_time_ms_per_step": denominator,
        "total_cuda_operator_calls_per_step": total_cuda_calls / steps,
        "groups": {row["group"]: row for row in group_rows},
        "top_operator": records[0] if records else None,
    }


def runtime_call_counts(profiler: Any, steps: int) -> dict[str, float]:
    counts = defaultdict(int)
    for event in profiler.key_averages(group_by_input_shape=False):
        counts[str(event.key)] += int(getattr(event, "count", 0))
    return {
        "cuda_kernel_launches_per_step": counts["cudaLaunchKernel"] / steps,
        "cuda_memcpy_async_calls_per_step": counts["cudaMemcpyAsync"] / steps,
        "cuda_memset_async_calls_per_step": counts["cudaMemsetAsync"] / steps,
    }


def profile_forward(
    predictor: DetectionPredictor, tensor: torch.Tensor, steps: int, outdir: Path, logger: logging.Logger
) -> dict[str, Any]:
    synchronize(predictor.device)
    torch.cuda.reset_peak_memory_stats(predictor.device)
    logger.info("Starting forward profiler: %d steps", steps)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as trace_profiler:
        for _ in range(steps):
            run_forward(predictor, tensor)
            trace_profiler.step()
        synchronize(predictor.device)
    export_trace(trace_profiler, outdir / "forward_torch_profiler_trace.json.gz")
    runtime_counts = runtime_call_counts(trace_profiler, steps)

    # torch.profiler in this PyTorch 2.3 build exports CUPTI traces but reports zero
    # CUDA key averages. The built-in autograd CUDA profiler provides the missing
    # operator aggregates without changing the model execution path.
    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as operator_profiler:
        for _ in range(steps):
            run_forward(predictor, tensor)
        synchronize(predictor.device)
    export_trace(operator_profiler, outdir / "forward_trace.json.gz")
    synchronize(predictor.device)
    peak_allocated = torch.cuda.max_memory_allocated(predictor.device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(predictor.device) / 2**20
    result = analyze_profiler(operator_profiler, steps, "", outdir)
    result.update(runtime_counts)
    result["operator_timing_backend"] = "torch.autograd.profiler CUDA aggregates (torch.profiler trace retained)"
    result["peak_memory_allocated_mib"] = peak_allocated
    result["peak_memory_reserved_mib"] = peak_reserved
    return result


def profile_pipeline(
    predictor: DetectionPredictor,
    decoded: list[tuple[str, np.ndarray]],
    imgsz: int,
    steps: int,
    outdir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    synchronize(predictor.device)
    torch.cuda.reset_peak_memory_stats(predictor.device)
    logger.info("Starting processing-pipeline profiler: %d steps", steps)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as trace_profiler:
        for index in range(steps):
            filename, image = decoded[index % len(decoded)]
            run_pipeline(predictor, filename, image, imgsz)
            trace_profiler.step()
        synchronize(predictor.device)
    export_trace(trace_profiler, outdir / "pipeline_torch_profiler_trace.json.gz")
    runtime_counts = runtime_call_counts(trace_profiler, steps)

    with torch.autograd.profiler.profile(use_cuda=True, record_shapes=True, profile_memory=True) as operator_profiler:
        for index in range(steps):
            filename, image = decoded[index % len(decoded)]
            run_pipeline(predictor, filename, image, imgsz)
        synchronize(predictor.device)
    export_trace(operator_profiler, outdir / "pipeline_trace.json.gz")
    synchronize(predictor.device)
    peak_allocated = torch.cuda.max_memory_allocated(predictor.device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(predictor.device) / 2**20
    result = analyze_profiler(operator_profiler, steps, "pipeline_", outdir)
    result.update(runtime_counts)
    result["operator_timing_backend"] = "torch.autograd.profiler CUDA aggregates (torch.profiler trace retained)"
    result["peak_memory_allocated_mib"] = peak_allocated
    result["peak_memory_reserved_mib"] = peak_reserved
    return result


def native_model(predictor: DetectionPredictor) -> torch.nn.Module:
    model = predictor.model
    while hasattr(model, "model") and not hasattr(model, "yaml"):
        model = model.model
    return model


def select_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module, str]]:
    selected: list[tuple[str, torch.nn.Module, str]] = []
    for name, module in model.named_modules():
        module_type = type(module).__name__
        is_direct_stage = name.startswith("model.") and name.count(".") == 1
        if is_direct_stage:
            category = "Detection head" if module_type.endswith("Detect") or module_type == "Detect" else "Backbone/neck stage"
            selected.append((name, module, category))
        elif module_type == "SS2DDALC":
            selected.append((name, module, "DA-LC-SS2D"))
        elif module_type in {"CoordAttn"}:
            selected.append((name, module, "CA"))
    return selected


def module_profile(
    predictor: DetectionPredictor,
    tensor: torch.Tensor,
    steps: int,
    outdir: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    model = native_model(predictor)
    selected = select_modules(model)
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
    starts: dict[str, list[torch.cuda.Event]] = defaultdict(list)
    handles = []

    for name, module, _ in selected:
        def pre_hook(_module: torch.nn.Module, _inputs: Any, module_name: str = name) -> None:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts[module_name].append(event)

        def post_hook(_module: torch.nn.Module, _inputs: Any, _output: Any, module_name: str = name) -> None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = starts[module_name].pop()
            events[module_name].append((start, end))

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))

    forward_events = []
    logger.info("Starting module-level CUDA Event profile: %d steps, %d modules", steps, len(selected))
    try:
        for _ in range(steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_forward(predictor, tensor)
            end.record()
            forward_events.append((start, end))
        synchronize(predictor.device)
    finally:
        for handle in handles:
            handle.remove()

    forward_times = [start.elapsed_time(end) for start, end in forward_events]
    forward_mean = float(np.mean(forward_times))
    metadata = {name: (type(module).__name__, category) for name, module, category in selected}
    rows = []
    for name, pairs in events.items():
        times = [start.elapsed_time(end) for start, end in pairs]
        module_type, category = metadata[name]
        mean_ms = float(np.mean(times))
        rows.append(
            {
                "module_name": name,
                "module_type": module_type,
                "category": category,
                "mean_cuda_ms": mean_ms,
                "sd_cuda_ms": float(np.std(times)),
                "percent_of_forward": mean_ms / forward_mean * 100.0 if forward_mean else 0.0,
                "calls_per_forward": len(times) / steps,
            }
        )
    rows.sort(key=lambda item: item["mean_cuda_ms"], reverse=True)
    with (outdir / "module_profile.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dalc_rows = [row for row in rows if row["category"] == "DA-LC-SS2D"]
    return {
        "forward_cuda_ms": forward_mean,
        "forward_cuda_sd_ms": float(np.std(forward_times)),
        "dalc_ss2d_cuda_ms": sum(row["mean_cuda_ms"] for row in dalc_rows),
        "dalc_ss2d_percent": sum(row["mean_cuda_ms"] for row in dalc_rows) / forward_mean * 100.0 if forward_mean else 0.0,
        "profiled_module_count": len(rows),
    }


def main() -> None:
    args = parse_args()
    if args.imgsz <= 0 or args.warmup < 0 or args.profile_steps <= 0:
        raise ValueError("imgsz/profile-steps must be positive and warmup non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler")

    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(outdir)
    images = collect_images(source, 0)
    if len(images) != args.expected_images:
        raise RuntimeError(f"Expected {args.expected_images} validation images, found {len(images)}")
    list_hash = image_list_hash(images, source)
    logger.info("Images found: %d", len(images))
    logger.info("Image list SHA256: %s", list_hash)
    decoded = preload_images(images)
    logger.info("Preloaded decoded images: %d", len(decoded))

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictor = DetectionPredictor(
        overrides={
            "model": str(weights),
            "device": args.device,
            "imgsz": [args.imgsz, args.imgsz],
            "batch": 1,
            "rect": False,
            "conf": args.conf,
            "iou": args.iou,
            "half": False,
            "max_det": 300,
            "agnostic_nms": False,
            "augment": False,
            "compile": False,
            "verbose": False,
            "save": False,
            "show": False,
        }
    )
    predictor.setup_model(str(weights), verbose=False)
    for item in restore_legacy_dalc_flags(predictor.model):
        logger.info("Legacy DA-LC compatibility restored for %s: %s", item["module"], ", ".join(item["restored"]))
    predictor.imgsz = [args.imgsz, args.imgsz]
    if predictor.model.fp16:
        raise RuntimeError("FP32 requested but model backend enabled FP16")
    tensor = prepare_input(predictor, decoded[0][1], args.imgsz)
    logger.info("Actual model input shape: %s", list(tensor.shape))
    logger.info("Precision confirmed: FP32")
    logger.info("Warm-up iterations without profiler: %d", args.warmup)
    for index in range(args.warmup):
        filename, image = decoded[index % len(decoded)]
        run_pipeline(predictor, filename, image, args.imgsz)
    synchronize(predictor.device)
    torch.cuda.reset_peak_memory_stats(predictor.device)

    forward = profile_forward(predictor, tensor, args.profile_steps, outdir, logger)
    pipeline = profile_pipeline(predictor, decoded, args.imgsz, args.profile_steps, outdir, logger)
    modules = module_profile(predictor, tensor, args.profile_steps, outdir, logger)
    environment = {
        "gpu": torch.cuda.get_device_name(predictor.device),
        "driver": get_driver_version(),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "ultralytics": ultralytics_version,
        "python": platform.python_version(),
        "os": platform.platform(),
    }
    summary = {
        "model": args.model_name or weights.stem,
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "resolution": args.imgsz,
        "precision": "FP32",
        "batch_size": 1,
        "rect": False,
        "conf": args.conf,
        "iou": args.iou,
        "warmup_iterations": args.warmup,
        "profile_steps": args.profile_steps,
        "actual_input_shape": list(tensor.shape),
        "num_validation_images": len(decoded),
        "image_list_sha256": list_hash,
        "forward_profile": forward,
        "pipeline_profile": pipeline,
        "module_profile": modules,
        "environment": environment,
        "git_commit": get_git_commit(ROOT),
        "profiler_notice": "Profiler timings are diagnostic and do not replace the validated decoded-frame latency benchmark.",
    }
    (outdir / "profile_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Forward CUDA Event mean: %.6f ms", modules["forward_cuda_ms"])
    logger.info("DA-LC-SS2D module share: %.3f%%", modules["dalc_ss2d_percent"])
    logger.info("Saved profiler outputs under %s", outdir)


if __name__ == "__main__":
    main()
