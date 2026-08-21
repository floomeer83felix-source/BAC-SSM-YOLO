from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import __version__ as ultralytics_version  # noqa: E402
from ultralytics.models.yolo.detect import DetectionPredictor  # noqa: E402


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_IMGSZ = 640
DEFAULT_WARMUP = 50
DEFAULT_TARGET_FPS = 30.0
EXPECTED_SCB2_VAL_IMAGES = 848


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="True batch-1 end-to-end detection latency benchmark.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--max-frames", type=int, default=0, help="0 uses every image.")
    parser.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--expected-images", type=int, default=EXPECTED_SCB2_VAL_IMAGES)
    return parser.parse_args()


def configure_logging(outdir: Path) -> logging.Logger:
    logger = logging.getLogger("end_to_end_latency")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(outdir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_list_hash(paths: list[Path], source: Path) -> str:
    payload = "\n".join(path.relative_to(source).as_posix() for path in paths).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def get_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()[0].strip()
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty latency list.")
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": float(max(values)),
    }


def collect_images(source: Path, max_frames: int) -> list[Path]:
    images = sorted(
        (path.resolve() for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.as_posix().lower(),
    )
    return images[:max_frames] if max_frames > 0 else images


def cpu_preprocess(predictor: DetectionPredictor, image: np.ndarray) -> torch.Tensor:
    transformed = predictor.pre_transform([image])[0]
    if transformed.shape[-1] == 3:
        transformed = transformed[..., ::-1]
    transformed = transformed.transpose((2, 0, 1))
    transformed = np.ascontiguousarray(transformed)
    return torch.from_numpy(transformed).unsqueeze(0)


def restore_legacy_dalc_flags(model: torch.nn.Module) -> list[dict[str, Any]]:
    """Restore flags added after legacy DA-LC checkpoints were serialized."""
    restored: list[dict[str, Any]] = []
    backend = getattr(model, "backend", None)
    native_model = getattr(backend, "model", model)
    for name, module in native_model.named_modules():
        if module.__class__.__name__ != "SS2DDALC":
            continue
        missing: list[str] = []
        if not hasattr(module, "use_direction_adaptive"):
            module.use_direction_adaptive = True
            missing.append("use_direction_adaptive=True")
        if not hasattr(module, "use_local_compensation"):
            module.use_local_compensation = True
            missing.append("use_local_compensation=True")
        if missing:
            restored.append({"module": name, "restored": missing})
    return restored


def complete_pipeline(
    predictor: DetectionPredictor,
    image_path: Path,
    imgsz: int,
    deadline_ms: float,
    record: bool,
) -> dict[str, Any] | None:
    device = predictor.device
    synchronize(device)
    total_start = time.perf_counter()

    stage_start = time.perf_counter()
    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    read_decode_ms = elapsed_ms(stage_start)
    if original is None:
        raise RuntimeError(f"Failed to read/decode image: {image_path}")

    stage_start = time.perf_counter()
    host_tensor = cpu_preprocess(predictor, original)
    preprocess_ms = elapsed_ms(stage_start)
    expected_shape = (1, 3, imgsz, imgsz)
    if tuple(host_tensor.shape) != expected_shape:
        raise RuntimeError(f"Actual model input shape {tuple(host_tensor.shape)} != {expected_shape}; aborting.")

    synchronize(device)
    stage_start = time.perf_counter()
    model_input = host_tensor.to(device)
    model_input = model_input.half() if predictor.model.fp16 else model_input.float()
    model_input /= 255.0
    synchronize(device)
    host_to_device_ms = elapsed_ms(stage_start)

    synchronize(device)
    stage_start = time.perf_counter()
    with torch.inference_mode():
        predictions = predictor.inference(model_input)
    synchronize(device)
    inference_ms = elapsed_ms(stage_start)

    predictor.batch = ([str(image_path)], [original], [""])
    synchronize(device)
    stage_start = time.perf_counter()
    with torch.inference_mode():
        results = predictor.postprocess(predictions, model_input, [original])
    synchronize(device)
    postprocess_ms = elapsed_ms(stage_start)

    synchronize(device)
    stage_start = time.perf_counter()
    cpu_outputs = [result.boxes.data.detach().cpu().numpy() for result in results]
    synchronize(device)
    output_to_cpu_ms = elapsed_ms(stage_start)

    synchronize(device)
    total_end_to_end_ms = elapsed_ms(total_start)
    if not record:
        return None
    return {
        "filename": image_path.name,
        "read_decode_ms": read_decode_ms,
        "preprocess_ms": preprocess_ms,
        "host_to_device_ms": host_to_device_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "output_to_cpu_ms": output_to_cpu_ms,
        "total_end_to_end_ms": total_end_to_end_ms,
        "deadline_missed": int(total_end_to_end_ms > deadline_ms),
        "num_detections": int(sum(output.shape[0] for output in cpu_outputs)),
    }


def write_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "filename",
        "read_decode_ms",
        "preprocess_ms",
        "host_to_device_ms",
        "inference_ms",
        "postprocess_ms",
        "output_to_cpu_ms",
        "total_end_to_end_ms",
        "deadline_missed",
        "num_detections",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).expanduser().resolve()
    source = Path(args.source).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(outdir)

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.is_dir():
        raise FileNotFoundError(f"Validation directory not found: {source}")
    if args.warmup < 0 or args.target_fps <= 0 or args.imgsz <= 0:
        raise ValueError("imgsz and target-fps must be positive; warmup must be non-negative.")
    if not torch.cuda.is_available() and str(args.device).lower() != "cpu":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    all_images = collect_images(source, 0)
    images = all_images[: args.max_frames] if args.max_frames > 0 else all_images
    if not images:
        raise RuntimeError(f"No supported images found under {source}")
    if args.max_frames <= 0 and args.expected_images > 0 and len(all_images) != args.expected_images:
        raise RuntimeError(f"Expected {args.expected_images} validation images, found {len(all_images)}; aborting.")

    list_hash = image_list_hash(images, source)
    (outdir / "image_files.txt").write_text(
        "\n".join(path.relative_to(source).as_posix() for path in images) + "\n", encoding="utf-8"
    )
    logger.info("Images found: %d; images selected: %d", len(all_images), len(images))
    logger.info("Image list SHA256: %s", list_hash)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    overrides = {
        "model": str(weights),
        "device": args.device,
        "imgsz": [args.imgsz, args.imgsz],
        "batch": 1,
        "rect": False,
        "conf": args.conf,
        "iou": args.iou,
        "half": args.half,
        "max_det": 300,
        "agnostic_nms": False,
        "augment": False,
        "compile": False,
        "verbose": False,
        "save": False,
        "show": False,
    }
    predictor = DetectionPredictor(overrides=overrides)
    predictor.setup_model(str(weights), verbose=False)
    legacy_compatibility = restore_legacy_dalc_flags(predictor.model)
    for item in legacy_compatibility:
        logger.info("Legacy DA-LC compatibility restored for %s: %s", item["module"], ", ".join(item["restored"]))
    predictor.imgsz = [args.imgsz, args.imgsz]
    device = predictor.device
    if args.half:
        if not predictor.model.fp16:
            raise RuntimeError("--half was requested, but the loaded backend did not enable FP16.")
    elif predictor.model.fp16:
        raise RuntimeError("FP32 requested, but the loaded backend is using FP16.")

    shape_probe = cpu_preprocess(predictor, cv2.imread(str(images[0]), cv2.IMREAD_COLOR))
    actual_shape = tuple(shape_probe.shape)
    logger.info("Actual model input shape: torch.Size(%s)", list(actual_shape))
    if actual_shape != (1, 3, args.imgsz, args.imgsz):
        raise RuntimeError(f"Actual model input shape {actual_shape} is not [1, 3, {args.imgsz}, {args.imgsz}].")
    logger.info("Precision: %s", "FP16" if predictor.model.fp16 else "FP32")
    logger.info("Device: %s", device)
    logger.info("Model end2end mode: %s", bool(getattr(predictor.model, "end2end", False)))

    deadline_ms = 1000.0 / args.target_fps
    logger.info("Warm-up iterations: %d", args.warmup)
    for index in range(args.warmup):
        complete_pipeline(predictor, images[index % len(images)], args.imgsz, deadline_ms, record=False)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, Any]] = []
    logger.info("Starting measured frames: %d", len(images))
    for index, image_path in enumerate(images, start=1):
        row = complete_pipeline(predictor, image_path, args.imgsz, deadline_ms, record=True)
        assert row is not None
        row = {"index": index, **row}
        rows.append(row)
        if index % 100 == 0 or index == len(images):
            logger.info("Measured %d/%d frames", index, len(images))

    csv_path = outdir / "latency_frames.csv"
    write_frame_csv(csv_path, rows)
    total_values = [float(row["total_end_to_end_ms"]) for row in rows]
    stage_names = [
        "read_decode_ms",
        "preprocess_ms",
        "host_to_device_ms",
        "inference_ms",
        "postprocess_ms",
        "output_to_cpu_ms",
    ]
    total_summary = summarize(total_values)
    deadline_miss_count = sum(int(row["deadline_missed"]) for row in rows)
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

    model_name = args.model_name or weights.stem
    summary = {
        "model": model_name,
        "weights": str(weights),
        "weights_sha256": file_sha256(weights),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "driver": get_driver_version(),
        "precision": "FP16" if predictor.model.fp16 else "FP32",
        "batch_size": 1,
        "imgsz": args.imgsz,
        "actual_input_shape": list(actual_shape),
        "rect": False,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": 300,
        "agnostic_nms": False,
        "model_end2end": bool(getattr(predictor.model, "end2end", False)),
        "legacy_dalc_compatibility": legacy_compatibility,
        "nms_policy": "Ultralytics DetectionPredictor non_max_suppression with end2end model flag",
        "warmup_iterations": args.warmup,
        "num_images_found": len(all_images),
        "num_images": len(images),
        "image_order": "sorted recursive absolute paths, case-insensitive",
        "image_list_sha256": list_hash,
        "dataset_path": str(source),
        "total_end_to_end_ms": total_summary,
        "stage_latency_ms": {
            stage: summarize([float(row[stage]) for row in rows]) for stage in stage_names
        },
        "end_to_end_fps": 1000.0 / total_summary["mean"],
        "soft_deadline": {
            "target_fps": args.target_fps,
            "deadline_ms": deadline_ms,
            "deadline_miss_count": deadline_miss_count,
            "deadline_miss_rate_percent": deadline_miss_count / len(rows) * 100.0,
            "mean_deadline_utilization_percent": total_summary["mean"] / deadline_ms * 100.0,
        },
        "peak_gpu_memory_allocated_mib": peak_allocated,
        "peak_gpu_memory_reserved_mib": peak_reserved,
        "environment": {
            "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "nvidia_driver": get_driver_version(),
            "pytorch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": platform.python_version(),
            "ultralytics_version": ultralytics_version,
            "git_commit": get_git_commit(ROOT),
            "os": platform.platform(),
            "opencv_version": cv2.__version__,
            "tf32_allowed": False,
        },
        "timing_definition": (
            "External synchronized wall-clock from cv2.imread/decode through final detection tensors copied to CPU. "
            "Model initialization, CUDA context initialization, warm-up, logging, and output-file writes are excluded."
        ),
        "stage_note": "host_to_device_ms includes device-side dtype conversion and normalization, matching predictor order.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    json_path = outdir / "latency_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Mean E2E: %.6f ms; FPS: %.6f", total_summary["mean"], summary["end_to_end_fps"])
    logger.info("P95: %.6f ms; P99: %.6f ms", total_summary["p95"], total_summary["p99"])
    logger.info("Deadline misses: %d/%d", deadline_miss_count, len(rows))
    logger.info("Saved: %s", csv_path)
    logger.info("Saved: %s", json_path)


if __name__ == "__main__":
    main()
