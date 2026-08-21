from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import torch


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
    summarize,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decoded-frame/in-memory batch-1 processing latency benchmark.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--max-frames", type=int, default=0, help="0 uses every image.")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--expected-images", type=int, default=EXPECTED_SCB2_VAL_IMAGES)
    return parser.parse_args()


def configure_logging(outdir: Path) -> logging.Logger:
    logger = logging.getLogger("in_memory_latency")
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


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def preload_images(paths: list[Path]) -> list[tuple[str, Any]]:
    decoded: list[tuple[str, Any]] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read/decode image during untimed preload: {path}")
        decoded.append((str(path), image))
    return decoded


def process_decoded_frame(
    predictor: DetectionPredictor,
    filename: str,
    original: Any,
    imgsz: int,
    deadline_ms: float,
    record: bool,
) -> dict[str, Any] | None:
    device = predictor.device
    synchronize(device)
    total_start = time.perf_counter()

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

    predictor.batch = ([filename], [original], [""])
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
    processing_total_ms = elapsed_ms(total_start)
    if not record:
        return None
    return {
        "filename": Path(filename).name,
        "preprocess_ms": preprocess_ms,
        "host_to_device_ms": host_to_device_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "output_to_cpu_ms": output_to_cpu_ms,
        "processing_total_ms": processing_total_ms,
        "deadline_missed": int(processing_total_ms > deadline_ms),
        "num_detections": int(sum(output.shape[0] for output in cpu_outputs)),
    }


def write_frame_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "filename",
        "preprocess_ms",
        "host_to_device_ms",
        "inference_ms",
        "postprocess_ms",
        "output_to_cpu_ms",
        "processing_total_ms",
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
    if args.imgsz <= 0 or args.warmup < 0 or args.target_fps <= 0:
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

    preload_start = time.perf_counter()
    decoded_images = preload_images(images)
    preload_wall_ms = elapsed_ms(preload_start)
    logger.info("Preloaded decoded images: %d", len(decoded_images))
    logger.info("Untimed preload wall-clock: %.3f ms (excluded from all latency statistics)", preload_wall_ms)

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

    shape_probe = cpu_preprocess(predictor, decoded_images[0][1])
    actual_shape = tuple(shape_probe.shape)
    logger.info("Actual model input shape: torch.Size(%s)", list(actual_shape))
    if actual_shape != (1, 3, args.imgsz, args.imgsz):
        raise RuntimeError(f"Actual model input shape {actual_shape} is not [1, 3, {args.imgsz}, {args.imgsz}].")
    logger.info("Precision: %s", "FP16" if predictor.model.fp16 else "FP32")
    logger.info("Device: %s", device)
    logger.info("Model end2end mode: %s", bool(getattr(predictor.model, "end2end", False)))

    deadline_ms = 1000.0 / args.target_fps
    logger.info("Warm-up iterations using decoded RAM frames: %d", args.warmup)
    for index in range(args.warmup):
        filename, image = decoded_images[index % len(decoded_images)]
        process_decoded_frame(predictor, filename, image, args.imgsz, deadline_ms, record=False)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows: list[dict[str, Any]] = []
    logger.info("Starting measured decoded frames: %d", len(decoded_images))
    for index, (filename, image) in enumerate(decoded_images, start=1):
        row = process_decoded_frame(predictor, filename, image, args.imgsz, deadline_ms, record=True)
        assert row is not None
        rows.append({"index": index, **row})
        if index % 100 == 0 or index == len(decoded_images):
            logger.info("Measured %d/%d decoded frames", index, len(decoded_images))

    csv_path = outdir / "latency_frames.csv"
    write_frame_csv(csv_path, rows)
    total_values = [float(row["processing_total_ms"]) for row in rows]
    total_summary = summarize(total_values)
    stage_names = ["preprocess_ms", "host_to_device_ms", "inference_ms", "postprocess_ms", "output_to_cpu_ms"]
    deadline_miss_count = sum(int(row["deadline_missed"]) for row in rows)
    peak_allocated = None
    peak_reserved = None
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024**2)

    summary = {
        "benchmark_name": "decoded-frame processing latency",
        "model": args.model_name or weights.stem,
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
        "nms_policy": "Ultralytics DetectionPredictor non_max_suppression with end2end model flag",
        "legacy_dalc_compatibility": legacy_compatibility,
        "warmup_iterations": args.warmup,
        "num_images_found": len(all_images),
        "num_images": len(decoded_images),
        "preloaded_decoded_images": len(decoded_images),
        "preload_wall_ms_excluded": preload_wall_ms,
        "image_order": "sorted recursive absolute paths, case-insensitive",
        "image_list_sha256": list_hash,
        "dataset_path": str(source),
        "processing_total_ms": total_summary,
        "stage_latency_ms": {stage: summarize([float(row[stage]) for row in rows]) for stage in stage_names},
        "processing_fps": 1000.0 / total_summary["mean"],
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
            "Synchronized wall-clock from an already-decoded OpenCV BGR ndarray in CPU RAM through preprocessing, "
            "host-to-device transfer, FP32 normalization, model inference, end-to-end postprocessing, and final boxes "
            "copied to CPU. File I/O and image decode are excluded."
        ),
        "stage_note": "host_to_device_ms includes device-side dtype conversion and normalization, matching predictor order.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    json_path = outdir / "latency_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Mean processing latency: %.6f ms; processing FPS: %.6f", total_summary["mean"], summary["processing_fps"])
    logger.info("P95: %.6f ms; P99: %.6f ms", total_summary["p95"], total_summary["p99"])
    logger.info("Mean inference latency: %.6f ms", summary["stage_latency_ms"]["inference_ms"]["mean"])
    logger.info("Deadline misses: %d/%d", deadline_miss_count, len(rows))
    logger.info("Saved: %s", csv_path)
    logger.info("Saved: %s", json_path)


if __name__ == "__main__":
    main()
