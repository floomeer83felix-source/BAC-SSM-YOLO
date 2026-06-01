from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BAC-SSM-YOLO forward FPS.")
    parser.add_argument("--weights", default=str(ROOT / "weights/BAC-SSM-YOLO_scb2_best.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action="store_true", help="Use FP16 on CUDA devices.")
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or "cuda" not in args.device else "cpu")
    model = YOLO(args.weights).model.to(device).eval()

    dtype = torch.float16 if args.half and device.type == "cuda" else torch.float32
    if dtype == torch.float16:
        model.half()

    dummy = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device, dtype=dtype)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(dummy)
        sync(device)

        start = time.perf_counter()
        for _ in range(args.iters):
            _ = model(dummy)
        sync(device)
        elapsed = time.perf_counter() - start

    fps = args.batch * args.iters / elapsed
    latency_ms = elapsed * 1000.0 / args.iters
    print(f"weights: {args.weights}")
    print(f"device: {device}")
    print(f"imgsz: {args.imgsz}")
    print(f"batch: {args.batch}")
    print(f"precision: {'FP16' if dtype == torch.float16 else 'FP32'}")
    print(f"latency: {latency_ms:.3f} ms/batch")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()

