# Matched SCB-Dataset2 Seed-0 Training Report

## Completion and protocol

PLA-YOLO11n-reimpl and WAD-YOLOv8n-reimpl completed 100 epochs sequentially. Both `best.pt` checkpoints were evaluated with the locked protocol: 640-pixel FP32 input, `rect=True`, `conf=0.001`, `IoU=0.7`, and `augment=False`. The split contains 3,418 training images and 848 validation images in the class order hand-raising, reading, and writing.

The BAC gate reproduced 48.0520% mAP@0.5:0.95, only 0.0020 percentage point from the 48.05% target; therefore, the 0.10-point gate passed.

## Ranking

| Rank | Model | Params (M) | GFLOPs | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | BAC-SSM-YOLO | 5.343 | 15.314 | 64.7440 | 65.0580 | 68.4768 | 48.0520 |
| 2 | WAD-YOLOv8n-reimpl | 3.910 | 9.624 | 66.0123 | 64.6041 | 68.5718 | 47.7210 |
| 3 | YOLO26s | 9.950 | 22.507 | 63.2895 | 64.3071 | 67.2087 | 46.7067 |
| 4 | PLA-YOLO11n-reimpl | 3.474 | 10.299 | 61.8680 | 63.4080 | 64.7655 | 43.7396 |


## Matched-baseline comparison

- PLA-YOLO11n-reimpl versus BAC: -3.7113 mAP@0.5 points, -4.3124 mAP@0.5:0.95 points, -1.868491 M parameters, and -5.0146168 GFLOPs.
- WAD-YOLOv8n-reimpl versus BAC: +0.0950 mAP@0.5 points, -0.3310 mAP@0.5:0.95 points, -1.433134 M parameters, and -5.6900344 GFLOPs.

WAD-YOLOv8n-reimpl is the strongest classroom reimplementation and is 0.3310 percentage point from BAC on mAP@0.5:0.95. Because this absolute gap is at most 0.50 point, a three-seed follow-up is recommended. It was not started in this run, as required.

## Reproducibility note

The CSV contains all overall and per-class locked-evaluation metrics. Checkpoint SHA256 values, the image-list hashes, exact protocols, deltas, and the deterministic-training caveat are recorded in `matched_analysis.json`. No latency, FP16, profiling, paper, or architecture changes were performed.
