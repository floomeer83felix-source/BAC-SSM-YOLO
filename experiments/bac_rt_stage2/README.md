# BAC-SSM-YOLO Real-Time Optimization: Stage 2

This directory contains the complete measured outputs for the profiler-guided
early-scan placement study. The experiment changes only the placement of
DA-LC-SS2D blocks; BAC, coordinate-aware recalibration, channels, detection
head, loss, optimizer policy, and DA-LC internals remain unchanged.

## Candidates

- **RT-NoP2:** standard SS2D at P2; DA-LC retained at P3, P4, P5, P5 context,
  and P4 neck (5 DA-LC blocks).
- **RT-NoEarly:** standard SS2D at P2 and P3; DA-LC retained at P4, P5, P5
  context, and P4 neck (4 DA-LC blocks).

The YAML validator confirms that these are the only architecture differences
relative to the full BAC-SSM-YOLO configuration.

## Locked protocol

Training uses seed 0, 100 epochs, 640-pixel input, batch 16, 8 workers,
`optimizer=auto`, `lr0=0.01`, and the same SCB-Dataset2 split and augmentation
defaults as the full model. Accuracy evaluation uses FP32, `rect=True`,
confidence 0.001, and IoU 0.7.

Decoded-frame latency uses an RTX 3090, FP32, batch 1, 640 x 640 input,
`rect=False`, confidence 0.25, IoU 0.7, 50 warm-up iterations, and all 848
validation images. Each candidate is measured in three independent runs.
Image decoding and storage I/O are excluded.

## Results

| Model | Params (M) | GFLOPs | mAP50 | mAP50-95 | Latency (ms) | FPS | Tier |
|---|---:|---:|---:|---:|---:|---:|:---:|
| RT-NoP2 | 5.3406 | 15.2541 | 68.2482 | 47.6726 | 17.5553 | 56.9630 | C |
| RT-NoEarly | 5.3338 | 15.2213 | 68.1180 | 47.5362 | 16.5616 | 60.3805 | C |

Tier A required mAP50-95 >= 47.60 and latency <= 14.5 ms. Tier B required
mAP50-95 >= 47.40 and latency <= 15.0 ms. Both candidates exceeded the Tier B
latency ceiling, so the predefined stopping rule was applied. No seed-1/seed-2
expansion and no additional profiler run were performed.

`RT-NoEarly` is Pareto-optimal within the tested comparison set, but it does
not satisfy the real-time screening threshold. The data do not support a claim
that either candidate is faster than YOLO26s.

## Files

- `configs/`: exact candidate model definitions.
- `scripts/`: experiment, validation, evaluation, latency, and aggregation code.
- `training/`: original training arguments and 100-epoch metric histories.
- `weights/`: seed-0 best checkpoints for both candidates.
- `results/bac_rt_stage2_comparison.csv`: paper-independent comparison table.
- `results/stage2_analysis.json`: deltas, tier screening, per-class AP, and Pareto analysis.
- `results/*/latency/run*/latency_frames.csv`: all measured per-frame latency values.
- `results/*/latency/run*/latency_summary.json`: protocol and run-level summaries.

## Reproduction notes

The scripts are preserved exactly as executed in the source workspace. Dataset
paths are intentionally not bundled. Update `scb2-wsl.yaml` and the validation
image source path to match the local SCB-Dataset2 installation before running.
The required software environment is Python 3.10.16, PyTorch 2.3.1+cu118,
CUDA 11.8, and Ultralytics 8.4.38.

Training command used by the coordinator:

```bash
python scripts/train.py \
  --model experiments/bac_rt_stage2/configs/bac_ssm_yolo_rt_nop2.yaml \
  --data configs/datasets/scb2-wsl.yaml \
  --name BAC-SSM-YOLO-RT-NoP2-seed0 \
  --epochs 100 --imgsz 640 --batch 16 --workers 8 \
  --optimizer auto --lr0 0.01 \
  --project runs/bac_rt_stage2/training
```

Replace the config and name with the NoEarly equivalents for the second run.
