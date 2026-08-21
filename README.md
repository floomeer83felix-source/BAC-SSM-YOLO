# BAC-SSM-YOLO

Official release package for **BAC-SSM-YOLO**, a lightweight classroom behavior detector with Direction-Adaptive Local-Compensated SS2D, Behavior-Aware Convolution, and coordinate-aware feature recalibration.

This repository is a clean package extracted from the experiment workspace. It keeps only the code, configuration files, utility scripts, and the final trained weight required to reproduce or evaluate the proposed model.

## Contents

```text
BAC-SSM-YOLO/
  configs/
    datasets/
      scb2.yaml
      scb2-wsl.yaml
    models/
      yolo26-da-lc-ss2d-scbconv-ca.yaml
  scripts/
    train.py
    val.py
    predict.py
  tools/
    visualize_dalc_direction_weights.py
  ultralytics/
  weights/
    BAC-SSM-YOLO_scb2_best.pt
```

## Model

The released model configuration is:

```text
configs/models/yolo26-da-lc-ss2d-scbconv-ca.yaml
```

The released checkpoint is:

```text
weights/BAC-SSM-YOLO_scb2_best.pt
```

The checkpoint was trained on SCB-Dataset2 with three classes: `hand-raising`, `reading`, and `writing`.

## Environment

Create a Python environment with PyTorch, then install this package in editable mode:

```bash
pip install -e .
```

The DA-LC-SS2D modules require Mamba/selective-scan dependencies. Install versions compatible with your CUDA and PyTorch environment, for example:

```bash
pip install mamba-ssm causal-conv1d
```

If your platform requires prebuilt wheels, install the matching wheels for your CUDA/PyTorch version before running training or inference.

## Dataset YAML

Before training or validation, edit the dataset paths in:

```text
configs/datasets/scb2.yaml
```

Example:

```yaml
train: D:/datasets/SCB/scb2/images/train
val: D:/datasets/SCB/scb2/images/val

nc: 3
names: ['hand-raising', 'reading', 'writing']
```

## Data Split

The released experiments follow the SCB-Dataset2 YOLO-format split:

| Split | Images | Instances |
|---|---:|---:|
| Train | 3,418 | 14,515 |
| Val | 848 | 3,992 |

Expected directory layout:

```text
scb2/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Class order must remain:

```text
0 hand-raising
1 reading
2 writing
```

## Train

```bash
python scripts/train.py --data configs/datasets/scb2.yaml
```

Common options:

```bash
python scripts/train.py --data configs/datasets/scb2.yaml --epochs 100 --imgsz 640 --batch 16 --workers 8 --device 0
```

The training script uses `AdamW` by default to avoid optimizer-specific issues with selective-scan parameters. You can override it with `--optimizer` and `--lr0` if needed.

## Validate

```bash
python scripts/val.py --data configs/datasets/scb2.yaml --weights weights/BAC-SSM-YOLO_scb2_best.pt
```

## Predict

```bash
python scripts/predict.py --weights weights/BAC-SSM-YOLO_scb2_best.pt --source path/to/images
```

## FPS Test

Forward FPS can be measured with:

```bash
python scripts/fps.py --weights weights/BAC-SSM-YOLO_scb2_best.pt --imgsz 640 --batch 1 --device cuda:0
```

For FP16 testing:

```bash
python scripts/fps.py --weights weights/BAC-SSM-YOLO_scb2_best.pt --imgsz 640 --batch 1 --device cuda:0 --half
```

The script uses warmup iterations and reports both latency and FPS.

## Reported SCB-Dataset2 Result

| Model | Params (M) | GFLOPs | FPS | P (%) | R (%) | mAP@0.5 (%) | mAP@0.5:0.95 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BAC-SSM-YOLO | 5.34 | 15.28 | 57.64 | 64.77 | 65.09 | 68.46 | 48.03 |

All FPS values in the manuscript were measured under a unified Mamba-SSM environment.

## Notes

- The repository intentionally excludes training logs, paper drafts, datasets, cache files, and unrelated experiment variants.
- The local `ultralytics` package contains the custom DA-LC-SS2D, BAC, and coordinate-aware recalibration modules used by the released model.
- If you use the released checkpoint directly, make sure the class order in your dataset YAML matches the order above.

## Supplementary Experiments

- [Profiler-guided real-time optimization, stage 2](experiments/bac_rt_stage2/README.md): controlled NoP2/NoEarly configurations, seed-0 checkpoints, per-frame latency records, and Pareto analysis.
