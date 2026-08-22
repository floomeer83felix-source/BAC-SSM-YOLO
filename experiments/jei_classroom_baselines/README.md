# JEI classroom-specific baseline reimplementations

This directory integrates paper-faithful reimplementations of PLA-YOLO11n
(Sensors, 2025) and WAD-YOLOv8n (Scientific Reports, 2025) for controlled
SCB-Dataset2 comparison.

These are paper-faithful reimplementations based on the published method
descriptions and are not claimed to be the authors' official source code.

The publications do not fully disclose every code-level choice. The explicit
assumptions retained here are:

- PLA: exact LSKA kernel/dilation, exact PConv details, and exact four-scale
  neck code.
- WAD: exact 2DPE-MHA fusion order and some optimizer/code-level details.

The current integration reuses this repository's existing `AIFI` and
`DySample` implementations. No duplicate implementations were added.

## Verification only

Run from the repository root:

```bash
PYTHONPATH=. python experiments/jei_classroom_baselines/scripts/verify_reimplementations.py
```

The verifier instantiates both models on CPU, performs one batch-1 640x640
forward pass on CUDA when available, validates finite outputs and detection
scales, and writes `smoke_test_report.md` and `smoke_test_report.json`.

Formal training is intentionally not executed in this integration stage.
