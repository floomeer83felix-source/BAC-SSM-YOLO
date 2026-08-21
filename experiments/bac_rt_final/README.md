# BAC-SSM-YOLO Final Controlled Real-Time Experiment

ConvEarly restores the exact native YOLO26 convolutional blocks at P2 and P3 and retains DA-LC-SS2D at P4, P5, P5 context, and P4 neck. No other full-model layer changes.

## Result

| Params (M) | GFLOPs | P | R | mAP50 | mAP50-95 | Latency (ms) | FPS | P95 (ms) | P99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4.905873 | 8.672153 | 0.603595 | 0.610764 | 0.632286 | 0.423665 | 14.323517 | 69.815256 | 24.316567 | 29.924654 |

Follow-up triggered: **NO**. Further architecture search: **NO**.

## Interpretation

ConvEarly is measurably faster than NoEarly under the locked protocol, supporting the conclusion that the early SS2D core, rather than only DA/LC, contributes materially to runtime overhead.

All values come from the locked seed-0 accuracy protocol and three independent decoded-frame FP32 latency runs on the RTX 3090. Previous experiments and the manuscript were not modified.
