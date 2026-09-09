# A01 Findings: checkpoint, tables, and figures

## Locked evidence

- Released checkpoint: `weights/BAC-SSM-YOLO_scb2_best.pt`
- SHA256: `96b801e4bf0ea7daa4c6e1d69db54eabfb2f12ada90e57c93c411f6697a5fa93`
- Checkpoint date: `2026-05-25T01:35:59.325555`
- Ultralytics checkpoint version: `8.4.38`
- Locked protocol: `imgsz=640`, `batch=16`, FP32, `rect=True`, `conf=0.001`, `iou=0.7`, `augment=False`, `plots=True`
- Validation set: 848 images and 3,992 instances; class order is hand-raising, reading, writing.

The WSL runtime used `manuscript_verification/a01_a02/runtime_data.yaml`, which is a path-only translation of `configs/datasets/scb2.yaml`. Its split and class definitions were not changed. The source YAML SHA256 is recorded in `checkpoint_metadata.json`.

## Authoritative locked-evaluation result

| Scope | P (%) | R (%) | AP50 / mAP50 (%) | AP50:95 / mAP50:95 (%) |
|---|---:|---:|---:|---:|
| Overall | 64.7440 | 65.0580 | 68.4768 | 48.0520 |
| hand-raising | - | - | 78.5142 | 54.0644 |
| reading | - | - | 62.5855 | 43.8299 |
| writing | - | - | 64.3307 | 46.2619 |

The arithmetic mean of the three AP50 values is `0.6847680098028484`, exactly equal to overall mAP50. The arithmetic mean of the three AP50:95 values is `0.4805203409540677`; its difference from overall mAP50:95 is `5.551115123125783e-17`, which is floating-point roundoff only. Therefore, the per-class and overall values from this run are internally consistent.

## Comparison with manuscript values

The released checkpoint differs from the Table 3 target (`68.46 / 48.03`) by only `+0.0168 / +0.0220` percentage point. It differs from the stated Table 13 means (`69.08 / 48.45`) by `-0.6032 / -0.3980` point. It is therefore substantially closer to Table 3.

The newly generated PR curve reports the same AP values as this locked validation, not the approximate `0.690` value attributed to the manuscript Fig. 3. The repository does not contain the manuscript source or original evaluation artifacts for Table 13/Fig. 3-4. Consequently, there is no auditable evidence that the current Table 3, Table 13, and manuscript Fig. 3-4 all came from one validation call or one checkpoint.

## Answers to A01

1. The released `best.pt` SHA256 is `96b801e4bf0ea7daa4c6e1d69db54eabfb2f12ada90e57c93c411f6697a5fa93`.
2. It reproduces mAP50 `68.4768%` and mAP50:95 `48.0520%` under the locked protocol. Its P and R are `64.7440%` and `65.0580%`.
3. The same validation gives hand-raising `78.5142 / 54.0644`, reading `62.5855 / 43.8299`, and writing `64.3307 / 46.2619` for AP50/AP50:95, in percent.
4. The class means reproduce the overall mAP values, with no difference beyond floating-point roundoff.
5. The checkpoint is closer to Table 3 (`68.46 / 48.03`) than to the stated Table 13/Fig. 3 values (`69.08 / 48.45` and approximately `0.690`).
6. No repository evidence establishes that Table 3, Table 13, and the manuscript figures belong to the same evaluation. Their stated values are inconsistent with that claim.
7. The authoritative released-checkpoint result is the complete locked result in this report and `metrics_locked_eval.json`: P `64.7440%`, R `65.0580%`, mAP50 `68.4768%`, mAP50:95 `48.0520%`, together with the class AP values above. The curves and confusion matrices in `a01_a02/locked_eval/` are the matching figure set.

## Status

`[A01] PASS` for checkpoint reproducibility and internal metric consistency. Manuscript cross-artifact provenance is not confirmed and requires replacement or rewording using this authoritative output.
