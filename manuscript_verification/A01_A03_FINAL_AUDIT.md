# Executive conclusion

**HOLD**

The released checkpoint and complete architecture are reproducible, but the manuscript must not be submitted while its main table, class table, and figures imply one common evaluation despite incompatible values. The ablation section also overstates public reproducibility and treats a composite P3 replacement as a simple additive CA factor. These are documentation/provenance problems, not evidence that new training is necessary.

# A01

The released checkpoint SHA256 is `96b801e4bf0ea7daa4c6e1d69db54eabfb2f12ada90e57c93c411f6697a5fa93`. One locked validation at `imgsz=640`, batch 16, FP32, `rect=True`, `conf=0.001`, IoU 0.7, and no augmentation produced P 64.7440%, R 65.0580%, mAP50 68.4768%, and mAP50:95 48.0520%.

The same call produced AP50/AP50:95 of 78.5142/54.0644 for hand-raising, 62.5855/43.8299 for reading, and 64.3307/46.2619 for writing. Their means reproduce both overall mAP values to floating-point precision. These results are close to 68.46/48.03, but incompatible with the stated Table 13 means near 69.08/48.45 and the approximately 0.690 PR figure. No evidence establishes that all manuscript artifacts came from one run. The locked CSV, curves, and confusion matrices are the authoritative released-checkpoint set.

# A02

Checkpoint train arguments and embedded YAML identify `yolo26-da-lc-ss2d-scbconv-ca.yaml`. The embedded backbone/head exactly match both public YAML copies, and a fresh three-class model has the same 807 state keys and shapes as the checkpoint. Release commit `bcdf87df9dfbe77e28ef310e0c7f04ca6e1cf5d8` contains the weight, YAML, `block.py`, and `mamba_ss2d.py` together.

DA-LC uses aligned four-direction responses, image-wise softmax weights, `4 * weighted_sum`, and `out_norm`. The local depthwise convolution is identity-initialized; `local_scale`, not the convolution, is zero-initialized. P3 `C3k2SCBNeckCAPlus` is a composite multi-branch, factorized-context, coordinate-attention, ECA, spatial-gate, refinement, and residual block. The checkpoint lacks a training commit SHA, so exact training-worktree identity remains unknown despite exact architecture compatibility.

# A03

Only the complete A8 YAML exists in public Git history. Legacy local YAMLs allow secondary inspection and same-revision profiling, but they are unversioned and cannot be represented as public provenance. A0-A5 largely reproduce the paper's rounded complexity; A6 and A7 do not. A8 profiles at 5.342903M parameters and 15.313958 GFLOPs under the current public implementation.

A6 to A8 replaces P3 `C3k2SCBConv` with `C3k2SCBNeckCAPlus`; it is not “A6 + CA.” Current-code profiling confirms a small complexity decrease from that replacement, but not the full historical paper delta. DA-only, LC-only, and the non-default gate-ratio variants are not buildable or traceable from the public release. Detailed row-level evidence is in `A03_ABLATION_MATRIX.csv`.

# Required manuscript changes

## BLOCKING

- Replace Table 13 and Fig. 3-4 values/artifacts with the single locked-evaluation set, or explicitly identify their different checkpoint and protocol. Do not imply one common evaluation without evidence.
- Use the authoritative released-checkpoint metrics and per-class values together; retain sufficient precision so their means remain consistent.
- Rewrite A8 as a P3 composite block replacement relative to A6, not a strict additive “+ CA” row.
- Resolve or remove complexity claims for A6/A7/Table 6/Table 8 that cannot be reproduced from the released code and public artifacts.

## NON-BLOCKING

- Publish exact versioned YAML/code artifacts for any retained legacy ablation and record the profiler revision and command.
- State that only the complete-model architecture is fully covered by current public release provenance.
- Keep the bounded statement that the exact training checkout is unknown because checkpoint Git metadata is null.

## REMOVE/REPHRASE ONLY

- Define CAPlus as the composite P3 refinement rather than equating it with bare coordinate attention.
- Qualify Table 5 as locally reconstructable but only partially publicly traceable.
- Remove or label Table 6 and Table 8 as exploratory if their exact historical implementation cannot be versioned.

# Retraining decision

**NO RETRAINING REQUIRED**

The released checkpoint, locked validation, and matching plots already provide a coherent authoritative result set. The remaining blockers can be resolved through artifact replacement, accurate architecture wording, complexity reprofiling from recovered exact revisions, and publication or removal of unsupported ablation artifacts. Retraining would not repair provenance and is not justified by this audit.

# Do not do

- Do not change class AP values to force their mean to match a preferred manuscript mAP.
- Do not claim that figures and tables from different runs came from one validation.
- Do not fabricate historical YAMLs or commit provenance.
- Do not call a composite block replacement a strict single-factor addition.
- Do not overwrite revalidated outputs merely because different numbers already appear in the manuscript.

# Audit artifacts

- `A01_FINDINGS.md`
- `A02_FINDINGS.md`
- `A03_FINDINGS.md`
- `A03_ABLATION_MATRIX.csv`
- `a03_profile_details.json`
- `a01_a02/checkpoint_metadata.json`
- `a01_a02/metrics_locked_eval.json`
- `a01_a02/overall_metrics.csv`
- `a01_a02/per_class_metrics.csv`
- `a01_a02/locked_eval/` plots and confusion matrices
