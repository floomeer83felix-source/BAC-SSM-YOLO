# Executive conclusion

**CONDITIONAL GO**

The detector itself is reproducible enough for submission: the released checkpoint passes a locked validation, its per-class AP values reproduce the overall mAP exactly, and its embedded architecture matches the released BAC-SSM-YOLO configuration. The remaining work is manuscript normalization and ablation-provenance cleanup, not model retraining.

The previous A03 audit overstated the provenance problem by claiming that only the complete A8 YAML existed in public Git history. Direct inspection of release commit `bcdf87df9dfbe77e28ef310e0c7f04ca6e1cf5d8` shows that Table 4 A0-A8-related YAMLs and the standard-SS2D Table 5 counterpart were publicly versioned at release. Several were subsequently removed by the prune commit `28257ee89ba40e992280398a1f20d491b5075ca7`, which explains their absence from current HEAD. Table 4/5 therefore have substantially stronger provenance than the first audit reported.

Submission should proceed after the blocking manuscript edits below are applied.

# A01 — PASS

The released checkpoint SHA256 is:

`96b801e4bf0ea7daa4c6e1d69db54eabfb2f12ada90e57c93c411f6697a5fa93`

One locked validation at `imgsz=640`, batch 16, FP32, `rect=True`, `conf=0.001`, IoU 0.7, and no augmentation produced:

- P: **64.7440%**
- R: **65.0580%**
- mAP50: **68.4768%**
- mAP50:95: **48.0520%**

The same validation produced:

- hand-raising: **78.5142 / 54.0644** AP50/AP50:95
- reading: **62.5855 / 43.8299**
- writing: **64.3307 / 46.2619**

The class means reproduce the overall mAP values to floating-point precision. These values are very close to the old Table 3 `68.46 / 48.03`, but incompatible with the old Table 13 mean near `69.08 / 48.45` and the old PR figure near `0.690`.

Therefore the locked released-checkpoint output is the authoritative manuscript result set. Table 3, Table 13, PR/F1/P/R plots, and confusion matrices must all use this same set.

# A02 — PASS

Checkpoint train arguments and embedded YAML identify `yolo26-da-lc-ss2d-scbconv-ca.yaml`. The checkpoint's embedded backbone/head match the public configuration, and a fresh three-class model has the same 807 state-dict keys and shapes as the checkpoint with no mismatches.

Release commit `bcdf87df9dfbe77e28ef310e0c7f04ca6e1cf5d8` contains the checkpoint, final YAML, `block.py`, and `mamba_ss2d.py` together. The checkpoint does not store a usable Git training commit SHA, so the exact training checkout remains unknown; this is a bounded provenance limitation rather than an architecture mismatch.

The manuscript method description must reflect the implementation:

- four directional responses are aligned before fusion;
- image-wise softmax direction weights are used;
- fusion is `4 * sum(alpha_d * y_d)` rather than a simple average;
- `out_norm` is applied after the fused scan response;
- the local depthwise 3x3 convolution is identity-initialized;
- `local_scale`, not the local convolution, is initialized to zero;
- P3 `C3k2SCBNeckCAPlus` is a composite multi-branch/context/coordinate/channel/spatial refinement block, not bare Coordinate Attention appended to BAC.

# A03 — PARTIAL, corrected

## Corrected Git provenance

The first A03 report incorrectly stated that only A8 existed in public Git history. Release commit `bcdf87...` publicly contains the relevant Table 4/5 configurations, including:

- `yolo26.yaml`
- `yolo26-scbconv.yaml`
- `yolo26-ca.yaml`
- `yolo26-scbconv-ca.yaml`
- `yolo26-ss2d.yaml`
- `yolo26-da-lc-ss2d.yaml`
- `yolo26-da-lc-ss2d-scbconv.yaml`
- `yolo26-da-lc-ss2d-ca.yaml`
- `yolo26-da-lc-ss2d-scbconv-ca.yaml`
- `yolo26-ss2d-scbconv-ca.yaml`

The later `28257ee...` prune commit removed multiple ablation YAMLs from the working tree. Their current absence does not mean that they were never public. Table 4 and Table 5 should therefore be treated as having **PUBLIC_HISTORICAL** YAML provenance, with A8 additionally **PUBLIC_CURRENT**.

Detailed provenance is recorded in `A03_GIT_HISTORY_INVENTORY.csv` and `A03_ABLATION_MATRIX.csv`.

## Complexity

The release and current audit branch use the same Git blob for `ultralytics/utils/torch_utils.py` (`557cc464864b08b3d4be3ae429e54c3d3e8aa9aa`), so `model_info/get_flops` did not drift between these states. The already executed uniform 640-pixel profiles therefore provide a reproducible complexity baseline once the historical YAML provenance is corrected.

Recommended reproducible values include:

| Row | Params (M) | GFLOPs |
|---|---:|---:|
| A0 | 2.504970 | 5.776077 |
| A1 | 3.139450 | 6.963917 |
| A2 | 2.526209 | 6.046182 |
| A3 | 3.160689 | 7.234022 |
| A4 | 4.785818 | 14.237529 |
| A5 | 4.836800 | 14.373644 |
| A6 | 5.352000 | 15.438706 |
| A7 | 4.858039 | 14.643750 |
| A8 | 5.342903 | 15.313958 |
| Table 5 standard SS2D | 5.291921 | 15.177842 |
| Table 5 DA-LC | 5.342903 | 15.313958 |

The safest final-manuscript policy is to replace retained complexity columns with one consistently rounded set from this uniform profiler rather than preserve incompatible legacy numbers, especially for A6/A7.

## Table 4 interpretation

Table 4 may be retained, but it should be described as a **configuration ablation**, not a fully orthogonal three-factor factorial experiment.

The rows labeled `CA` actually use `C3k2SCBNeckCAPlus` at P3, which is a composite block. A2/A3/A7 should therefore use a name such as **CAPlus composite P3 refinement** rather than bare `CA`.

A5 to A6 is not perfectly single-factor because early DA-LC block arguments also change in addition to BAC/SCBConv placement.

A6 to A8 is not `A6 + CA`. A6 uses P3 `C3k2SCBConv`; A8 replaces that block with `C3k2SCBNeckCAPlus` while retaining the other DA-LC/BAC stages. The decrease in complexity is therefore structurally plausible because this is a replacement, not a simple append.

## Table 5

Table 5 can be retained. The standard-SS2D + BAC + CAPlus YAML and the DA-LC + BAC + CAPlus YAML both exist in public Git history, and the inspected structures form a defensible controlled standard-SS2D versus DA-LC comparison.

The manuscript should distinguish public historical YAML provenance from checkpoint provenance: the complete A8 checkpoint is publicly released, while the standard-SS2D counterpart does not have equivalent current checkpoint-level release provenance.

## Table 6

DA-only and LC-only remain unsupported by the public implementation/history reviewed in this audit. Their local YAMLs reference missing public modules such as `C3k2DASS2D` and `C3k2LCSS2D` and fail to build under the released code.

**Recommendation: remove Table 6 from the main paper unless the original versioned DA-only/LC-only source is recovered.** Keeping only standard SS2D and DA-LC would largely duplicate Table 5.

## Table 8

The public code confirms `0.0625` as the default directional-gate hidden ratio. The non-default `0.03125` and `0.125` variants rely on R1/R3 wrappers that are not traceable in the public implementation/history reviewed.

**Recommendation: remove Table 8 or explicitly downgrade it to an exploratory historical result unless those exact implementations are recovered and versioned.**

# Required manuscript changes

## BLOCKING BEFORE SUBMISSION

1. Replace the BAC row in the main result table with the locked released-checkpoint metrics, rounded consistently: approximately **64.74 / 65.06 / 68.48 / 48.05** for P/R/mAP50/mAP50:95.
2. Replace Table 13 with the same run's per-class values and replace Fig. 3-4 with the matching locked-evaluation figures.
3. Correct the DA-LC method description to match the implementation, including factor-four weighted fusion, `out_norm`, identity local convolution initialization, and zero-initialized `local_scale`.
4. Rename bare `CA` ablation labels to a defined CAPlus/composite P3 refinement and stop describing A8 as `A6 + CA`.
5. Recast Table 4 as configuration ablation rather than strict orthogonal single-factor evidence.
6. Normalize retained Table 4/5 Params and GFLOPs using one reproducible profiling source and rounding rule.
7. Remove Table 6 unless DA-only/LC-only implementation provenance is recovered.
8. Remove or downgrade Table 8 unless the R1/R3 implementation provenance is recovered.

## NON-BLOCKING / DISCLOSURE

- State that the released checkpoint is architecture-compatible with the public final YAML but does not contain a usable training Git SHA.
- State that several ablation YAMLs are public historical artifacts that were subsequently pruned from current HEAD.
- Keep claims bounded to the evaluated validation protocol and datasets; do not imply independent classroom/video generalization that was not tested.

# Retraining decision

**NO RETRAINING REQUIRED**

A01 establishes a coherent released-checkpoint result set; A02 establishes architecture compatibility; corrected A03 establishes public historical provenance for the main Table 4/5 configurations. The remaining deficiencies are manuscript artifact consistency, terminology, complexity normalization, and removal/downgrading of unsupported secondary ablations. None of these is repaired by detector retraining.

# Submission decision

**CONDITIONAL GO**

Once the blocking manuscript edits above are completed and cross-checked, there is no remaining audit finding here that requires new training before submission.

# Do not do

- Do not edit class AP values to force a preferred mAP.
- Do not mix figures and tables from different validation runs while implying one common run.
- Do not fabricate historical YAMLs, module implementations, checkpoints, or Git provenance.
- Do not equate current-HEAD absence with absence from Git history.
- Do not call `C3k2SCBNeckCAPlus` a bare Coordinate Attention insertion.
- Do not call A6-to-A8 a strict additive single-factor step.
- Do not preserve incompatible legacy Params/GFLOPs merely because they already appear in the manuscript.

# Audit artifacts

- `A01_FINDINGS.md`
- `A02_FINDINGS.md`
- `A03_FINDINGS.md`
- `A03_ABLATION_MATRIX.csv`
- `A03_GIT_HISTORY_INVENTORY.csv`
- `A03_HISTORICAL_PROFILE.json`
- `a03_profile_details.json`
- `a01_a02/checkpoint_metadata.json`
- `a01_a02/metrics_locked_eval.json`
- `a01_a02/overall_metrics.csv`
- `a01_a02/per_class_metrics.csv`
- `a01_a02/locked_eval/` plots and confusion matrices
