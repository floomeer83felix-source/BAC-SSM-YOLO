# A03 Findings: corrected Git provenance and complexity audit

## Executive result

`[A03] PARTIAL`, but for a narrower reason than the previous audit stated.

The previous A03 report incorrectly claimed that only the complete A8 YAML existed in public Git history and that the remaining A0-A7/Table 5 configurations were only unversioned local evidence. Direct inspection of release commit `bcdf87df9dfbe77e28ef310e0c7f04ca6e1cf5d8` disproves that claim. The release tree publicly contained the requested ablation YAMLs, including `yolo26-scbconv.yaml`, `yolo26-ca.yaml`, `yolo26-scbconv-ca.yaml`, `yolo26-ss2d.yaml`, `yolo26-da-lc-ss2d.yaml`, `yolo26-da-lc-ss2d-ca.yaml`, `yolo26-da-lc-ss2d-scbconv.yaml`, `yolo26-ss2d-scbconv-ca.yaml`, and the complete `yolo26-da-lc-ss2d-scbconv-ca.yaml`.

The later commit `28257ee89ba40e992280398a1f20d491b5075ca7` (`Prune release package to Ours assets only`) removed multiple ablation configs from the working tree. Removal is explicitly confirmed for at least `yolo26-scbconv.yaml`, `yolo26-da-lc-ss2d-scbconv.yaml`, `yolo26-da-lc-ss2d-ca.yaml`, and `yolo26-ss2d-scbconv-ca.yaml`. The current branch keeps only the complete A8 YAML under `ultralytics/cfg/models/26/`. Therefore, these ablations are **PUBLIC_HISTORICAL**, not `LOCAL_ONLY` or `NOT_TRACEABLE` merely because they are absent from current HEAD.

## Historical YAML inventory relevant to Table 4 and Table 5

| Paper row | Historical YAML at `bcdf87...` | Git blob | Provenance |
|---|---|---|---|
| A0 YOLO26n | `yolo26.yaml` | `1fd5cfc60d78954e1d2751ae466fcde7aeff6066` | PUBLIC_HISTORICAL |
| A1 BAC | `yolo26-scbconv.yaml` | `9414be45a34286edd6523988d5ef1aebedead2d9` | PUBLIC_HISTORICAL |
| A2 CA label | `yolo26-ca.yaml` | `46be0b69cb69851b3f086946ac2be5dd965dc470` | PUBLIC_HISTORICAL |
| A3 BAC + CA label | `yolo26-scbconv-ca.yaml` | `a42bc38ea5f17f035d01fe25a2ca54363ccb8286` | PUBLIC_HISTORICAL |
| A4 standard SS2D | `yolo26-ss2d.yaml` | `a84c0df57e8bec9bd8deaf78d43440ccd3fd7888` | PUBLIC_HISTORICAL |
| A5 DA-LC-SS2D | `yolo26-da-lc-ss2d.yaml` | `d3cfd42ab829983270fca9c0d90b5e57086425cf` | PUBLIC_HISTORICAL |
| A6 DA-LC + BAC | `yolo26-da-lc-ss2d-scbconv.yaml` | `8012be80af4b21758f5fe79f78d236eb989dcd92` | PUBLIC_HISTORICAL |
| A7 DA-LC + CA label | `yolo26-da-lc-ss2d-ca.yaml` | `bd3afc64310efa4cb0c7f89ba51de23e10590acc` | PUBLIC_HISTORICAL |
| A8 complete | `yolo26-da-lc-ss2d-scbconv-ca.yaml` | `ffaede85cc26af2d4c27302cbddfdd50aebff5ea` | PUBLIC_CURRENT + PUBLIC_HISTORICAL |
| Table 5 standard | `yolo26-ss2d-scbconv-ca.yaml` | `4edb2dfbe9af33c3c175d36dbcf232ee7648b3ae` | PUBLIC_HISTORICAL |
| Table 5 DA-LC | complete A8 YAML | `ffaede85cc26af2d4c27302cbddfdd50aebff5ea` | PUBLIC_CURRENT + PUBLIC_HISTORICAL |

This corrects the largest error in the first A03 audit: Table 4 and Table 5 are not based solely on unpublished local YAMLs.

## Profiling-version check

The historical release `bcdf87...` and the current audit branch use the exact same Git blob for `ultralytics/utils/torch_utils.py`:

`557cc464864b08b3d4be3ae429e54c3d3e8aa9aa`

Therefore the repository `model_info/get_flops` implementation did not drift between the release and the current audit branch. A02 also established that the architecture implementation files relevant to the complete model did not change after release. The historical YAML structures inspected above agree with the layer/module structures recorded in the existing A03 profiling matrix. Consequently, the existing uniform 640-pixel profile is a valid reproducible complexity baseline for these historical configurations; the remaining differences from the paper should be treated as manuscript complexity-number mismatches, not as evidence that the YAMLs never existed.

The audit did not fabricate a second set of historical numbers. It retains the already executed uniform profile values and records the stronger public-history provenance established here.

## Uniform reproducible complexity baseline

| Row | Paper Params/GFLOPs | Uniform profile Params/GFLOPs | Assessment |
|---|---:|---:|---|
| A0 YOLO26n | 2.51 / 5.78 | 2.504970 / 5.776077 | reproduces after rounding |
| A1 BAC | 3.14 / 6.96 | 3.139450 / 6.963917 | reproduces after rounding |
| A2 CA label | 2.53 / 6.04 | 2.526209 / 6.046182 | close; CA label is structurally oversimplified |
| A3 BAC + CA label | 3.16 / 7.23 | 3.160689 / 7.234022 | reproduces closely; label is oversimplified |
| A4 standard SS2D | 4.79 / 14.24 | 4.785818 / 14.237529 | reproduces after rounding |
| A5 DA-LC-SS2D | 4.84 / 14.35 | 4.836800 / 14.373644 | small GFLOPs mismatch |
| A6 DA-LC + BAC | 5.42 / 15.76 | 5.352000 / 15.438706 | material mismatch |
| A7 DA-LC + CA label | 4.93 / 14.96 | 4.858039 / 14.643750 | material mismatch |
| A8 complete | 5.34 / 15.28 | 5.342903 / 15.313958 | small GFLOPs mismatch |
| Table 5 standard SS2D + BAC + CAPlus | 5.30 / 15.30 | 5.291921 / 15.177842 | Params close; GFLOPs mismatch |
| Table 5 DA-LC + BAC + CAPlus | 5.34 / 15.28 | 5.342903 / 15.313958 | same complete A8 profile |

For a reproducible final manuscript, the safest policy is to report one uniform profiler source and replace the complexity columns with the uniform-profile values (rounded consistently), rather than preserve incompatible historical complexity numbers whose original profiler execution record is unavailable.

## Table 4 structural interpretation

Table 4 should not be presented as a fully orthogonal three-factor experiment.

A0-A1 is a defensible BAC/SCBConv intervention relative to the base architecture. A0-A4 and A0-A5 are defensible context-family replacements. However, the rows labeled `CA` do not insert bare Coordinate Attention. Historical `yolo26-ca.yaml` and `yolo26-scbconv-ca.yaml` replace the P3 fusion block with `C3k2SCBNeckCAPlus`, which is a composite refinement block containing more than coordinate attention. Thus A2 and A3 must be labeled as a CAPlus/composite P3 refinement intervention, not bare CA.

A5 to A6 is also not a perfectly strict BAC-only comparison: besides adding BAC/SCBConv neck blocks, the preserved YAMLs differ in early DA-LC block arguments. A5 uses the narrower early forms such as `[256, False, 0.25]` and `[512, False, 0.25]`, whereas A6 uses `[256, True]` and `[512, True]`. This changes shortcut/expansion behavior in addition to BAC placement. The row remains a valid configuration ablation, but it must not be called a strict single-factor BAC ablation.

A6 to A8 is especially clear. A6 uses at P3:

`C3k2SCBConv, [256, True]`

while A8 uses:

`C3k2SCBNeckCAPlus, [256, True]`

with the other DA-LC/BAC stages retained. Therefore A8 is a **composite P3 block replacement relative to A6**, not `A6 + bare CA`. A parameter/GFLOPs decrease is structurally possible because the old block is replaced rather than appended.

## Table 5

Table 5 is substantially stronger than the previous audit stated. Both the standard-SS2D + BAC + CAPlus YAML and the DA-LC-SS2D + BAC + CAPlus YAML are present in public Git history. Their inspected structures retain the same BAC/CAPlus neck and change the SS2D context family to DA-LC-SS2D, so this is a defensible structurally controlled comparison.

Its provenance should still be described precisely: YAML provenance is public and versioned; only the complete A8 checkpoint is part of the current public release. The standard-SS2D row does not have the same current checkpoint-level provenance as A8.

## Table 6

The correction above does **not** rescue the DA-only and LC-only variants.

The local audit material names modules such as `C3k2DASS2D`, `C2PSADASS2D`, `C3k2LCSS2D`, and `C2PSALCSS2D`. These classes are not present in the public release implementation available to the audit, and the corresponding local YAMLs fail to build under the public code (`KeyError` on the missing module names). No public Git evidence reviewed in this audit establishes those implementations.

Accordingly:

- standard SS2D row: PUBLIC_HISTORICAL structure;
- DA-LC row: PUBLIC_CURRENT/PUBLIC_HISTORICAL structure;
- DA-only: NOT_TRACEABLE;
- LC-only: NOT_TRACEABLE.

Because the two central rows of the four-way decomposition are not publicly reconstructable, Table 6 should be removed from the main evidentiary chain unless the original versioned source/classes are recovered. Retaining only standard vs DA-LC would duplicate Table 5.

## Table 8

The public DA-LC implementation confirms the default directional-gate hidden ratio `0.0625`. The public YAML/wrapper interface does not expose the non-default ratios used by the local R1/R3 configurations, and the public code does not provide the named R1/R3 wrapper classes used in those local YAMLs.

Therefore:

- 0.0625: CODE_CONFIRMED_DEFAULT / complete-model provenance;
- 0.03125: NOT_TRACEABLE from the public release/history reviewed here;
- 0.125: NOT_TRACEABLE from the public release/history reviewed here.

Table 8 should be removed or explicitly downgraded to an exploratory historical experiment unless the exact original implementation is recovered and versioned.

## Required manuscript action

1. Keep Table 4, but rename the `CA` factor to a clearly defined **CAPlus/composite P3 refinement** and state that Table 4 is a configuration ablation rather than a fully orthogonal factorial design.
2. Do not describe A8 as `A6 + CA`; describe it as replacement of the A6 P3 BAC fusion block by `C3k2SCBNeckCAPlus` while retaining the other DA-LC/BAC stages.
3. Prefer replacing Table 4/5 Params and GFLOPs with the uniform reproducible profiling values above, rounded consistently.
4. Keep Table 5 as a structurally controlled standard-SS2D versus DA-LC comparison, with a provenance note distinguishing historical YAML availability from checkpoint availability.
5. Remove Table 6 from the main paper unless DA-only/LC-only source provenance can be recovered.
6. Remove or downgrade Table 8 unless the R1/R3 implementation can be recovered.

## Answers to corrected A03

1. Table 4 A0-A8 configurations are represented in public Git history at `bcdf87...`; A8 is also current.
2. The later prune commit removed multiple ablation YAMLs; their absence from current HEAD does not erase public historical provenance.
3. A6 has a public historical YAML.
4. A7 has a public historical YAML.
5. Table 5 standard SS2D + BAC + CAPlus has a public historical YAML.
6. A6 to A8 is a composite P3 block replacement, not a bare additive CA step.
7. The repository profiler implementation is identical between release and current audit; the existing uniform profile provides the reproducible complexity baseline. Several paper complexity values, especially A6/A7, do not match that baseline.
8. A6 differs by -0.068M Params and about -0.321G GFLOPs from the paper value; A7 differs by about -0.072M and -0.316G. A8 differs only slightly (+0.003M and +0.034G).
9. DA-only/LC-only remain NOT_TRACEABLE in the public implementation/history reviewed.
10. R1/R3 (0.03125/0.125) remain NOT_TRACEABLE; 0.0625 is the public default.
11. Table 4 and Table 5 can be retained after relabeling and complexity normalization.
12. Table 4 requires labels/footnotes and removal of strict-factor wording.
13. Table 6 should be removed unless source provenance is recovered; Table 8 should be removed or downgraded similarly.

## Status

`[A03-GIT] PASS`

`[A03-PROFILE] PARTIAL` — a uniform reproducible baseline exists and profiler-version drift is ruled out, but the original execution record that produced every manuscript complexity number is unavailable.

`[A03] PARTIAL`

This is a provenance/wording/complexity-normalization issue. It does **not** justify retraining the detector.