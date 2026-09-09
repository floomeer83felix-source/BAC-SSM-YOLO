# A02 Findings: architecture, YAML, and code provenance

## Checkpoint-to-YAML mapping

The checkpoint train arguments and embedded `yaml_file` both name `ultralytics/cfg/models/26/yolo26-da-lc-ss2d-scbconv-ca.yaml`. Its embedded backbone and head lists exactly equal the current public YAML structure. The only expected dataset-time difference is `nc=3` in the checkpoint versus the reusable public default `nc=80`.

The following public files are byte-identical (SHA256 `f68b3c390d43992e24457009d4ec04f3d892080640c132ed0ee9808262fe4d4d`):

- `configs/models/yolo26-da-lc-ss2d-scbconv-ca.yaml`
- `ultralytics/cfg/models/26/yolo26-da-lc-ss2d-scbconv-ca.yaml`

A fresh `nc=3`, scale-n model constructed from the public YAML has 807 state-dict keys. The released checkpoint also has 807 keys, with no missing keys and no shape mismatches. This is strong evidence that the checkpoint architecture matches the public DA-LC-SS2D + SCBConv + CAPlus YAML.

## Release provenance

Commit `bcdf87df9dfbe77e28ef310e0c7f04ca6e1cf5d8` (`Release BAC-SSM-YOLO code and weights`, 2026-06-01) simultaneously contains:

- `weights/BAC-SSM-YOLO_scb2_best.pt`
- `configs/models/yolo26-da-lc-ss2d-scbconv-ca.yaml`
- `ultralytics/nn/modules/block.py`
- `ultralytics/nn/modules/mamba_ss2d.py`

There are no changes to the three architecture/code files between that release commit and current HEAD. However, the checkpoint's `git` field contains `commit=None`, `branch=None`, and `origin=None`. Therefore, the public release proves co-publication and exact state-dict compatibility, but the checkpoint alone cannot prove the exact source commit used during training.

## DA-LC-SS2D implementation

The four scan sequences are row-major, column-major, reversed row-major, and reversed column-major. Before fusion, reverse scans are flipped back and column-major scans are reshaped and transposed to the row-major spatial layout.

Direction weights are generated per image by global average pooling of the inner feature, followed by a `1x1 Conv -> SiLU -> 1x1 Conv` gate and softmax over four directions. The hidden width is `max(4, int(d_inner * 0.0625))`. The final gate convolution is zero-initialized, so the initial softmax weights are uniform.

Fusion is neither an unscaled mean nor a plain sum. It is `4 * sum(alpha_d * aligned_direction_d)`. The factor four makes uniform weights equivalent to the original SS2D four-direction sum. The fused sequence is processed by `out_norm` before returning to the spatial layout.

Local compensation uses a depthwise `3x3` convolution. Its weights and bias are first zeroed, after which the center coefficient is set to one; the resulting kernel is identity-initialized, not zero-initialized. The scalar `local_scale` is the parameter that is truly initialized to zero. The local path is added as `y + local_scale * local_comp(x)`.

## P3 refinement implementation

The P3 module is `C3k2SCBNeckCAPlus`, a composite refinement block rather than a simple serial “BAC + Coordinate Attention” module. Each internal bottleneck contains:

- a local `3x3` convolution;
- depthwise `3x3` and `5x5` branches;
- a sequential asymmetric `1x3` then `3x1` branch;
- a factorized `1x7` then `7x1` context branch;
- channel concatenation and `1x1` fusion;
- coordinate attention with reduction 16;
- a spatial gate built from channel mean/max followed by a `7x7` convolution;
- ECA with kernel size 5;
- a final `3x3` refinement convolution;
- a residual shortcut when enabled.

The CAPlus block does not enable the optional fixed Sobel edge branch used by other SCBConv variants.

## Required manuscript corrections

Statements that describe DA-LC fusion as a simple average, omit the factor of four, omit `out_norm`, say that the local convolution remains zero-initialized, or describe P3 as only “BAC + Coordinate Attention” must be corrected. The method should identify P3 as a composite multi-branch, coordinate/channel/spatial recalibration and refinement block. Coordinate attention, ECA, and the spatial gate must not be conflated under the single label CA without definition.

The following English claims are supported and may be retained in an Author verification draft:

- direction weights are image-wise, softmax-normalized, and initialized uniformly;
- the factor-four weighted fusion initially preserves standard SS2D summation;
- local compensation is gated by a learnable scalar initialized to zero;
- P3 combines local, depthwise multi-scale, asymmetric, factorized large-kernel context, coordinate attention, ECA, spatial gating, refinement, and a residual path;
- the released checkpoint is exactly key/shape compatible with the public scale-n, `nc=3` architecture.

The actual Author verification draft is not present in this repository, so sentence-level approval of that document is `UNKNOWN`. Also, because checkpoint Git metadata does not contain a commit SHA, code inspection alone cannot establish that training executed this exact source checkout; it establishes architecture compatibility and release provenance, not the unavailable training-worktree identity.

## Answers to A02

1. The checkpoint matches `ultralytics/cfg/models/26/yolo26-da-lc-ss2d-scbconv-ca.yaml`; its copy under `configs/models/` is identical.
2. The embedded backbone/head and public YAML agree exactly; `nc=3` is the expected dataset override of public `nc=80`.
3. The first release commit contains the checkpoint and all specified architecture files together.
4. The manuscript must use the implementation details above and remove simplified or incorrect descriptions of fusion, initialization, and P3 composition.
5. Only the explicitly listed code-supported English claims can be retained without further evidence.
6. The exact training code revision remains unprovable because the checkpoint's Git commit field is null, despite exact public architecture compatibility.

## Status

`[A02] PASS` for public architecture mapping and release provenance, with a clearly bounded `UNKNOWN` for the exact training-worktree commit.
