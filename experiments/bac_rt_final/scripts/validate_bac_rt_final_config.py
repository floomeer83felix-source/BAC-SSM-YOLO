#!/usr/bin/env python3
"""Validate that ConvEarly only restores native YOLO26 P2/P3 blocks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402


BASE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
FULL = ROOT / "ultralytics/cfg/models/26/yolo26-da-lc-ss2d-scbconv-ca.yaml"
CANDIDATE = ROOT / "experiments/bac_rt_final/config/bac_ssm_yolo_rt_convearly.yaml"
OUT = ROOT / "experiments/bac_rt_final/config/architecture_validation.json"


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    base = load(BASE)
    full = load(FULL)
    candidate = load(CANDIDATE)
    base_layers = base["backbone"] + base["head"]
    full_layers = full["backbone"] + full["head"]
    candidate_layers = candidate["backbone"] + candidate["head"]
    for key in ("nc", "end2end", "reg_max", "scales"):
        if candidate[key] != full[key]:
            raise AssertionError(f"Forbidden top-level difference: {key}")
    if not (len(base_layers) == len(full_layers) == len(candidate_layers)):
        raise AssertionError("Layer count differs")

    changed = {index for index, pair in enumerate(zip(full_layers, candidate_layers)) if pair[0] != pair[1]}
    if changed != {2, 4}:
        raise AssertionError(f"ConvEarly changes layers {sorted(changed)} instead of only [2, 4]")
    for index in (2, 4):
        if candidate_layers[index] != base_layers[index]:
            raise AssertionError(f"Layer {index} does not exactly match native YOLO26")
    for index in set(range(len(full_layers))) - {2, 4}:
        if candidate_layers[index] != full_layers[index]:
            raise AssertionError(f"Unexpected difference at layer {index}")

    dalc_count = sum(layer[2] in {"C3k2DALCSS2D", "C2PSADALCSS2D"} for layer in candidate_layers)
    standard_ss2d_count = sum(layer[2] in {"C3k2SS2D", "C2PSASS2D"} for layer in candidate_layers)
    early_conv_count = sum(candidate_layers[index][2] == "C3k2" for index in (2, 4))
    if (dalc_count, standard_ss2d_count, early_conv_count) != (4, 0, 2):
        raise AssertionError(
            f"Unexpected block counts: DA-LC={dalc_count}, standard SS2D={standard_ss2d_count}, early conv={early_conv_count}"
        )

    model = YOLO(str(CANDIDATE)).model.cuda().eval()
    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    gflops = float(get_flops(model, imgsz=640))
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640, device="cuda"))
    report = {
        "original_YOLO26_P2_block": base_layers[2],
        "original_YOLO26_P3_block": base_layers[4],
        "selected_ConvEarly_P2_block": candidate_layers[2],
        "selected_ConvEarly_P3_block": candidate_layers[4],
        "changed_layer_indices": sorted(changed),
        "number_of_DA_LC_blocks": dalc_count,
        "number_of_standard_SS2D_blocks": standard_ss2d_count,
        "number_of_early_convolutional_blocks": early_conv_count,
        "params_m_before_nc_override": params_m,
        "gflops_640_before_nc_override": gflops,
        "forward_output_type": type(output).__name__,
        "validation": "PASS",
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
