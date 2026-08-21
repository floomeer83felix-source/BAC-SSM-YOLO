#!/usr/bin/env python3
"""Assert that stage-2 configs differ from full BAC only at allowed early DA-LC placements."""

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

FULL = ROOT / "ultralytics/cfg/models/26/yolo26-da-lc-ss2d-scbconv-ca.yaml"
CANDIDATES = {
    "BAC-SSM-YOLO-RT-NoP2": (
        ROOT / "ultralytics/cfg/models/26/bac_rt_stage2/bac_ssm_yolo_rt_nop2.yaml",
        {2},
        5,
    ),
    "BAC-SSM-YOLO-RT-NoEarly": (
        ROOT / "ultralytics/cfg/models/26/bac_rt_stage2/bac_ssm_yolo_rt_noearly.yaml",
        {2, 4},
        4,
    ),
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    full = load(FULL)
    full_layers = full["backbone"] + full["head"]
    report = {}
    for name, (path, expected_layers, expected_dalc) in CANDIDATES.items():
        candidate = load(path)
        for key in ("nc", "end2end", "reg_max", "scales"):
            if candidate[key] != full[key]:
                raise AssertionError(f"{name}: forbidden top-level difference in {key}")
        candidate_layers = candidate["backbone"] + candidate["head"]
        if len(candidate_layers) != len(full_layers):
            raise AssertionError(f"{name}: layer count changed")
        changed = set()
        for index, (reference, actual) in enumerate(zip(full_layers, candidate_layers)):
            if reference != actual:
                changed.add(index)
                expected = list(reference)
                expected[2] = "C3k2SS2D"
                if actual != expected:
                    raise AssertionError(f"{name}: layer {index} changed beyond DA-LC -> standard SS2D")
        if changed != expected_layers:
            raise AssertionError(f"{name}: changed layers {changed}, expected {expected_layers}")
        dalc_count = sum(layer[2] in {"C3k2DALCSS2D", "C2PSADALCSS2D"} for layer in candidate_layers)
        if dalc_count != expected_dalc:
            raise AssertionError(f"{name}: DA-LC count {dalc_count}, expected {expected_dalc}")
        model = YOLO(str(path)).model.cuda().eval()
        params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        gflops = float(get_flops(model, imgsz=640))
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 640, 640, device="cuda"))
        report[name] = {
            "config": str(path),
            "changed_layer_indices": sorted(changed),
            "num_DA_LC_blocks": dalc_count,
            "params_m": params_m,
            "gflops_640": gflops,
            "forward_output_type": type(output).__name__,
            "validation": "PASS",
        }
        del output, model
        torch.cuda.empty_cache()
    out = ROOT / "runs/bac_rt_stage2/config_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
