#!/usr/bin/env python3
"""Validate the two classroom-baseline reimplementations without training."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules import AIFI, C2fWADCA, C3k2PConv, Detect, DySample, LSKA, TwoDPEMHA  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

CONFIGS = {
    "PLA-YOLO11n": EXPERIMENT_ROOT / "configs/pla_yolo11n_scb2.yaml",
    "WAD-YOLOv8n": EXPERIMENT_ROOT / "configs/wad_yolov8n_scb2.yaml",
}
EXPECTED = {
    "PLA-YOLO11n": {"detect_scales": 4, "feature_sizes": [[160, 160], [80, 80], [40, 40], [20, 20]]},
    "WAD-YOLOv8n": {"detect_scales": 3, "feature_sizes": [[80, 80], [40, 40], [20, 20]]},
}


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def flatten_tensors(value, path="output"):
    tensors = []
    if isinstance(value, torch.Tensor):
        tensors.append((path, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            tensors.extend(flatten_tensors(item, f"{path}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            tensors.extend(flatten_tensors(item, f"{path}[{index}]"))
    return tensors


def first_instance(model, module_type):
    return next((module for module in model.modules() if isinstance(module, module_type)), None)


def count_modules(model, module_type) -> int:
    return sum(isinstance(module, module_type) for module in model.modules())


def constructor_validation(name: str, model) -> dict:
    if name == "PLA-YOLO11n":
        pconv = first_instance(model, C3k2PConv).constructor_args
        lska = first_instance(model, LSKA).constructor_args
        return {
            "C3k2PConv": {"actual": pconv, "correct": pconv == {"c1": 32, "c2": 64, "n": 1, "c3k": False, "e": 0.25, "n_div": 4}},
            "LSKA": {"actual": lska, "correct": lska == {"c": 32, "k": 7, "dilation": 2}},
        }
    wadca = first_instance(model, C2fWADCA).constructor_args
    mha = first_instance(model, TwoDPEMHA).constructor_args
    return {
        "C2fWADCA": {
            "actual": wadca,
            "correct": wadca == {"c1": 32, "c2": 32, "n": 1, "shortcut": True, "g": 1, "e": 0.5, "reduction": 16},
        },
        "TwoDPEMHA": {"actual": mha, "correct": mha == {"c": 256, "num_heads": 8, "dropout": 0.0}},
    }


def verify_model(name: str, config: Path, device: torch.device) -> dict:
    print(f"\n=== {name} ===", flush=True)
    wrapper = YOLO(str(config), task="detect")
    model = wrapper.model
    instantiated_on_cpu = next(model.parameters()).device.type == "cpu"
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    model.eval()
    gflops = float(get_flops(model, imgsz=640))

    detect = first_instance(model, Detect)
    if detect is None:
        raise RuntimeError(f"{name}: Detect module not found")
    detect_inputs = []
    mha_inputs = []

    def detect_hook(_module, inputs):
        features = inputs[0]
        detect_inputs[:] = [list(feature.shape) for feature in features]

    def mha_hook(_module, inputs):
        mha_inputs.append(list(inputs[0].shape))

    handles = [detect.register_forward_pre_hook(detect_hook)]
    for module in model.modules():
        if isinstance(module, TwoDPEMHA):
            handles.append(module.register_forward_pre_hook(mha_hook))

    model = model.to(device).eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(1, 3, 640, 640, device=device)
    try:
        with torch.no_grad():
            output = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        forward_passed = True
        error = None
    except Exception as exc:
        output = None
        forward_passed = False
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()

    tensors = flatten_tensors(output) if forward_passed else []
    has_nan = any(torch.isnan(tensor).any().item() for _, tensor in tensors if tensor.is_floating_point())
    has_inf = any(torch.isinf(tensor).any().item() for _, tensor in tensors if tensor.is_floating_point())
    output_shapes = [{"path": path, "shape": list(tensor.shape), "dtype": str(tensor.dtype)} for path, tensor in tensors]
    feature_sizes = [[shape[-2], shape[-1]] for shape in detect_inputs]
    feature_channels = [shape[1] for shape in detect_inputs]
    measured_strides = [640 // size[0] for size in feature_sizes]
    expected = EXPECTED[name]
    structure_passed = len(detect_inputs) == expected["detect_scales"] and feature_sizes == expected["feature_sizes"]
    constructor_args = constructor_validation(name, model)
    constructors_passed = all(item["correct"] for item in constructor_args.values())
    peak_memory_mib = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None

    counts = {
        "C3k2PConv": count_modules(model, C3k2PConv),
        "AIFI": count_modules(model, AIFI),
        "LSKA": count_modules(model, LSKA),
        "C2fWADCA": count_modules(model, C2fWADCA),
        "TwoDPEMHA": count_modules(model, TwoDPEMHA),
        "DySample": count_modules(model, DySample),
    }
    report = {
        "config": str(config.relative_to(REPO_ROOT)),
        "instantiate": True,
        "instantiated_on_cpu": instantiated_on_cpu,
        "forward": forward_passed,
        "error": error,
        "device": str(device),
        "input_shape": [1, 3, 640, 640],
        "params": params,
        "params_m": params / 1e6,
        "trainable_params": trainable,
        "trainable_params_m": trainable / 1e6,
        "gflops_640": gflops,
        "detect_scales": len(detect_inputs),
        "detect_feature_shapes": detect_inputs,
        "detect_feature_channels": feature_channels,
        "detect_feature_sizes": feature_sizes,
        "detect_strides": measured_strides,
        "model_detect_strides": [float(value) for value in detect.stride.detach().cpu().tolist()],
        "module_counts": counts,
        "constructor_validation": constructor_args,
        "two_dpemha_input_shapes": mha_inputs,
        "output_type": type(output).__name__ if output is not None else None,
        "output_shapes": output_shapes,
        "nan": has_nan,
        "inf": has_inf,
        "peak_gpu_memory_mib": peak_memory_mib,
        "structure_passed": structure_passed,
        "constructors_passed": constructors_passed,
        "smoke_test_passed": forward_passed and not has_nan and not has_inf and structure_passed and constructors_passed,
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


def build_markdown(payload: dict) -> str:
    environment = payload["environment"]
    lines = [
        "# Classroom baseline smoke-test report",
        "",
        "> These are paper-faithful reimplementations based on the published method descriptions and are not claimed to be the authors' official source code.",
        "",
        "## Environment",
        "",
        f"- Python: {environment['python']}",
        f"- PyTorch: {environment['pytorch']}",
        f"- CUDA: {environment['cuda']}",
        f"- GPU: {environment['gpu']}",
        f"- Ultralytics: {environment['ultralytics']}",
        f"- Git commit: {environment['git_commit']}",
        "- Formal training executed: NO",
        "- Paper modified: NO",
    ]
    for name, result in payload["models"].items():
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"- Instantiation: {'PASS' if result['instantiate'] else 'FAIL'} (CPU: {result['instantiated_on_cpu']})",
                f"- Forward: {'PASS' if result['forward'] else 'FAIL'}",
                f"- Input: {result['input_shape']}",
                f"- Params: {result['params']:,} ({result['params_m']:.6f} M)",
                f"- Trainable Params: {result['trainable_params']:,} ({result['trainable_params_m']:.6f} M)",
                f"- GFLOPs @640: {result['gflops_640']:.6f}",
                f"- Detect scales: {result['detect_scales']}",
                f"- Detect feature sizes: {result['detect_feature_sizes']}",
                f"- Detect channels: {result['detect_feature_channels']}",
                f"- Detect strides: {result['detect_strides']}",
                f"- Module counts: {result['module_counts']}",
                f"- Constructor validation: {result['constructor_validation']}",
                f"- TwoDPEMHA input shape: {result['two_dpemha_input_shapes'] or 'N/A'}",
                f"- Output type: {result['output_type']}",
                f"- Output shapes: {result['output_shapes']}",
                f"- NaN: {'YES' if result['nan'] else 'NO'}",
                f"- Inf: {'YES' if result['inf'] else 'NO'}",
                f"- Peak GPU memory: {result['peak_gpu_memory_mib'] if result['peak_gpu_memory_mib'] is not None else 'N/A'} MiB",
                f"- Smoke test: {'PASS' if result['smoke_test_passed'] else 'FAIL'}",
            ]
        )
        if name == "PLA-YOLO11n":
            difference = result["params_m"] - 3.08
            lines.extend(
                [
                    "- Paper-reported Params: approximately 3.08 M",
                    f"- Reimplementation difference: {difference:+.6f} M ({difference / 3.08 * 100:+.2f}%)",
                    "- Difference may reflect LSKA/PConv details, four-scale neck reconstruction, and Ultralytics version.",
                ]
            )
    lines.extend(
        [
            "",
            "## Final status",
            "",
            f"- PLA smoke test passed: {'YES' if payload['models']['PLA-YOLO11n']['smoke_test_passed'] else 'NO'}",
            f"- WAD smoke test passed: {'YES' if payload['models']['WAD-YOLOv8n']['smoke_test_passed'] else 'NO'}",
            f"- Ready for training review: {'YES' if payload['ready_for_training_review'] else 'NO'}",
            "- Formal training executed: NO",
            "- Paper modified: NO",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "git_commit": git_commit(),
    }
    models = {name: verify_model(name, config, device) for name, config in CONFIGS.items()}
    payload = {
        "statement": "These are paper-faithful reimplementations based on the published method descriptions and are not claimed to be the authors' official source code.",
        "environment": environment,
        "models": models,
        "parser_validation": {
            "C3k2PConv": models["PLA-YOLO11n"]["constructor_validation"]["C3k2PConv"]["correct"],
            "C2fWADCA": models["WAD-YOLOv8n"]["constructor_validation"]["C2fWADCA"]["correct"],
            "LSKA": models["PLA-YOLO11n"]["constructor_validation"]["LSKA"]["correct"],
            "TwoDPEMHA": models["WAD-YOLOv8n"]["constructor_validation"]["TwoDPEMHA"]["correct"],
        },
        "ready_for_training_review": all(result["smoke_test_passed"] for result in models.values()),
        "formal_training_executed": False,
        "paper_modified": False,
    }
    json_path = EXPERIMENT_ROOT / "smoke_test_report.json"
    md_path = EXPERIMENT_ROOT / "smoke_test_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(build_markdown(payload), encoding="utf-8")
    print(f"\nReports written: {md_path} and {json_path}")
    if not payload["ready_for_training_review"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
