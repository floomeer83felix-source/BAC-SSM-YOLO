#!/usr/bin/env python3
"""Create machine-readable A02 checkpoint/YAML/release provenance evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.tasks import DetectionModel  # noqa: E402


WEIGHTS = ROOT / "weights/BAC-SSM-YOLO_scb2_best.pt"
CONFIG = ROOT / "configs/models/yolo26-da-lc-ss2d-scbconv-ca.yaml"
PACKAGE_CONFIG = ROOT / "ultralytics/cfg/models/26/yolo26-da-lc-ss2d-scbconv-ca.yaml"
RELEASE = "bcdf87df9dfbe77e28ef310e0c7f04ca6e1cf5d8"
OUT = ROOT / "manuscript_verification/a02_architecture_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def git(*args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ROOT), *args], check=check, capture_output=True, text=True
    )
    return completed.stdout.strip()


def safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return str(value)


def main() -> None:
    checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    embedded_model = checkpoint.get("ema") or checkpoint.get("model")
    embedded_yaml = safe(getattr(embedded_model, "yaml", None))
    config_yaml = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    package_yaml = yaml.safe_load(PACKAGE_CONFIG.read_text(encoding="utf-8"))

    public_model = DetectionModel(str(CONFIG), nc=3, verbose=False)
    released_state = embedded_model.float().state_dict()
    public_state = public_model.state_dict()
    released_keys = set(released_state)
    public_keys = set(public_state)
    common = released_keys & public_keys
    shape_mismatches = {
        key: {"checkpoint": list(released_state[key].shape), "public_yaml": list(public_state[key].shape)}
        for key in sorted(common)
        if released_state[key].shape != public_state[key].shape
    }

    release_paths = [
        "weights/BAC-SSM-YOLO_scb2_best.pt",
        "configs/models/yolo26-da-lc-ss2d-scbconv-ca.yaml",
        "ultralytics/nn/modules/block.py",
        "ultralytics/nn/modules/mamba_ss2d.py",
    ]
    release_presence = {}
    for path in release_paths:
        probe = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", f"{RELEASE}:{path}"], capture_output=True
        )
        release_presence[path] = probe.returncode == 0

    code_paths = release_paths[1:]
    changes_since_release = git("diff", "--name-only", f"{RELEASE}..HEAD", "--", *code_paths)
    embedded_structure = {
        "backbone": embedded_yaml.get("backbone") if isinstance(embedded_yaml, dict) else None,
        "head": embedded_yaml.get("head") if isinstance(embedded_yaml, dict) else None,
    }
    public_structure = {"backbone": config_yaml["backbone"], "head": config_yaml["head"]}

    evidence = {
        "checkpoint_sha256": sha256(WEIGHTS),
        "checkpoint_train_model_argument": safe((checkpoint.get("train_args") or {}).get("model")),
        "checkpoint_embedded_yaml_file": embedded_yaml.get("yaml_file") if isinstance(embedded_yaml, dict) else None,
        "checkpoint_embedded_nc": embedded_yaml.get("nc") if isinstance(embedded_yaml, dict) else None,
        "public_yaml_nc": config_yaml.get("nc"),
        "config_yaml_sha256": sha256(CONFIG),
        "package_yaml_sha256": sha256(PACKAGE_CONFIG),
        "two_public_yamls_byte_identical": CONFIG.read_bytes() == PACKAGE_CONFIG.read_bytes(),
        "embedded_backbone_head_equals_public": embedded_structure == public_structure,
        "state_dict_compatibility": {
            "checkpoint_keys": len(released_keys),
            "public_yaml_keys": len(public_keys),
            "missing_from_public_yaml": sorted(released_keys - public_keys),
            "missing_from_checkpoint": sorted(public_keys - released_keys),
            "shape_mismatches": shape_mismatches,
            "exact_key_and_shape_match": (
                released_keys == public_keys and not shape_mismatches
            ),
        },
        "release_commit": RELEASE,
        "release_commit_summary": git("show", "-s", "--format=%H%n%ad%n%s", "--date=iso-strict", RELEASE),
        "release_presence": release_presence,
        "architecture_code_changes_since_release": changes_since_release.splitlines() if changes_since_release else [],
        "checkpoint_contains_git_commit_metadata": any(
            key.lower() in {"git", "commit", "git_commit", "sha", "revision"} for key in checkpoint
        ),
        "checkpoint_git_metadata": safe(checkpoint.get("git")),
        "checkpoint_keys": sorted(checkpoint),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(evidence, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
