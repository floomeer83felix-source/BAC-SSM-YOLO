#!/usr/bin/env python3
"""Audit ablation YAML traceability and profile model complexity without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops


PAPER_ROWS = [
    ("YAML inventory", "YOLO26s", "yolo26s.yaml", None, None, "N/A"),
    ("Table 4 A0", "YOLO26n", "yolo26.yaml", 2.51, 5.78, "YES"),
    ("Table 4 A1", "YOLO26n + BAC", "yolo26-scbconv.yaml", 3.14, 6.96, "YES"),
    ("Table 4 A2", "YOLO26n + CA", "yolo26-ca.yaml", 2.53, 6.04, "NO"),
    ("Table 4 A3", "YOLO26n + BAC + CA", "yolo26-scbconv-ca.yaml", 3.16, 7.23, "NO"),
    ("Table 4 A4", "YOLO26n + standard SS2D", "yolo26-ss2d.yaml", 4.79, 14.24, "YES"),
    ("Table 4 A5", "YOLO26n + DA-LC-SS2D", "yolo26-da-lc-ss2d.yaml", 4.84, 14.35, "YES"),
    ("Table 4 A6", "YOLO26n + DA-LC-SS2D + BAC", "yolo26-da-lc-ss2d-scbconv.yaml", 5.42, 15.76, "NO"),
    ("Table 4 A7", "YOLO26n + DA-LC-SS2D + CA", "yolo26-da-lc-ss2d-ca.yaml", 4.93, 14.96, "NO"),
    ("Table 4 A8", "BAC-SSM-YOLO", "yolo26-da-lc-ss2d-scbconv-ca.yaml", 5.34, 15.28, "NO"),
    ("Table 5 row 1", "standard SS2D + BAC + CA", "yolo26-ss2d-scbconv-ca.yaml", 5.30, 15.30, "YES"),
    ("Table 5 row 2", "DA-LC-SS2D + BAC + CA", "yolo26-da-lc-ss2d-scbconv-ca.yaml", 5.34, 15.28, "YES"),
    ("YAML inventory", "standard SS2D + CA", "yolo26-ss2d-ca.yaml", None, None, "UNCLEAR"),
    ("YAML inventory", "standard SS2D + BAC", "yolo26-ss2d-scbconv.yaml", None, None, "UNCLEAR"),
    ("Table 6 row 1", "standard SS2D + BAC + CA", "editor_ablation/01_standard_ss2d_scbconv_ca.yaml", 5.16, 14.63, "YES"),
    ("Table 6 row 2", "DA-only SS2D + BAC + CA", "editor_ablation/02_da_only_ss2d_scbconv_ca.yaml", 5.22, 14.66, "YES"),
    ("Table 6 row 3", "LC-only SS2D + BAC + CA", "editor_ablation/03_lc_only_ss2d_scbconv_ca.yaml", 5.22, 14.74, "YES"),
    ("Table 6 row 4", "DA-LC-SS2D + BAC + CA", "editor_ablation/04_da_lc_ss2d_scbconv_ca.yaml", 5.22, 14.77, "YES"),
    ("Table 8 R1", "gate ratio 0.03125", "editor_ablation/09_gate_ratio_r1_003125_scbconv_ca.yaml", None, None, "YES"),
    ("Table 8 R2", "gate ratio 0.0625", "yolo26-da-lc-ss2d-scbconv-ca.yaml", 5.34, 15.28, "YES"),
    ("Table 8 R3", "gate ratio 0.125", "editor_ablation/10_gate_ratio_r3_0125_scbconv_ca.yaml", None, None, "YES"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_output(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def git_commit_for(repo: Path, name: str) -> str | None:
    candidates = [f"configs/models/{Path(name).name}", f"ultralytics/cfg/models/26/{name}"]
    for candidate in candidates:
        try:
            value = git_output(repo, "log", "--all", "-1", "--format=%H", "--", candidate)
        except subprocess.CalledProcessError:
            value = ""
        if value:
            return value
    return None


def architecture_summary(path: Path) -> str:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    layers = data.get("backbone", []) + data.get("head", [])
    selected = []
    for index, layer in enumerate(layers):
        source, repeat, module, args = layer
        module = str(module)
        if module.startswith("C3k2") or module.startswith("C2PSA"):
            selected.append(f"L{index}:{module}(r={repeat},args={args})")
    return "; ".join(selected)


def profile(path: Path) -> tuple[float | None, float | None, float | None, str | None]:
    try:
        model = DetectionModel(str(path), nc=3, verbose=False)
        total = sum(p.numel() for p in model.parameters()) / 1e6
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        flops = float(get_flops(model, imgsz=640))
        return total, trainable, flops, None
    except Exception as exc:  # Preserve unsupported historical variants as evidence.
        return None, None, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--legacy-yaml-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    public_full = repo / "configs/models/yolo26-da-lc-ss2d-scbconv-ca.yaml"
    records = []

    for paper_row, label, name, paper_params, paper_flops, strict in PAPER_ROWS:
        commit = git_commit_for(repo, name)
        legacy_path = args.legacy_yaml_root / name
        is_public_full = Path(name).name == public_full.name and "/" not in name
        if is_public_full and public_full.exists():
            path = public_full
            traceability = "CONFIRMED"
            source = "public repository"
        elif legacy_path.exists():
            path = legacy_path
            traceability = "PARTIAL"
            source = "local legacy experiment directory; absent from public Git history"
        else:
            path = None
            traceability = "NOT_TRACEABLE"
            source = "not found"

        summary = ""
        total = trainable = flops = None
        error = None
        digest = None
        if path:
            digest = sha256(path)
            try:
                summary = architecture_summary(path)
            except Exception as exc:
                error = f"summary: {type(exc).__name__}: {exc}"
            total, trainable, flops, profile_error = profile(path)
            if profile_error:
                error = f"{error}; {profile_error}" if error else profile_error

        notes = source
        if paper_row == "Table 4 A2":
            notes += "; CAPlus is a composite P3 replacement, not a bare CA insertion"
        if paper_row == "Table 4 A3":
            notes += "; combines SCBConv backbone replacement with composite P3 CAPlus replacement"
        if paper_row == "Table 4 A6":
            notes += "; BAC placement differs from A1 (neck/context fusion rather than BAC backbone)"
        if paper_row == "Table 4 A8":
            notes += "; relative to A6, P3 C3k2SCBConv is replaced by C3k2SCBNeckCAPlus"
        if "Table 6 row 2" in paper_row or "Table 6 row 3" in paper_row:
            traceability = "NOT_TRACEABLE"
            notes += "; NOT REPRODUCIBLE FROM PUBLIC REPOSITORY"
        if paper_row in {"Table 8 R1", "Table 8 R3"}:
            traceability = "NOT_TRACEABLE"
            notes += "; NOT TRACEABLE FROM PUBLIC RELEASE"
        if error:
            notes += f"; profile failed under public code revision: {error}"

        records.append(
            {
                "paper_row": paper_row,
                "paper_label": label,
                "yaml_path": str(path) if path else "",
                "yaml_sha256": digest or "",
                "git_commit": commit or "",
                "architecture_summary": summary,
                "params_paper_M": paper_params if paper_params is not None else "",
                "params_profiled_M": round(total, 6) if total is not None else "",
                "trainable_params_profiled_M": round(trainable, 6) if trainable is not None else "",
                "gflops_paper": paper_flops if paper_flops is not None else "",
                "gflops_profiled": round(flops, 6) if flops is not None else "",
                "strict_single_factor": strict,
                "traceability": traceability,
                "notes": notes,
            }
        )

    csv_path = output / "A03_ABLATION_MATRIX.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "repository_head": git_output(repo, "rev-parse", "HEAD"),
        "profiling_code_revision": git_output(repo, "rev-parse", "HEAD"),
        "image_size": 640,
        "classes": 3,
        "profiling_implementation": "ultralytics.utils.torch_utils.get_flops",
        "records": records,
    }
    (output / "a03_profile_details.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
