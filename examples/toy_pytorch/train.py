# File: examples/toy_pytorch/train.py
# Purpose: Provide a dependency-free toy training target that emits the expected results.json contract.
# License: GPL-3.0-or-later
"""Dependency-free toy training script for the MVP study loop."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import yaml


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def score_config(cfg: dict) -> tuple[float, dict[str, float]]:
    lr = float(cfg["optimizer"]["lr"])
    wd = float(cfg["optimizer"]["weight_decay"])
    batch_size = int(cfg["train"]["batch_size"])
    dropout = float(cfg["model"]["dropout"])

    target_lr = 2.5e-4
    target_wd = 1e-4
    target_bs = 16
    target_dropout = 0.2
    penalty = (
        abs(math.log10(lr) - math.log10(target_lr)) * 0.08
        + abs(math.log10(wd) - math.log10(target_wd)) * 0.04
        + abs(batch_size - target_bs) * 0.01
        + abs(dropout - target_dropout) * 0.5
    )
    base = 0.9 - penalty
    noise = random.Random(17).uniform(-0.005, 0.005)
    val_auc = max(0.0, min(1.0, base + noise))
    val_f1 = max(0.0, min(1.0, val_auc - 0.06))
    train_loss_final = max(0.01, 0.4 - val_auc * 0.2)
    return val_auc, {
        "val_auc": val_auc,
        "val_f1": val_f1,
        "train_loss_final": train_loss_final,
        "best_epoch": 17,
    }


def maybe_fail(cfg: dict) -> str | None:
    batch_size = int(cfg["train"]["batch_size"])
    dropout = float(cfg["model"]["dropout"])
    lr = float(cfg["optimizer"]["lr"])
    if batch_size == 32 and dropout > 0.42:
        return "oom: batch size too large for the toy budget"
    if lr > 4.5e-4 and dropout > 0.45:
        return "nan encountered in loss"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config_path)

    time.sleep(0.2)
    failure = maybe_fail(cfg)
    if failure is not None:
        payload = {
            "primary_metric": None,
            "metrics": {},
            "status": "failed",
            "notes": failure,
        }
        (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
        print(failure)
        return 1

    primary_metric, metrics = score_config(cfg)
    payload = {
        "primary_metric": primary_metric,
        "metrics": metrics,
        "status": "success",
        "notes": "early_stop_triggered",
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
