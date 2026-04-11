#!/usr/bin/env python3
"""Qwen3-ASR adapter for ClosedLoopExperimentAgent.

This wrapper converts a closed-loop study config into a Qwen3-ASR training
run, then evaluates the selected checkpoint on the dev set and writes a
results.json file that CLAE can consume.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Config file must contain a mapping: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for item in output_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                checkpoints.append((int(item.name.split("-", 1)[1]), item))
            except Exception:
                continue
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda pair: pair[0])
    return checkpoints[-1][1]


def _load_selected_checkpoint(output_dir: Path) -> Path | None:
    best_file = output_dir / "best_checkpoint.json"
    if best_file.exists():
        payload = _read_json(best_file)
        candidate = payload.get("best_checkpoint")
        if candidate:
            path = Path(candidate)
            if not path.is_absolute():
                path = output_dir / candidate
            if path.exists():
                return path
    return _latest_checkpoint(output_dir)


def _run(cmd: list[str], cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=stdout_handle, stderr=stderr_handle, check=False)
    return int(proc.returncode)


def build_train_command(qwen_root: Path, config: dict[str, Any], run_dir: Path) -> list[str]:
    train_cfg = config.get("train", {})
    augmentation_config = config.get("augmentation_config")
    augmentation_config_data = config.get("augmentation_config_data")
    augmentation_config_base = _as_str(config.get("augmentation_config_base"), "")
    augmentation_config_overrides = config.get("augmentation_config_overrides")
    hard_lexicon_config = config.get("hard_lexicon_config")
    hard_lexicon_config_data = config.get("hard_lexicon_config_data")
    hard_lexicon_config_base = _as_str(config.get("hard_lexicon_config_base"), "")
    hard_lexicon_config_overrides = config.get("hard_lexicon_config_overrides")

    rendered_augmentation_config = ""
    if isinstance(augmentation_config_data, dict) and augmentation_config_data:
        rendered_augmentation_config = str(run_dir / "rendered_augmentation_config.json")
        _write_json(Path(rendered_augmentation_config), augmentation_config_data)
    elif augmentation_config_base and isinstance(augmentation_config_overrides, dict) and augmentation_config_overrides:
        base_path = Path(augmentation_config_base)
        base_data = json.loads(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_data, dict):
            raise SystemExit(f"augmentation_config_base must point to a JSON object: {base_path}")
        rendered_augmentation_config = str(run_dir / "rendered_augmentation_config.json")
        _write_json(Path(rendered_augmentation_config), _deep_merge(base_data, augmentation_config_overrides))
    elif isinstance(augmentation_config, str) and augmentation_config.strip():
        rendered_augmentation_config = augmentation_config.strip()

    rendered_hard_lexicon_config = ""
    if isinstance(hard_lexicon_config_data, dict) and hard_lexicon_config_data:
        rendered_hard_lexicon_config = str(run_dir / "rendered_hard_lexicon_config.json")
        _write_json(Path(rendered_hard_lexicon_config), hard_lexicon_config_data)
    elif hard_lexicon_config_base and isinstance(hard_lexicon_config_overrides, dict) and hard_lexicon_config_overrides:
        base_path = Path(hard_lexicon_config_base)
        base_data = json.loads(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_data, dict):
            raise SystemExit(f"hard_lexicon_config_base must point to a JSON object: {base_path}")
        rendered_hard_lexicon_config = str(run_dir / "rendered_hard_lexicon_config.json")
        _write_json(Path(rendered_hard_lexicon_config), _deep_merge(base_data, hard_lexicon_config_overrides))
    elif isinstance(hard_lexicon_config, str) and hard_lexicon_config.strip():
        rendered_hard_lexicon_config = hard_lexicon_config.strip()

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=1",
        str(qwen_root / "tools" / "qwen3_asr_sft.py"),
        "--model_path",
        _as_str(config.get("model_path")),
        "--train_file",
        _as_str(config.get("train_file")),
        "--eval_file",
        _as_str(config.get("eval_file")),
        "--output_dir",
        str(run_dir),
        "--precision",
        _as_str(train_cfg.get("precision", "bf16")),
        "--batch_size",
        str(_as_int(train_cfg.get("batch_size"), 1)),
        "--grad_acc",
        str(_as_int(train_cfg.get("grad_acc"), 1)),
        "--lr",
        str(train_cfg.get("lr", 8e-6)),
        "--epochs",
        str(train_cfg.get("epochs", 1)),
        "--log_steps",
        str(_as_int(train_cfg.get("log_steps"), 10)),
        "--lr_scheduler_type",
        _as_str(train_cfg.get("lr_scheduler_type", "linear")),
        "--warmup_ratio",
        str(train_cfg.get("warmup_ratio", 0.03)),
        "--max_grad_norm",
        str(train_cfg.get("max_grad_norm", 1.0)),
        "--gradient_checkpointing",
        str(_as_int(train_cfg.get("gradient_checkpointing"), 1)),
        "--num_workers",
        str(_as_int(train_cfg.get("num_workers"), 4)),
        "--pin_memory",
        str(_as_int(train_cfg.get("pin_memory"), 1)),
        "--persistent_workers",
        str(_as_int(train_cfg.get("persistent_workers"), 1)),
        "--prefetch_factor",
        str(_as_int(train_cfg.get("prefetch_factor"), 2)),
        "--sequence_packing",
        str(_as_int(train_cfg.get("sequence_packing"), 1)),
        "--eval_sequence_packing",
        str(_as_int(train_cfg.get("eval_sequence_packing"), 0)),
        "--packing_max_seq_len",
        str(_as_int(train_cfg.get("packing_max_seq_len"), 3072)),
        "--attn_implementation",
        _as_str(train_cfg.get("attn_implementation", "eager")),
        "--train_scope",
        _as_str(train_cfg.get("train_scope", "full")),
        "--llm_lora",
        str(_as_int(train_cfg.get("llm_lora"), 0)),
        "--lora_r",
        str(_as_int(train_cfg.get("lora_r"), 8)),
        "--lora_alpha",
        str(_as_int(train_cfg.get("lora_alpha"), 16)),
        "--lora_dropout",
        str(train_cfg.get("lora_dropout", 0.05)),
        "--lora_bias",
        _as_str(train_cfg.get("lora_bias", "none")),
        "--ctc_enabled",
        str(_as_int(train_cfg.get("ctc_enabled"), 0)),
        "--ctc_weight",
        str(train_cfg.get("ctc_weight", 0.0)),
        "--ctc_tokenizer_type",
        _as_str(train_cfg.get("ctc_tokenizer_type", "utf8_bytes")),
        "--ddp_backend",
        _as_str(train_cfg.get("ddp_backend", "gloo")),
        "--save_strategy",
        _as_str(train_cfg.get("save_strategy", "steps")),
        "--save_steps",
        str(_as_int(train_cfg.get("save_steps"), 2000)),
        "--save_total_limit",
        str(_as_int(train_cfg.get("save_total_limit"), 3)),
        "--eval_strategy",
        _as_str(train_cfg.get("eval_strategy", "steps")),
        "--eval_steps",
        str(_as_int(train_cfg.get("eval_steps"), 2000)),
        "--early_stopping_patience",
        str(_as_int(train_cfg.get("early_stopping_patience"), 3)),
        "--early_stopping_threshold",
        str(train_cfg.get("early_stopping_threshold", 0.0)),
        "--resume",
        str(_as_int(train_cfg.get("resume"), 0)),
        "--resume_from",
        _as_str(train_cfg.get("resume_from", "")),
        "--best_checkpoint_file",
        str(run_dir / "best_checkpoint.json"),
    ]

    tokenizer_path = _as_str(config.get("tokenizer_path", ""))
    if tokenizer_path:
        cmd.extend(["--tokenizer_path", tokenizer_path])
    tokenizer_language = _as_str(config.get("tokenizer_language", ""))
    if tokenizer_language:
        cmd.extend(["--tokenizer_language", tokenizer_language])
    label_csv_path = _as_str(config.get("label_csv_path", ""))
    if label_csv_path:
        cmd.extend(["--label_csv_path", label_csv_path])
        cmd.extend(["--train_csv", _as_str(config.get("train_csv", "Train.csv"))])
        cmd.extend(["--dev_csv", _as_str(config.get("dev_csv", "Dev.csv"))])
    if rendered_augmentation_config:
        cmd.extend(["--augmentation_config", rendered_augmentation_config])
    if rendered_hard_lexicon_config:
        cmd.extend(["--hard_lexicon_config", rendered_hard_lexicon_config])
    return cmd


def build_eval_command(qwen_root: Path, config: dict[str, Any], run_dir: Path, checkpoint_path: Path) -> list[str]:
    eval_cfg = config.get("evaluation", {})
    eval_dir = run_dir / "evaluation"
    return [
        sys.executable,
        str(qwen_root / "tools" / "grpo_run_eval_jsonl_wer.py"),
        "--model_path",
        str(checkpoint_path),
        "--eval_file",
        _as_str(config.get("eval_file")),
        "--output_jsonl",
        str(eval_dir / "dev_pred_by_wer.jsonl"),
        "--summary_json",
        str(eval_dir / "dev_metrics.json"),
        "--batch_size",
        str(_as_int(eval_cfg.get("batch_size"), 4)),
        "--max_new_tokens",
        str(_as_int(eval_cfg.get("max_new_tokens"), 192)),
        "--device_map",
        _as_str(eval_cfg.get("device_map", "cuda:0")),
        "--attn_implementation",
        _as_str(eval_cfg.get("attn_implementation", "sdpa")),
        "--torch_dtype",
        _as_str(eval_cfg.get("torch_dtype", "bf16")),
    ]


def build_selection_command(qwen_root: Path, run_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(qwen_root / "tools" / "select_best_checkpoint_by_dev_metrics.py"),
        "--candidate",
        str(run_dir / "evaluation"),
        "--output_json",
        str(run_dir / "best_checkpoint_by_dev.json"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3-ASR adapter for ClosedLoopExperimentAgent")
    parser.add_argument("--config", required=True, help="Study config YAML.")
    parser.add_argument("--output-dir", required=True, help="Trial output directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    config_path = Path(args.config)
    run_dir = Path(args.output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise SystemExit("Config YAML must be a mapping")

    for _ in range(8):
        base_config_path = _as_str(config.get("config_base"), "")
        if not base_config_path:
            break
        resolved_base = Path(base_config_path)
        if not resolved_base.is_absolute():
            resolved_base = (config_path.parent / resolved_base).resolve()
        if not resolved_base.exists():
            raise SystemExit(f"Base config does not exist: {resolved_base}")
        base_config = _read_mapping(resolved_base)
        config_overrides = config.get("config_overrides")
        override_config = {
            key: value
            for key, value in config.items()
            if key not in {"config_base", "config_overrides"}
        }
        config = _deep_merge(base_config, override_config)
        if isinstance(config_overrides, dict) and config_overrides:
            config = _deep_merge(config, config_overrides)

    qwen_root = Path(_as_str(config.get("qwen3_asr_root"), os.environ.get("QWEN3_ASR_ROOT", ""))).expanduser()
    if not qwen_root:
        raise SystemExit("Missing qwen3_asr_root in config or QWEN3_ASR_ROOT env")
    if not qwen_root.exists():
        raise SystemExit(f"Qwen3-ASR root does not exist: {qwen_root}")

    run_dir.mkdir(parents=True, exist_ok=True)
    rendered_config_path = run_dir / "rendered_config.yaml"
    rendered_config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    env = os.environ.copy()
    runtime = config.get("runtime", {})
    cuda_visible_devices = env.get("CUDA_VISIBLE_DEVICES") or _as_str(runtime.get("cuda_visible_devices"), "0")
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    train_cmd = build_train_command(qwen_root, config, run_dir)
    eval_cmd = None
    selection_cmd = None

    if args.dry_run:
        print(json.dumps({
            "config": str(config_path),
            "run_dir": str(run_dir),
            "qwen3_asr_root": str(qwen_root),
            "train_command": train_cmd,
            "evaluation_command": build_eval_command(qwen_root, config, run_dir, run_dir / "checkpoint-0"),
            "selection_command": build_selection_command(qwen_root, run_dir),
        }, indent=2, ensure_ascii=False))
        return 0

    started = time.time()
    train_stdout = run_dir / "train.stdout.txt"
    train_stderr = run_dir / "train.stderr.txt"
    train_returncode = _run(train_cmd, qwen_root, env, train_stdout, train_stderr)

    status = "success" if train_returncode == 0 else "failed"
    selected_checkpoint = _load_selected_checkpoint(run_dir)
    metrics: dict[str, Any] = {
        "train_returncode": train_returncode,
    }
    artifacts: dict[str, str] = {
        "config": str(rendered_config_path),
        "train_stdout": str(train_stdout),
        "train_stderr": str(train_stderr),
    }

    eval_returncode = None
    if train_returncode == 0 and selected_checkpoint is not None:
        eval_cmd = build_eval_command(qwen_root, config, run_dir, selected_checkpoint)
        eval_returncode = _run(
            eval_cmd,
            qwen_root,
            env,
            run_dir / "evaluation" / "eval.stdout.txt",
            run_dir / "evaluation" / "eval.stderr.txt",
        )
        metrics["eval_returncode"] = eval_returncode
        artifacts["eval_pred_jsonl"] = str(run_dir / "evaluation" / "dev_pred_by_wer.jsonl")
        artifacts["eval_metrics_json"] = str(run_dir / "evaluation" / "dev_metrics.json")

        if eval_returncode == 0:
            selection_cmd = build_selection_command(qwen_root, run_dir)
            selection_returncode = _run(
                selection_cmd,
                qwen_root,
                env,
                run_dir / "evaluation" / "selection.stdout.txt",
                run_dir / "evaluation" / "selection.stderr.txt",
            )
            metrics["selection_returncode"] = selection_returncode
            selection_path = run_dir / "best_checkpoint_by_dev.json"
            if selection_returncode == 0 and selection_path.exists():
                selection = _read_json(selection_path)
                selected = selection.get("selected", {})
                metrics.update(
                    {
                        "corpus_wer": selected.get("corpus_wer"),
                        "corpus_cer": selected.get("corpus_cer"),
                        "exact_match_rate": selected.get("exact_match_rate"),
                        "severe_rate": selected.get("severe_rate"),
                        "runaway_rate": selected.get("runaway_rate"),
                        "checkpoint_path": selected.get("checkpoint_path"),
                        "checkpoint_step": selected.get("checkpoint_step"),
                        "num_samples": selected.get("num_samples"),
                    }
                )
                artifacts["selection_json"] = str(selection_path)
                artifacts["best_checkpoint"] = str(selected_checkpoint)
                primary_metric = selected.get("corpus_wer")
            else:
                primary_metric = None
        else:
            primary_metric = None
    else:
        primary_metric = None

    result = {
        "status": status,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "artifacts": artifacts,
        "run_dir": str(run_dir),
        "duration_sec": round(time.time() - started, 3),
        "stderr_summary": None,
        "notes": _as_str(config.get("notes"), ""),
    }
    _write_json(run_dir / "results.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if status == "success" and primary_metric is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
