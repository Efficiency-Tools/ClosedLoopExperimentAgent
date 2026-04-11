# Qwen3-ASR Closed Loop Example

This example adapts `ClosedLoopExperimentAgent` to Qwen3-ASR.

## What it does

- launches a Qwen3-ASR SFT trial from a YAML config
- evaluates the selected checkpoint on the dev set
- writes `results.json` in the trial directory
- uses dev `corpus_wer` as the primary metric

## Install

Install the agent into the active Qwen3-ASR environment:

```bash
/mnt/shareEEEx/liuxiaokang/.conda/envs/qwen3-asr/bin/python -m pip install -e /mnt/shareEEEx/liuxiaokang/workspace/ClosedLoopExperimentAgent
```

## Dry run

```bash
clae launch-study \
  --baseline-config examples/qwen3_asr/baseline.yaml \
  --search-space examples/qwen3_asr/search_space.yaml \
  --study-name qwen3-asr-uyghur-sft \
  --study-dir runs/qwen3-asr-uyghur-sft \
  --storage-path runs/qwen3-asr-uyghur-sft/optuna.sqlite3 \
  --tracking-uri file:./runs/mlruns \
  --mlflow-experiment-name qwen3-asr-closed-loop \
  --train-command "python examples/qwen3_asr/train.py --config {config_path} --output-dir {run_dir}" \
  --timeout-sec 7200
```

The trial wrapper accepts `--dry-run` too:

```bash
/mnt/shareEEEx/liuxiaokang/.conda/envs/qwen3-asr/bin/python examples/qwen3_asr/train.py \
  --config examples/qwen3_asr/baseline.yaml \
  --output-dir runs/qwen3-asr-dry-run \
  --dry-run
```

