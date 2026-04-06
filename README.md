# Closed-Loop Experiment Agent / 闭环实验代理

Open-source MVP for controlled experiment loops under a fixed baseline, fixed direction, fixed search space, and fixed budget.

这是一个面向生产的最小可用版本，用于在固定 baseline、固定方向、固定搜索空间和固定预算下运行实验闭环。

这个项目的边界是明确的：

- 从声明好的搜索空间中提出 trial。
- 按 guardrail 校验 proposal。
- 启动本地训练命令。
- 解析 `results.json`。
- 将 parent/child run 记录到 MLflow。
- 使用 Optuna ask-and-tell 更新搜索状态。
- 在预算或 guardrail 告知时停止。

It does not rewrite code, invent new research directions, or act as a general coding agent.
它不会改写任意代码，也不会发明新的研究方向，更不是通用编码代理。

## Architecture / 架构

```text
   +------------------+
   |   Launch study   |
   +------------------+
            |
            v
   +------------------+
   | Init study state |
   +------------------+
            |
            v
   +------------------------+
   | Ask Optuna for a trial |
   +------------------------+
            |
            v
   +----------------------+
   | Validate proposal    |
   +----------------------+
        |             |
        | invalid     | valid
        v             v
 +----------------+   +------------------+
 | Analyze invalid|   | Start training   |
 | proposal       |   +------------------+
 +----------------+            |
                                v
                    +---------------------------+
                    | Watchdog monitors logs    |
                    +---------------------------+
                       |                   |
                       | loss is fine     | error / bad loss
                       v                   v
              +----------------+   +------------------+
              | Collect metrics |   | Analyze failure  |
              +----------------+   +------------------+
                       |                   |
                       v                   v
              +--------------------+  +----------------------+
              | Deterministic      |  | Repair proposal /    |
              | analysis           |  | config               |
              +--------------------+  +----------------------+
                       |                   |
                       v                   |
              +--------------------+       |
              | Update Optuna +    |       |
              | tracking           |       |
              +--------------------+       |
                       |                   |
                       v                   |
              +---------------------------+
              | Stop condition met?       |
              +---------------------------+
                       |           |
                       | no        | yes
                       v           v
                 back to trial   +-------------------+
                                 | Write final       |
                                 | summary           |
                                 +-------------------+
```

Core responsibilities:

- `app/graph`: LangGraph controller state, nodes, and edges. / LangGraph 控制器状态、节点和边。
- `app/optimizer`: Optuna ask-and-tell wrapper and search-space parsing. / Optuna ask-and-tell 封装与搜索空间解析。
- `app/runner`: Local experiment execution. / 本地实验执行。
- `app/tracking`: MLflow logging and summary schemas. / MLflow 记录与摘要 schema。
- `app/analysis`: Metric parsing, failure classification, and heuristic summaries. / 指标解析、失败分类和启发式分析。
- `app/guards`: Whitelist validation and stop rules. / 白名单校验与停止规则。
- `app/evaluation`: Benchmark suites and version-to-version comparisons. / 基准套件与版本对比。
- `examples/toy_pytorch`: A dependency-free toy training target. / 一个无外部深度学习依赖的 toy 训练脚本。

## Quickstart / 快速开始

1. Create a Python 3.11+ environment.
2. Install the project in editable mode.
3. Run the toy study.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
clae --help
```

Run the toy study:

```bash
python -m app.main launch-study \
  --baseline-config examples/toy_pytorch/baseline.yaml \
  --search-space examples/toy_pytorch/search_space.yaml \
  --study-name toy-study \
  --study-dir runs/toy-study \
  --storage-path runs/toy-study/optuna.sqlite3 \
  --tracking-uri file:./runs/mlruns \
  --mlflow-experiment-name closed-loop-experiment-agent \
  --train-command "python examples/toy_pytorch/train.py --config {config_path} --output-dir {run_dir}"
```

### Qwen3-ASR / Qwen3-ASR 接入

For the Qwen3-ASR workspace, use the adapter example in
[examples/qwen3_asr/README.md](examples/qwen3_asr/README.md). It wraps the
Qwen3-ASR training script, evaluates the selected checkpoint on dev, and
writes a `results.json` file that CLAE can consume.

Qwen3-ASR 示例见 [examples/qwen3_asr/README.md](examples/qwen3_asr/README.md)。

Or use the installed CLI directly:

```bash
clae launch-study \
  --baseline-config examples/toy_pytorch/baseline.yaml \
  --search-space examples/toy_pytorch/search_space.yaml \
  --study-name toy-study \
  --study-dir runs/toy-study \
  --storage-path runs/toy-study/optuna.sqlite3 \
  --tracking-uri file:./runs/mlruns \
  --mlflow-experiment-name closed-loop-experiment-agent \
  --train-command "python examples/toy_pytorch/train.py --config {config_path} --output-dir {run_dir}"
```

Run a benchmark suite and compare against a previous version:

```bash
clae evaluate-system \
  --suite-path examples/evaluation/toy_suite.yaml \
  --output-dir runs/evaluations/toy-suite \
  --version-tag v2.0.0 \
  --reference-report runs/evaluations/toy-suite/toy-suite_evaluation.json
```

The toy example writes:

- `config.yaml`
- `results.json`
- `analysis.json`
- stdout/stderr captures
- an MLflow parent run plus one child run per trial

toy example 会写出：

- `config.yaml`
- `results.json`
- `analysis.json`
- stdout/stderr 日志
- 一个 MLflow parent run 和多个 child run

## Installation / 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you prefer not to activate the environment, use `.venv/bin/python` or `.venv/bin/clae`.

## Usage / 使用方法

### 1. Prepare the baseline config

Start from [examples/toy_pytorch/baseline.yaml](examples/toy_pytorch/baseline.yaml) and replace the values with your own model, optimizer, and training defaults.

### 2. Define the search space

Edit [examples/toy_pytorch/search_space.yaml](examples/toy_pytorch/search_space.yaml) or create your own file with:

- `objective.name`
- `objective.direction`
- `space.*` tunable parameters
- `constraints.*` budgets and guardrails

Only parameters declared in `space` are mutable. Protected keys are never touched.

### 3. Make your training script obey the contract

Your training command must accept:

- `--config {config_path}`
- `--output-dir {run_dir}`

It must write `results.json` into the output directory. The expected payload includes:

- `status`: `success` or `failed`
- `primary_metric`
- `metrics`
- optional `notes`

The toy example in [examples/toy_pytorch/train.py](examples/toy_pytorch/train.py) shows the contract.

### 4. Launch the study

```bash
clae launch-study \
  --baseline-config path/to/baseline.yaml \
  --search-space path/to/search_space.yaml \
  --study-name my-study \
  --study-dir runs/my-study \
  --storage-path runs/my-study/optuna.sqlite3 \
  --tracking-uri file:./runs/mlruns \
  --mlflow-experiment-name my-experiment \
  --train-command "python train.py --config {config_path} --output-dir {run_dir}"
```

### 5. Inspect outputs

Each study writes:

- `state.json`
- `final_summary.json`
- `artifacts/`
- `optuna.sqlite3`
- MLflow runs under the tracking URI you chose

## Evaluation / 评测

Use `evaluate-system` to run a benchmark suite and optionally compare against a saved report:

```bash
clae evaluate-system \
  --suite-path examples/evaluation/toy_suite.yaml \
  --output-dir runs/evaluations/toy-suite \
  --version-tag v2.0.0 \
  --reference-report runs/evaluations/toy-suite/toy-suite_evaluation.json
```

The suite file is a YAML list of cases, each case pointing at a baseline config, a search space, and a training command.

## Search Space Contract / 搜索空间约定

The search-space YAML declares:

- `objective.name`
- `objective.direction`
- `space.*` parameters
- `constraints.*` limits
- Optional loss supervision hints such as `constraints.loss_metric_keys` and `constraints.max_loss_value`

Only parameters declared in `space` are mutable. Protected keys are never touched.

搜索空间 YAML 会声明：

- `objective.name`
- `objective.direction`
- `space.*` 参数
- `constraints.*` 限制

只有 `space` 中声明的参数允许修改。`protected_keys` 永远不会被改动。

## Restartability / 可恢复性

The Optuna study uses SQLite storage. Trial artifacts are written to per-trial directories under the study root, and state is persisted to JSON so the loop can resume between trials.

Optuna study 使用 SQLite 存储。每个 trial 的产物会写入 study 目录下的独立子目录，状态会持久化为 JSON，方便中断后恢复。

## Repository Layout / 仓库结构

- `app/graph`: LangGraph controller state, nodes, and edges.
- `app/optimizer`: Optuna ask-and-tell wrapper and search-space parsing.
- `app/runner`: Local experiment execution.
- `app/tracking`: MLflow logging and summary schemas.
- `app/analysis`: Metric parsing, failure classification, and heuristic summaries.
- `app/guards`: Whitelist validation and stop rules.
- `app/evaluation`: Benchmark suites and version-to-version comparisons.
- `examples/toy_pytorch`: A dependency-free toy training target.

## Notes for contributors / 提交前注意事项

- Do not commit `runs/`, `mlruns/`, SQLite databases, or cache directories.
- Keep training scripts compatible with the `results.json` contract.
- Use `clae --help` to inspect the available commands.

## License / 许可证

This repository is licensed under the GNU General Public License v3.0 or later (`GPL-3.0-or-later`).

本仓库采用 GNU General Public License v3.0 或更高版本授权（`GPL-3.0-or-later`）。

## TODOs / 后续计划

- Add LLM-assisted analysis summaries. / 增加 LLM 辅助的分析摘要。
- Add Slurm and Kubernetes runners. / 增加 Slurm 和 Kubernetes runner。
- Add multi-stage and multi-objective search policies. / 增加多阶段与多目标搜索策略。
- Add richer repair rules for partially invalid proposals. / 增加更丰富的 proposal 修复规则。
