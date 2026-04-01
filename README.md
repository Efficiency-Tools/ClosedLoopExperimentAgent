# Closed-Loop Experiment Agent / 闭环实验代理

Open-source MVP for controlled experiment loops under a fixed baseline, fixed direction, fixed search space, and fixed budget.

This project is intentionally narrow:

- It proposes trials from a declared search space.
- It validates proposals against guardrails.
- It launches a local training command.
- It parses `results.json`.
- It logs parent/child runs to MLflow.
- It updates an Optuna study using ask-and-tell.
- It stops when budget or guardrails say to stop.

It does not rewrite code, invent new research directions, or act as a general coding agent.

这是一个面向生产的最小可用版本，用于在固定 baseline、固定方向、固定搜索空间和固定预算下运行实验闭环。

这个项目的边界是明确的：

- 从声明好的搜索空间中提出 trial。
- 按 guardrail 校验 proposal。
- 启动本地训练命令。
- 解析 `results.json`。
- 将 parent/child run 记录到 MLflow。
- 使用 Optuna ask-and-tell 更新搜索状态。
- 在预算或 guardrail 告知时停止。

它不会改写任意代码，也不会发明新的研究方向，更不是通用编码代理。

## Architecture / 架构

- `app/graph`: LangGraph controller state, nodes, and edges. / LangGraph 控制器状态、节点和边。
- `app/optimizer`: Optuna ask-and-tell wrapper and search-space parsing. / Optuna ask-and-tell 封装与搜索空间解析。
- `app/runner`: Local experiment execution. / 本地实验执行。
- `app/tracking`: MLflow logging and summary schemas. / MLflow 记录与摘要 schema。
- `app/analysis`: Metric parsing, failure classification, and heuristic summaries. / 指标解析、失败分类和启发式分析。
- `app/guards`: Whitelist validation and stop rules. / 白名单校验与停止规则。
- `examples/toy_pytorch`: A dependency-free toy training target. / 一个无外部深度学习依赖的 toy 训练脚本。

## Quickstart / 快速开始

1. Install dependencies. / 安装依赖。
2. Run the toy study. / 运行 toy study。

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

## Search Space Contract / 搜索空间约定

The search-space YAML declares:

- `objective.name`
- `objective.direction`
- `space.*` parameters
- `constraints.*` limits

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

## License / 许可证

This repository is licensed under the GNU General Public License v3.0 or later (`GPL-3.0-or-later`).

本仓库采用 GNU General Public License v3.0 或更高版本授权（`GPL-3.0-or-later`）。

## TODOs / 后续计划

- Add LLM-assisted analysis summaries. / 增加 LLM 辅助的分析摘要。
- Add Slurm and Kubernetes runners. / 增加 Slurm 和 Kubernetes runner。
- Add multi-stage and multi-objective search policies. / 增加多阶段与多目标搜索策略。
- Add richer repair rules for partially invalid proposals. / 增加更丰富的 proposal 修复规则。
