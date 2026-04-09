# RA-NAS: Reducing NAS Cost using LLM-Based Reasoning Agents

RA-NAS combines Neural Architecture Search (NAS) with an LLM reasoning agent that proposes and refines CNN architectures based on prior experiment outcomes. The goal is to reduce total search cost by spending less compute on weak candidates and improving candidate quality across iterations.

Compared to pure random search or RL-based controllers, RA-NAS improves efficiency by:

- Reusing experiment memory to avoid repeating poor architectural patterns.
- Injecting explicit feedback (accuracy, loss, efficiency) into iterative architecture refinement.
- Supporting low-cost offline operation with `mock_mode: true` for reproducible development and CI.

## Pipeline Diagram

```text
┌─────────────┐
│ LLM Agent   │
│ (propose)   │
└──────┬──────┘
       │ architecture JSON
       v
┌─────────────┐
│ build_model │
└──────┬──────┘
       │ PyTorch model
       v
┌─────────────┐
│ Trainer     │
│ (train/val) │
└──────┬──────┘
       │ checkpoint + train metrics
       v
┌─────────────┐
│ Evaluator   │
│ (val stats) │
└──────┬──────┘
       │ feedback metrics
       v
┌─────────────┐
│ Memory      │
│ (store top) │
└──────┬──────┘
       │ top-k history
       v
┌─────────────┐
│ LLM Agent   │
│ (refine)    │
└─────────────┘
```

## Setup Instructions

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set API key only if `mock_mode` is disabled:

```bash
export OPENAI_API_KEY="your_key_here"
```

By default, `configs/agent.yaml` sets `mock_mode: true`, so no key is required.

## How to Run

Main experiment command:

```bash
python scripts/run_experiment.py \
  --train-config configs/train.yaml \
  --agent-config configs/agent.yaml \
  --iterations 5 \
  --device cpu
```

Arguments:

- `--train-config`: Path to training/NAS config YAML.
- `--agent-config`: Path to LLM/agent config YAML.
- `--iterations`: Number of NAS iterations (overrides config when provided).
- `--device`: `cpu` or `cuda` (auto-selected if omitted).

Checkpoint evaluation:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint experiments/exp_001/model.pt \
  --train-config configs/train.yaml \
  --device cpu
```

## Config Reference

### `configs/train.yaml`

| Key | Type | Default | Description |
|---|---|---|---|
| `dataset.name` | `str` | `cifar10` | Dataset identifier used by loader. |
| `dataset.data_dir` | `str` | `./data` | Root directory for dataset storage. |
| `dataset.num_classes` | `int` | `10` | Number of output classes. |
| `dataset.val_split` | `float` | `0.1` | Fraction of train set used as validation. |
| `training.epochs` | `int` | `20` | Max epochs per architecture. |
| `training.batch_size` | `int` | `128` | Batch size for train/validation loaders. |
| `training.learning_rate` | `float` | `0.001` | Initial learning rate for AdamW. |
| `training.weight_decay` | `float` | `1e-4` | Weight decay regularization. |
| `training.scheduler` | `str` | `cosine` | LR scheduler (`cosine`, `step`, `none`). |
| `training.num_workers` | `int` | `0` | DataLoader worker processes (set 0 for strict sandbox compatibility). |
| `training.seed` | `int` | `42` | Global seed for reproducibility. |
| `early_stopping.enabled` | `bool` | `true` | Enables patience-based early stopping. |
| `early_stopping.patience` | `int` | `5` | Allowed unimproved epochs. |
| `early_stopping.monitor` | `str` | `val_accuracy` | Metric used for stopping/checkpointing. |
| `early_stopping.mode` | `str` | `max` | Monitor direction (`max` or `min`). |
| `experiment.name` | `str` | `exp_001` | Base experiment name. |
| `experiment.output_dir` | `str` | `./experiments` | Root output directory for runs. |
| `experiment.save_best_only` | `bool` | `true` | Save checkpoints only when improved. |
| `architecture_constraints.min_layers` | `int` | `2` | Minimum CNN depth. |
| `architecture_constraints.max_layers` | `int` | `6` | Maximum CNN depth. |
| `architecture_constraints.min_filters` | `int` | `16` | Minimum filter count per layer. |
| `architecture_constraints.max_filters` | `int` | `256` | Maximum filter count per layer. |
| `architecture_constraints.allowed_activations` | `list[str]` | `[relu, gelu, silu]` | Allowed activation functions. |
| `architecture_constraints.allowed_kernels` | `list[int]` | `[3, 5]` | Allowed kernel sizes. |

### `configs/agent.yaml`

| Key | Type | Default | Description |
|---|---|---|---|
| `llm.provider` | `str` | `openai` | LLM backend provider. |
| `llm.model` | `str` | `gpt-4o` | Model identifier for API calls. |
| `llm.temperature` | `float` | `0.7` | Sampling temperature for generation. |
| `llm.max_tokens` | `int` | `1024` | Maximum completion tokens. |
| `llm.api_key_env` | `str` | `OPENAI_API_KEY` | Environment variable holding API key. |
| `agent.max_iterations` | `int` | `10` | Default NAS iterations. |
| `agent.top_k_memory` | `int` | `5` | Memory entries passed into prompt context. |
| `agent.retry_on_invalid` | `int` | `3` | Retries for invalid LLM architecture output. |
| `agent.feedback_strategy` | `str` | `top_k` | Prompt feedback policy (`top_k`, `threshold`, `all`). |
| `agent.mock_mode` | `bool` | `true` | If true, bypasses API calls and samples valid arches. |

## Experiment Output Structure

Each run creates a timestamped directory:

```text
experiments/<exp_name>_<timestamp>/
├── config.yaml       # merged config snapshot for reproducibility
├── metrics.json      # per-iteration architecture and metric records
├── memory.json       # full memory buffer used by the agent
├── experiment.log    # structured logs
└── iter_XXX/
    └── model.pt      # best checkpoint for that iteration
```

## Extending the Project

1. Add a new search space:
   - Extend `src/nas/search_space.py` schema and validator.
   - Update `src/nas/architecture_generator.py` sampling/mutation logic.
2. Swap LLM provider:
   - Extend `src/agents/llm_agent.py::_init_openai_client` and `_call_llm`.
   - Keep prompt and parse interfaces unchanged for compatibility.
3. Add a second agent:
   - Implement a new agent class with `propose_architecture` and `refine_architecture`.
   - Inject it into `src/nas/controller.py` without touching trainer/evaluator code.

## Ablation Study Guide

Recommended ablations and measurements:

1. Agent vs Random:
   - Set `agent.mock_mode: true` and replace agent proposals with `ArchitectureGenerator.sample_random`.
   - Measure best val accuracy under fixed iteration budget.
2. Memory Depth:
   - Vary `agent.top_k_memory` in `{1, 3, 5, 10}`.
   - Measure convergence speed and architecture diversity.
3. LLM Stochasticity:
   - Vary `llm.temperature` in `{0.2, 0.7, 1.0}`.
   - Measure invalid-response rate and final best score.
4. Training Budget:
   - Vary `training.epochs` and `early_stopping.patience`.
   - Measure compute cost vs. resulting best architecture quality.
5. Search Constraints:
   - Tighten/relax `architecture_constraints.max_layers`, filter bounds, and activation set.
   - Measure tradeoff between speed, parameter count, and accuracy.
