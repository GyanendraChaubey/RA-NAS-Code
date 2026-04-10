# RA-NAS: Reducing NAS Cost using LLM-Based Reasoning Agents

RA-NAS combines Neural Architecture Search (NAS) with an LLM reasoning agent that proposes and refines CNN architectures based on prior experiment outcomes. The goal is to reduce total search cost by spending less compute on weak candidates and improving candidate quality across iterations.

Compared to pure random search or RL-based controllers, RA-NAS improves efficiency by:

- Reusing experiment memory to avoid repeating poor architectural patterns.
- Injecting explicit feedback (accuracy, loss, efficiency) into iterative architecture refinement.
- Supporting low-cost offline operation with `mock_mode: true` for reproducible development and CI.

## Pipeline Diagram

```text
┌─────────────┐
│ LLM Agent   │  structured reasoning: observations → hypothesis → changes
│ (propose)   │  diversity penalty discourages repeated arch patterns
└──────┬──────┘
       │ architecture JSON (ResNet bottleneck; block_depths, filters, SE blocks)
       v
┌──────────────────┐
│ Screen Trainer   │  Phase 4: train for screening_epochs (fast 10-epoch proxy)
│ (multi-fidelity) │  odd iters: screen only; even iters: compare pair, promote winner
└──────┬───────────┘
       │ winner arch (if even iteration)
       v
┌─────────────┐
│ Full Trainer │  train for epochs (200) with RandAugment, Mixup, CutOut,
│ (train/val) │  cosine-warmup LR, label smoothing, SWA (starts at 75%)
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
│ Memory      │  stores top-k (accuracy, params, FLOPs, arch)
│ (store top) │  self-correction: flags poor predictions, updates confidence
└──────┬──────┘
       │ top-k history + prediction errors
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
export GROQ_API_KEY="your_key_here"
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
| `training.epochs` | `int` | `50` | Max epochs per architecture (use 200 for best results). |
| `training.batch_size` | `int` | `256` | Batch size for train/validation loaders. |
| `training.learning_rate` | `float` | `0.1` | Initial learning rate for SGD. |
| `training.momentum` | `float` | `0.9` | SGD momentum. |
| `training.weight_decay` | `float` | `1e-3` | Weight decay regularization. |
| `training.optimizer` | `str` | `sgd` | Optimizer (`sgd` recommended for ResNet). |
| `training.scheduler` | `str` | `cosine_warmup` | LR scheduler (`cosine_warmup`, `cosine`, `step`, `none`). |
| `training.warmup_epochs` | `int` | `10` | Linear LR warmup epochs. |
| `training.screening_epochs` | `int` | `10` | Epochs used in Phase 4 multi-fidelity screening pass. |
| `training.label_smoothing` | `float` | `0.1` | Label smoothing factor for cross-entropy loss. |
| `training.num_workers` | `int` | `2` | DataLoader worker processes. |
| `training.seed` | `int` | `42` | Global seed for reproducibility. |
| `training.augmentation.cutout` | `bool` | `true` | Enable CutOut regularization. |
| `training.augmentation.cutout_length` | `int` | `8` | CutOut patch size in pixels. |
| `training.augmentation.mixup` | `bool` | `true` | Enable Mixup data augmentation. |
| `training.augmentation.mixup_alpha` | `float` | `0.4` | Mixup Beta distribution alpha parameter. |
| `training.augmentation.randaugment` | `bool` | `true` | Enable RandAugment (num_ops=2, magnitude=9). |
| `training.swa.enabled` | `bool` | `true` | Enable Stochastic Weight Averaging. |
| `training.swa.start_frac` | `float` | `0.75` | Fraction of training elapsed before SWA starts. |
| `training.swa.lr` | `float` | `0.05` | Constant SWA learning rate. |
| `early_stopping.enabled` | `bool` | `true` | Enables patience-based early stopping. |
| `early_stopping.patience` | `int` | `30` | Allowed unimproved epochs. |
| `early_stopping.monitor` | `str` | `val_accuracy` | Metric used for stopping/checkpointing. |
| `early_stopping.mode` | `str` | `max` | Monitor direction (`max` or `min`). |
| `experiment.name` | `str` | `exp_001` | Base experiment name. |
| `experiment.output_dir` | `str` | `./experiments` | Root output directory for runs. |
| `experiment.save_best_only` | `bool` | `true` | Save checkpoints only when improved. |
| `architecture_constraints.min_layers` | `int` | `2` | Minimum number of ResNet stages. |
| `architecture_constraints.max_layers` | `int` | `8` | Maximum number of ResNet stages. |
| `architecture_constraints.min_filters` | `int` | `64` | Minimum filter count (first stage). |
| `architecture_constraints.max_filters` | `int` | `512` | Maximum filter count (any stage). |
| `architecture_constraints.allowed_activations` | `list[str]` | `[relu, gelu, silu]` | Allowed activation functions. |
| `architecture_constraints.allowed_kernels` | `list[int]` | `[3, 5]` | Allowed kernel sizes for 3×3 conv in bottleneck. |

### `configs/agent.yaml`

| Key | Type | Default | Description |
|---|---|---|---|
| `llm.provider` | `str` | `groq` | LLM backend provider. |
| `llm.model` | `str` | `llama-3.3-70b-versatile` | Model identifier for API calls. |
| `llm.base_url` | `str` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint URL. |
| `llm.temperature` | `float` | `1.2` | Initial sampling temperature (annealed per iteration). |
| `llm.temperature_min` | `float` | `0.7` | Minimum temperature after annealing. |
| `llm.temperature_decay` | `float` | `0.05` | Temperature reduction per iteration. |
| `llm.max_tokens` | `int` | `1536` | Maximum completion tokens. |
| `llm.api_key_env` | `str` | `GROQ_API_KEY` | Environment variable holding API key. |
| `agent.max_iterations` | `int` | `20` | Default NAS iterations. |
| `agent.top_k_memory` | `int` | `5` | Memory entries passed into prompt context. |
| `agent.retry_on_invalid` | `int` | `3` | Retries for invalid LLM architecture output. |
| `agent.feedback_strategy` | `str` | `top_k` | Prompt feedback policy (`top_k`, `threshold`, `all`). |
| `agent.mock_mode` | `bool` | `true` | If true, bypasses API calls and samples valid arches. |
| `agent.explore_every` | `int` | `2` | Force a fresh random proposal every N iterations (0 = disable). |
| `agent.diversity_penalty` | `bool` | `true` | Penalise architectures too similar to top-k memory entries. |

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
