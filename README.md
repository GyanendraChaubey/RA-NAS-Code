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
- `--override`: Zero or more additional YAML files merged on top, in order (later wins) — see
  [Ablation Study Guide](#ablation-study-guide) for ready-made files under `configs/ablations/`.

Checkpoint evaluation:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint experiments/exp_001/model.pt \
  --train-config configs/train.yaml \
  --device cpu
```

## NAS-Bench-201 Track

A secondary benchmark track (`scripts/run_nasbench201.py`) runs the same `LLMAgent` — same
reasoning, memory, diversity penalty, self-correction, and cost accounting — against the
[NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201) topology cell space (4 nodes, 6 edges,
5 ops each) instead of the custom ResNet-bottleneck space. Candidate cells are scored with an
instant lookup against [NATS-Bench](https://github.com/D-X-Y/NATS-Bench)'s precomputed CIFAR-10
results instead of real training, so this track:

- Needs no GPU — it's a table lookup, runs entirely on CPU.
- Gives a number directly comparable to other LLM-NAS papers that report on this benchmark
  (e.g. LLMatic, RZ-NAS), instead of the apples-to-oranges comparison a custom search space gives.
- Provides genuine multi-fidelity data (NATS-Bench reports results at 12 and 200 training epochs
  for every candidate), so the multi-fidelity screening ablation is exact, not an approximation.

**Setup:**

```bash
pip install nats_bench
```

Download the topology-search-space (`tss`) benchmark file per the
[NATS-Bench download instructions](https://github.com/D-X-Y/NATS-Bench#preparation-and-download)
(the `NATS-tss-v1_0-3ffb9-simple.tar` archive is the practical choice — uncompress it and either
place it under `$TORCH_HOME` or pass its path explicitly).

**Run:**

```bash
python scripts/run_nasbench201.py \
  --agent-config configs/agent.yaml \
  --nasbench-config configs/nasbench201.yaml \
  --iterations 20 \
  --api-file /path/to/NATS-tss-v1_0-3ffb9-simple
```

Omit `--api-file` if the benchmark archive is under `$TORCH_HOME` (or set
`nasbench201.api_file` in the config instead). Results are written to
`experiments/<name>_<timestamp>/{metrics.json, memory.json, config.yaml, experiment.log}`,
same as the main track.

### `configs/nasbench201.yaml`

| Key | Type | Default | Description |
|---|---|---|---|
| `nasbench201.api_file` | `str \| null` | `null` | Path to the downloaded NATS-tss archive; `null` looks under `$TORCH_HOME`. |
| `nasbench201.dataset` | `str` | `cifar10-valid` | `cifar10-valid`, `cifar10`, `cifar100`, or `ImageNet16-120`. |
| `nasbench201.fast_mode` | `bool` | `true` | Loads per-architecture records on demand instead of one large pickle. |
| `nasbench201.screen_hp` | `str` | `"12"` | Epoch budget for multi-fidelity screening queries. |
| `nasbench201.full_hp` | `str` | `"200"` | Epoch budget for full-fidelity queries. |
| `nasbench201.screening_enabled` | `bool` | `true` | Pair iterations and only full-query the screening winner (see Ablations below). |

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
| `llm.model` | `str` | `openai/gpt-oss-120b` | Model identifier for API calls. |
| `llm.base_url` | `str` | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint URL. |
| `llm.temperature` | `float` | `1.2` | Initial sampling temperature (annealed per iteration). |
| `llm.temperature_min` | `float` | `0.7` | Minimum temperature after annealing. |
| `llm.temperature_decay` | `float` | `0.05` | Temperature reduction per iteration. |
| `llm.max_tokens` | `int` | `1536` | Maximum completion tokens. |
| `llm.api_key_env` | `str` | `GROQ_API_KEY` | Environment variable holding API key. |
| `llm.pricing.input_per_million_usd` | `float` | `0.14` | Prompt-token price used for cost accounting. |
| `llm.pricing.output_per_million_usd` | `float` | `0.68` | Completion-token price used for cost accounting. |
| `agent.max_iterations` | `int` | `20` | Default NAS iterations. |
| `agent.top_k_memory` | `int` | `5` | Memory entries passed into prompt context. |
| `agent.retry_on_invalid` | `int` | `3` | Retries for invalid LLM architecture output. |
| `agent.feedback_strategy` | `str` | `top_k` | Prompt feedback policy (`top_k`, `threshold`, `all`). |
| `agent.mock_mode` | `bool` | `true` | If true, bypasses API calls and samples valid arches. |
| `agent.explore_every` | `int` | `2` | Force a fresh random proposal every N iterations (0 = disable). |
| `agent.diversity_penalty` | `bool` | `true` | Penalise architectures too similar to top-k memory entries. |
| `agent.self_correction` | `bool` | `true` | Ask the LLM to predict val_accuracy and feed back prediction error on refinement. |

LLM usage/cost accounting is tracked per `LLMAgent` instance (`get_cost_summary()`) and included under each iteration record's `cost` key — set `llm.pricing` to `0` to disable cost estimation, or to match a different provider's rates.

## Experiment Output Structure

Each run creates a timestamped directory, named from `experiment.name` in the merged config —
so ablation runs (which each set their own name via `configs/ablations/*.yaml`) land in
distinct, self-describing folders rather than overwriting each other:

```text
experiments/<exp_name>_<timestamp>/
├── config.yaml       # merged config snapshot for reproducibility (what scripts/aggregate_results.py reads)
├── metrics.json      # per-iteration architecture, metric, and cost records
├── memory.json       # full memory buffer used by the agent
├── experiment.log    # structured logs
└── iter_XXX/
    └── model.pt      # best checkpoint for that iteration (main track only; NAS-Bench-201 has no checkpoints)
```

Run `python scripts/aggregate_results.py` to fold every run under `experiments/` into one
`experiments/ablation_summary.csv` comparison table — see
[Aggregating results for a paper](#aggregating-results-for-a-paper).

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
4. Add a new benchmark backend:
   - Implement a generator with `sample_random`/`mutate`/`validate` and a prompt builder with
     `build_proposal_prompt`/`build_refinement_prompt` for the new architecture representation.
   - Pass the prompt builder into `LLMAgent(..., prompt_builder=...)` — no changes to `LLMAgent`
     itself are needed. See `src/nasbench201/` for a complete reference implementation.

## Ablation Study Guide

`configs/ablations/` holds one small YAML file per ablation — each overrides only the keys it
needs to change (and sets a distinct `experiment.name` so results land in a self-describing
folder). Pass one via `--override` on top of the base config; nothing else about the command
changes. Both scripts accept `--override` (see `--help`), and `merge_configs` applies overrides
last, so they win over the base file.

**Main CIFAR-10 track** (`scripts/run_experiment.py`):

```bash
# 1. Full method (baseline/control)
python scripts/run_experiment.py --override configs/ablations/full_method.yaml --iterations 20

# 2. Random-search baseline — every iteration samples a fresh random architecture,
#    zero LLM calls (explore_every=1 forces a fresh propose() every iteration; mock_mode
#    makes that call ArchitectureGenerator.sample_random() directly, no LLM involved)
python scripts/run_experiment.py --override configs/ablations/random_search.yaml --iterations 20

# 3. Diversity penalty off (Phase 3)
python scripts/run_experiment.py --override configs/ablations/no_diversity_penalty.yaml --iterations 20

# 4. Self-correction off (Phase 5) — LLM is never asked to predict val_accuracy,
#    no prediction-error feedback attached on refinement
python scripts/run_experiment.py --override configs/ablations/no_self_correction.yaml --iterations 20

# 5. Multi-fidelity screening off (Phase 4) — every architecture trained for the full
#    training.epochs budget directly, no screen-then-promote pairing
python scripts/run_experiment.py --override configs/ablations/no_multi_fidelity.yaml --iterations 20
```

**NAS-Bench-201 track** (`scripts/run_nasbench201.py`) — the same `agent.*` override files work
unchanged, since both scripts share `LLMAgent`; only the multi-fidelity ablation is track-specific:

```bash
python scripts/run_nasbench201.py --override configs/ablations/full_method.yaml --iterations 20
python scripts/run_nasbench201.py --override configs/ablations/random_search.yaml --iterations 20
python scripts/run_nasbench201.py --override configs/ablations/no_diversity_penalty.yaml --iterations 20
python scripts/run_nasbench201.py --override configs/ablations/no_self_correction.yaml --iterations 20
python scripts/run_nasbench201.py --override configs/ablations/nb201_no_multi_fidelity.yaml --iterations 20
```

Two ablations combine cleanly by passing multiple `--override` files, e.g. random search with
self-correction also off: `--override configs/ablations/random_search.yaml configs/ablations/no_self_correction.yaml`.

Further ablations, still config-only (no dedicated file — pass the key directly via a one-off
override, or edit a copy of `configs/train.yaml`/`configs/agent.yaml`):

- **Memory depth**: vary `agent.top_k_memory` in `{1, 3, 5, 10}` — convergence speed, architecture diversity.
- **LLM stochasticity**: vary `llm.temperature` in `{0.2, 0.7, 1.0}` — invalid-response retry rate in `experiment.log`, final best score.
- **Training budget** (main track): vary `training.epochs`/`early_stopping.patience` — compute cost vs. quality.
- **Search constraints**: tighten/relax `architecture_constraints.*` (main track) or set `nasbench201.dataset` to `cifar100`/`ImageNet16-120` (NAS-Bench-201 track) — speed/params/accuracy tradeoff, cross-dataset generalization.

### Aggregating results for a paper

Every iteration record in `metrics.json` already carries a `cost` field (`total_llm_calls`,
`total_prompt_tokens`, `total_completion_tokens`, `total_llm_latency_s`, `estimated_cost_usd` —
cumulative up to that iteration) alongside `arch` and `metrics`, so a single run's folder already
has everything needed to plot an accuracy-vs-compute-cost curve for that run.

After running the ablations above (each writes to its own `experiments/<name>_<timestamp>/`, since
`experiment.name` differs per file — nothing overwrites), fold every completed run into one table:

```bash
python scripts/aggregate_results.py
```

This scans every subdirectory of `experiments/`, pulls best val/test accuracy, iteration-to-best,
cumulative LLM cost, and which ablation flags were set out of each run's `metrics.json` +
`config.yaml`, and writes `experiments/ablation_summary.csv` (also printed as a table) — one row
per run, directly pastable into a paper's ablation table. Incomplete runs (no `metrics.json` yet)
are skipped automatically, so it's safe to run mid-sweep.
