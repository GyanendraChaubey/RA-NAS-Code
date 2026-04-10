# RA-NAS Architecture Document
**Reviewer**: Senior AI Engineer (Google Brain / DeepMind NAS perspective)
**Date**: 10 April 2026
**Codebase**: Reasoning-Agent Neural Architecture Search (RA-NAS)
**Status**: All identified issues resolved ✅

---

## 1. System Overview

RA-NAS is an LLM-guided Neural Architecture Search system. An LLM reasoning agent iteratively proposes and refines CNN architectures that are trained and evaluated on CIFAR-10. Results are stored in a memory module that provides in-context few-shot examples back to the LLM, closing a propose → train → evaluate → remember feedback loop.

**LLM Provider**: Groq (`llama-3.3-70b-versatile` via OpenAI-compatible API). Configurable to any OpenAI-compatible provider via `configs/agent.yaml`.

Five research phases are fully implemented:

| Phase | Feature | Module |
|---|---|---|
| 1 | Structured LLM reasoning (observations → hypothesis → changes → risks) | `llm_agent.py`, `prompt_builder.py` |
| 2 | SE blocks as optional NAS dimension | `cnn.py`, `search_space.py` |
| 3 | Diversity penalty — discourages architectures similar to top-k memory | `llm_agent.py`, `controller.py` |
| 4 | Multi-fidelity screening — 10-epoch proxy before 200-epoch full train | `controller.py` |
| 5 | Self-correcting agent — tracks prediction errors, adjusts confidence | `llm_agent.py`, `memory.py` |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              NASController                                  │
│                                                                             │
│  ┌──────────────┐   propose / refine   ┌────────────────────────────────┐  │
│  │  LLMAgent    │◄────────────────────►│  PromptBuilder                 │  │
│  │  (Groq /     │  structured reasoning│  (dynamic schema, block_depths,│  │
│  │   mock mode) │  + diversity penalty │   ResNet hints, top-k memory)  │  │
│  └──────┬───────┘                      └────────────────────────────────┘  │
│         │ arch dict (num_layers, filters, kernels, block_depths, …)         │
│         ▼                                                                   │
│  ┌──────────────────┐  validate   ┌──────────────────────────────────────┐ │
│  │ ArchitectureGen  │◄────────────│  SearchSpace + Constraints           │ │
│  │ sample / mutate  │             │  filters=[64,128,256,512]            │ │
│  └──────┬───────────┘             │  block_depths=[1,2,3] per stage      │ │
│         │ arch dict               └──────────────────────────────────────┘ │
│         ▼                                                                   │
│  ┌──────────────┐  build  ┌──────────────────────┐                         │
│  │ ModelBuilder │────────►│ DynamicCNN           │                         │
│  │              │         │ Stem(64ch,3×3)        │                         │
│  └──────────────┘         │ → N ResNet stages     │                         │
│                           │   (ResBottleneckBlock)│                         │
│                           │ → GlobalAvgPool → FC  │                         │
│                           └──────────┬───────────┘                         │
│                                      │ PyTorch model                        │
│                                      ▼                                      │
│  ┌────────────────────────────────────────┐                                 │
│  │  Phase 4: Multi-Fidelity Screening     │                                 │
│  │  Odd iter  → screen 10 epochs (fast)   │                                 │
│  │  Even iter → compare pair, full-train  │                                 │
│  │              winner for 200 epochs     │                                 │
│  └──────────────────┬─────────────────────┘                                │
│                     │ train metrics                                          │
│                     ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Trainer: SGD+Nesterov, cosine_warmup LR, RandAugment, Mixup,       │  │
│  │           CutOut, label smoothing, SWA (starts at 75% of training)  │  │
│  └──────────────────────────────┬───────────────────────────────────────┘  │
│                                 │ checkpoint + metrics                       │
│                                 ▼                                            │
│               ┌─────────────────────────┐  metrics  ┌──────────────────┐   │
│               │ ExperimentMemory        │◄──────────│   Evaluator      │   │
│               │ top-k, mid-run saves,   │           │ top1/5 acc,      │   │
│               │ prediction error track  │           │ FLOPs, latency   │   │
│               └─────────────────────────┘           └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module-by-Module Status

### 2.1 `src/agents/memory.py` — ExperimentMemory ✅

- `add`, `get_top_k`, `get_all`, `save`, `load`, `summary` — all correct.
- `save`/`load` JSON round-trip verified by tests.
- **Fixed**: `summary(k, metric)` now accepts a `metric` parameter (default `"val_accuracy"`) consistent with `get_top_k()`. The sort key and the output dict key are both driven by `metric`.
- Remaining known limitation: no hard capacity cap on `_entries` (acceptable for typical NAS run lengths).

---

### 2.2 `src/agents/llm_agent.py` — LLMAgent ✅

- Mock mode enables full offline testing with no API calls.
- Retry-on-invalid loop (`retry_on_invalid`) is correct.
- Three feedback strategies (`top_k`, `threshold`, `all`) are sound.
- API key read from environment variable — no credentials in source.
- `__init__` accepts an optional `prompt_builder` argument; builds one internally otherwise.
- `feedback_strategy: all` is capped at `top_k_memory` entries — no unbounded prompt token growth.
- `_init_openai_client` reads `base_url` from config for any OpenAI-compatible provider (Groq, etc.).
- Provider validation accepts `"openai"` and `"groq"`.
- **Phase 3**: Diversity penalty flag (`diversity_penalty: true`) embeds a warning in the prompt when proposed arch is too similar to top-k memory entries.
- **Phase 5**: Self-correcting agent tracks LLM accuracy predictions vs. actual results; feeds prediction error summary back into the refinement prompt.

---

### 2.3 `src/agents/prompt_builder.py` — PromptBuilder ✅

- `build_proposal_prompt` and `build_refinement_prompt` cleanly separated.
- Payload truncation at `top_k` inside the builder is defensive and correct.
- `_schema_text()` derives activation/pooling options dynamically from `self.search_space` — schema and search space cannot drift.
- Schema includes `block_depths` (list of ints, one per stage) with inline comments explaining fixed fields (`use_batchnorm`, `use_skip_connections`, `pooling`).
- Proposal prompt includes a ResNet-oriented template: *4 stages, filters=[64,128,256,512], block_depths=[2,2,2,2]* as a strong starting hint for 93%+.
- Refinement prompt hints focus on increasing `block_depths`, wider filters (256/512), and SE blocks for pushing past 90%.

---

### 2.4 `src/nas/search_space.py` — SEARCH_SPACE + validate_architecture ✅

- Single source of truth for the search space.
- `validate_architecture` covers all required keys, type checks, range bounds, and list-length consistency.
- `_assert` helper keeps validation concise.
- Current search space reflects ResNet bottleneck design:

```python
SEARCH_SPACE = {
    "num_layers":         {"min": 2, "max": 8},        # stages
    "filters_per_layer":  [64, 128, 256, 512],
    "kernel_sizes":       [3, 5],
    "block_depths":       [1, 2, 3],                   # bottleneck blocks per stage
    "activations":        ["relu", "gelu", "silu"],
    "use_batchnorm":      [True],                      # fixed
    "use_dropout":        [True, False],
    "dropout_rate":       {"min": 0.0, "max": 0.3},
    "use_skip_connections": [True],                   # fixed
    "use_se_blocks":      [True, False],
    "pooling":            ["avg"],                     # fixed
}
```

- `validate_architecture` now also requires and validates the `block_depths` key (list length == `num_layers`, each value in `[1, 2, 3]`).

---

### 2.5 `src/nas/architecture_generator.py` — ArchitectureGenerator ✅

- Seeded `random.Random` keeps sampling reproducible without mutating global state.
- Constraint filtering in `_filter_choices`, `_kernel_choices`, `_activation_choices` is correct.
- `sample_random` generates a `block_depths` list (one random value from `[1,2,3]` per stage); always sets `use_batchnorm=True`, `use_skip_connections=True`, `pooling='avg'`.
- `mutate` candidates include `block_depths`; `num_layers` change also resizes `block_depths` list consistently.
- Fixed fields (`use_batchnorm`, `use_skip_connections`, `pooling`) are excluded from mutation candidates.
- Remaining known limitation: `mutate` does not call `validate_architecture` after mutation — callers must validate (the controller does this). Acceptable as-is.

---

### 2.6 `src/nas/controller.py` — NASController ✅

- Clean loop: propose → validate → build → screen/train → evaluate → remember → refine.
- Double validation (before training and after refinement).
- Per-iteration structured logging.
- `memory.save()` called after every `memory.add()` — mid-run crash never loses data.
- Runtime objects (`train_loader`, `val_loader`, `device`, `logger`, `experiment_dir`, `num_classes`) are explicit constructor parameters. `config` holds only serialisable YAML data.
- Exploration every `explore_every` iterations (default 2): `propose_architecture()` is forced instead of `refine_architecture()`, preventing pure greedy convergence to local optima.
- **Phase 4 — Multi-fidelity screening** (`_screened_train`):
  - *Odd iterations*: screen current arch for `screening_epochs` (10) only; return screening metrics directly (no extra full-train).
  - *Even iterations*: compare the last two buffered screens; full-train the winner for the full `epochs` (200); discard the loser.
  - Net effect: every 2 proposals → 1 full training run. Halves wall-clock time compared to full-training every arch.

---

### 2.7 `src/models/cnn.py` — DynamicCNN ✅ (complete rewrite)

The flat `Conv → BN → Act → Pool` design has been replaced with a **pre-activation ResNet v2 bottleneck** architecture capable of 93–95% on CIFAR-10.

**`SEBlock`** — Squeeze-and-Excitation channel recalibration (optional per arch):
- `AdaptiveAvgPool2d(1)` → `Linear(ch, ch//16)` → `ReLU` → `Linear(ch//16, ch)` → `Sigmoid` → scale.

**`ResBottleneckBlock`** — Pre-activation bottleneck (He et al. 2016 v2):
```
BN → Act → Conv1×1(in→mid)  [mid = out//4]
BN → Act → Conv3×3(mid→mid, stride, kernel_size)
BN → Act → Conv1×1(mid→out)
→ SE (optional)
→ Dropout2d (optional)
+ shortcut (AvgPool+Conv1×1 when stride>1 or channels differ, else Identity)
```

**`DynamicCNN`** — Variable-depth staged network:
```
Stem: Conv3×3(3→64)/BN/Act   ← no downsampling on 32×32 CIFAR
→ Stage 0: stride=1, depth=block_depths[0] ResBottleneckBlocks
→ Stage 1: stride=2, depth=block_depths[1] ResBottleneckBlocks
→ …
→ Stage N-1: stride=2
→ AdaptiveAvgPool2d(1)
→ Linear(last_filters, num_classes)
```
- Kaiming init for Conv2d, ones/zeros for BN, zero-bias for Linear.
- No manual pooling guards needed — downsampling is stride-based and cannot produce 0×0 maps.

---

### 2.8 `src/models/model_builder.py` — ModelBuilder ✅

- `build_model` and `count_parameters` are correct.
- **Fixed (HIGH)**: `estimate_flops()` removed. The canonical implementation `flops_estimate()` in `src/evaluation/metrics.py` is the single source of truth. Both `Evaluator` and any future callers import from there.

---

### 2.9 `src/training/trainer.py` — Trainer ✅ (extended)

- **Optimizer**: SGD with Nesterov momentum (`momentum=0.9`, `weight_decay=1e-3`). AdamW still available via `optimizer: adamw` config key.
- **Scheduler**: `cosine_warmup` — `LinearLR` warmup for `warmup_epochs` (10) composed with `CosineAnnealingLR` via `SequentialLR`.
- **Label smoothing**: `nn.CrossEntropyLoss(label_smoothing=0.1)` — read from `training.label_smoothing` config key.
- **RandAugment**: `torchvision.transforms.RandAugment(num_ops=2, magnitude=9)` applied per batch (uint8 conversion → augment → float). Enabled via `training.augmentation.randaugment`.
- **Mixup**: Beta(alpha, alpha) interpolation per batch. Alpha read from `training.augmentation.mixup_alpha` (default 0.4).
- **CutOut**: Random square masking. Length read from `training.augmentation.cutout_length` (default 8).
- **SWA** (Stochastic Weight Averaging): `AveragedModel` + `SWALR` from `torch.optim.swa_utils`. Starts at `swa_start_frac` (0.75) of total epochs. `update_bn(train_loader)` called at end; SWA weights swapped into model before final eval.
- `zero_grad(set_to_none=True)` — PyTorch 2.x best practice.
- EarlyStopping integration is clean.
- Checkpoint saves `arch_config + state_dict + epoch + metrics`.
- Sample-weighted (not batch-count-weighted) loss/accuracy accumulation is correct.

---

### 2.10 `src/training/early_stopping.py` — EarlyStopping ✅

Clean, minimal, correct. `reset()` allows safe reuse across runs. No changes required.

---

### 2.11 `src/evaluation/evaluator.py` — Evaluator ✅

- Reports top-1, top-5 accuracy, loss, param count, FLOPs, inference latency.
- Warmup runs before timing — correct for GPU synchronization.
- `input_size` inferred from live data rather than hardcoded.
- No changes required.

---

### 2.12 `src/utils/config_loader.py` — ConfigLoader ✅

Deep merge correct. `yaml.safe_dump` used throughout — correct security practice. No changes required.

---

### 2.13 `src/utils/logger.py` — Logger ✅

`propagate = False` and handler-clearing guards prevent duplicate log lines. No changes required.

---

### 2.14 `scripts/run_experiment.py` — Entry Point ✅

- `seed_everything` covers Python, NumPy, PyTorch, and CUDA.
- CIFAR-10 offline fallback via `FakeData` handles air-gapped environments.
- Config snapshot saved to experiment directory for reproducibility.
- **Fixed**: Removed redundant `PromptBuilder` import and construction. `LLMAgent` builds its own via dependency injection.
- **Fixed**: Removed `runtime_config = merge_configs(...)` anti-pattern that deep-copied non-serialisable objects (`DataLoader`, `Logger`). `NASController` now receives all runtime values as explicit keyword arguments.
- **Fixed**: `memory.save()` is now called inside the controller loop after every iteration — the post-experiment call in `main()` remains as a final flush.

---

## 3. LLM Provider Configuration

The system uses the `openai` Python package as a unified client for any compatible provider.

| Config key | Purpose |
|---|---|
| `llm.provider` | `openai` or `groq` (validated at startup) |
| `llm.model` | Model name passed to the API (e.g. `llama-3.3-70b-versatile`, `gpt-4o`) |
| `llm.base_url` | Optional API base URL override (required for Groq: `https://api.groq.com/openai/v1`) |
| `llm.temperature` | Initial sampling temperature (annealed each iteration by `temperature_decay`) |
| `llm.temperature_min` | Floor temperature after annealing |
| `llm.temperature_decay` | Subtracted from temperature each iteration |
| `llm.api_key_env` | Name of the environment variable holding the key (never the key itself) |

Current `configs/agent.yaml`:
```yaml
llm:
  provider: groq
  model: llama-3.3-70b-versatile
  base_url: https://api.groq.com/openai/v1
  temperature: 1.2
  temperature_min: 0.7
  temperature_decay: 0.05
  max_tokens: 1536
  api_key_env: GROQ_API_KEY
```

To run:
```bash
export GROQ_API_KEY="gsk_..."
python3 scripts/run_experiment.py
```

---

## 4. Test Coverage

**52 tests — all passing.**

| Test File | Tests | What is covered |
|-----------|-------|-----------------|
| `test_memory.py` | 1 | Top-k ranking, save/load JSON round-trip |
| `test_search_space.py` | 3 | Valid arch (with `block_depths`), invalid kernel, constraint filter range |
| `test_model_builder.py` | 1 | Output shape (3 arch configs), param count, ResNet bottleneck build |
| `test_trainer.py` | 1 | 2-epoch smoke: loss trend, checkpoint saved |
| `test_architecture_generator.py` | 12 | `sample_random` reproducibility, constraint honouring, list-length consistency, `block_depths` generation; `mutate` produces valid arch, different arch, depth/list consistency, dropout=0 on disabled |
| `test_early_stopping.py` | 12 | max/min modes, patience boundary, `reset()` reuse, invalid mode, missing metric |
| `test_evaluator.py` | 11 | Required keys, correct types, accuracy range, top5≥top1, FLOPs/params positive, perfect/worst-case accuracy, deeper model has more params+FLOPs |
| `test_nas_controller.py` | 11 | Correct number of records, sequential indices, expected metric keys, valid archs, iteration dirs created, checkpoint saved, memory saved per iteration, memory is valid JSON, exploration triggers fresh propose, no exploration when disabled |

---

## 5. Resolved Issues Tracker

| # | Severity | Location | Issue | Status |
|---|----------|----------|-------|--------|
| 1 | HIGH | `controller.py` | Memory never saved mid-run — crash = total data loss | ✅ Fixed |
| 2 | HIGH | `model_builder.py` | `estimate_flops` duplicated `flops_estimate` in `metrics.py` | ✅ Fixed |
| 3 | MEDIUM | `controller.py` | Runtime objects in config dict — un-serialisable anti-pattern | ✅ Fixed |
| 4 | MEDIUM | `controller.py` | Pure greedy refinement with no exploration — local optima risk | ✅ Fixed |
| 5 | MEDIUM | `controller.py` | Phase 4 bug: odd iterations did screen + full-train (double work) | ✅ Fixed |
| 6 | MEDIUM | `llm_agent.py` | `PromptBuilder` constructed then immediately replaced externally | ✅ Fixed |
| 7 | MEDIUM | `llm_agent.py` | `feedback_strategy: all` had no token budget guard | ✅ Fixed |
| 8 | MEDIUM | `cnn.py` | Flat `Conv→Pool` design caused `RuntimeError: pooling output 0×0` on deep archs | ✅ Fixed (stride-based downsampling) |
| 9 | MEDIUM | `cnn.py` | Skip channel alignment via zero-pad/slice, not learned projection | ✅ Fixed (now full ResNet bottleneck with AvgPool+Conv1×1 shortcut) |
| 10 | LOW | `memory.py` | `summary()` hardcoded `val_accuracy` metric | ✅ Fixed |
| 11 | LOW | `prompt_builder.py` | Schema text was a static string that could drift from search space | ✅ Fixed |
| 12 | LOW | `run_experiment.py` | `memory.save()` not called after experiment completes | ✅ Fixed |
| 13 | SECURITY | `configs/agent.yaml` | Raw API key committed to file instead of env-var reference | ✅ Fixed |

---

## 6. Architecture Strengths

- **Clean separation of concerns** — agent, memory, search space, model, training, evaluation are independent modules with well-defined interfaces.
- **Mock mode** — the full pipeline runs offline without an API key; essential for CI.
- **Reproducibility** — seeded RNG, `seed_everything`, deterministic DataLoader splits, config snapshot at experiment start.
- **Config-driven** — all hyperparameters live in YAML; no magic numbers in code.
- **Provider-agnostic LLM client** — `base_url` + env-var key pattern supports any OpenAI-compatible endpoint (Groq, DeepSeek, OpenAI, etc.).
- **Crash-safe memory** — per-iteration disk persistence means partially completed runs are always recoverable.
- **Exploration-exploitation balance** — periodic forced re-proposals (`explore_every=2`) prevent greedy local optima.
- **ResNet bottleneck model** — pre-activation v2 design with stride-based downsampling eliminates pooling-on-small-map crashes and targets 93–95% CIFAR-10 accuracy.
- **`block_depths` as NAS dimension** — the LLM searches over per-stage block counts `[1,2,3]` in addition to filter widths and kernel sizes.
- **SE blocks** — channel-attention recalibration is searchable; LLM learns when to use it.
- **Multi-fidelity screening (Phase 4)** — 10-epoch proxy halves wall-clock time; only the winner of each pair gets the full 200-epoch run.
- **Advanced augmentation pipeline** — RandAugment + Mixup (alpha=0.4) + CutOut + label smoothing all active together.
- **SWA** — Stochastic Weight Averaging starting at 75% of training improves generalisation without extra compute.
- **Correct checkpoint format** — `arch_config + state_dict + epoch + metrics` in one `.pt` file makes reconstruction unambiguous.

---
