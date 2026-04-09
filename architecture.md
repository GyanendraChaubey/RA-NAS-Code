# RA-NAS Architecture Document
**Reviewer**: Senior AI Engineer (Google Brain / DeepMind NAS perspective)
**Date**: 9 April 2026
**Codebase**: Reasoning-Agent Neural Architecture Search (RA-NAS)
**Status**: All identified issues resolved ✅

---

## 1. System Overview

RA-NAS is an LLM-guided Neural Architecture Search system. An LLM reasoning agent iteratively proposes and refines CNN architectures that are trained and evaluated on CIFAR-10. Results are stored in a memory module that provides in-context few-shot examples back to the LLM, closing a propose → train → evaluate → remember feedback loop.

**LLM Provider**: DeepSeek (`deepseek-chat` via OpenAI-compatible API). Configurable to any OpenAI-compatible provider via `configs/agent.yaml`.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           NASController                                 │
│                                                                         │
│  ┌──────────────┐   propose / refine   ┌──────────────────────────┐    │
│  │  LLMAgent    │◄────────────────────►│  PromptBuilder           │    │
│  │  (DeepSeek / │                      │  (dynamic schema +       │    │
│  │   mock mode) │                      │   memory injection)      │    │
│  └──────┬───────┘                      └──────────────────────────┘    │
│         │ arch dict                                                      │
│         ▼                                                                │
│  ┌──────────────┐   validate    ┌──────────────────────────────────┐   │
│  │Architecture  │◄─────────────│  SearchSpace + Constraints        │   │
│  │Generator     │  sample/mutate│  (search_space.py)               │   │
│  └──────┬───────┘               └──────────────────────────────────┘   │
│         │ arch dict                                                      │
│         ▼                                                                │
│  ┌──────────────┐  build   ┌───────────┐  train  ┌──────────────────┐ │
│  │ ModelBuilder │─────────►│ DynamicCNN│────────►│  Trainer         │ │
│  │              │          │(skip_projs│         │  (AdamW +        │ │
│  └──────────────┘          │ 1×1 Conv) │         │   Scheduler +    │ │
│                            └───────────┘         │   EarlyStopping) │ │
│                                                   └────────┬─────────┘ │
│                                                            │ metrics    │
│                                                            ▼            │
│                   ┌──────────────────┐  metrics  ┌──────────────────┐ │
│                   │ ExperimentMemory │◄──────────│    Evaluator     │ │
│                   │ (top-k, mid-run  │           │ (top1/5 acc,     │ │
│                   │  disk saves)     │           │  FLOPs, latency) │ │
│                   └──────────────────┘           └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
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
- **Fixed**: `__init__` now accepts an optional `prompt_builder` argument. If provided it is used directly; otherwise one is built internally. Eliminates the dead constructor work that was previously discarded by the caller.
- **Fixed**: `feedback_strategy: all` is now capped at `top_k_memory` entries, preventing unbounded prompt token growth.
- **Fixed**: `_init_openai_client` reads an optional `base_url` from config and passes it to the `OpenAI(...)` client, enabling OpenAI-compatible providers (e.g. DeepSeek) without any code change.
- **Fixed**: Provider validation now accepts `"openai"` and `"deepseek"`.

---

### 2.3 `src/agents/prompt_builder.py` — PromptBuilder ✅

- `build_proposal_prompt` and `build_refinement_prompt` cleanly separated.
- Payload truncation at `top_k` inside the builder is defensive and correct.
- **Fixed**: `_schema_text()` now derives activation and pooling options dynamically from `self.search_space["search_space"]`. Schema text and the actual search space cannot drift.

---

### 2.4 `src/nas/search_space.py` — SEARCH_SPACE + validate_architecture ✅

- Single source of truth for the search space.
- `validate_architecture` covers all required keys, type checks, range bounds, and list-length consistency.
- `_assert` helper keeps validation concise.
- No issues. Unchanged.

---

### 2.5 `src/nas/architecture_generator.py` — ArchitectureGenerator ✅

- Seeded `random.Random` keeps sampling reproducible without mutating global state.
- Constraint filtering in `_filter_choices`, `_kernel_choices`, `_activation_choices` is correct.
- `mutate` correctly re-samples `filters`/`kernels` lists when `num_layers` changes.
- Remaining known limitation: `mutate` does not call `validate_architecture` after mutation — callers must validate (the controller does this). Acceptable as-is.

---

### 2.6 `src/nas/controller.py` — NASController ✅

- Clean loop: propose → validate → build → train → evaluate → remember → refine.
- Double validation (before training and after refinement).
- Per-iteration structured logging.
- **Fixed (HIGH)**: `memory.save(str(memory_save_path))` is called after every `memory.add()`. A mid-run crash no longer loses collected data.
- **Fixed (MEDIUM)**: Runtime objects (`train_loader`, `val_loader`, `device`, `logger`, `experiment_dir`, `num_classes`) are now explicit constructor parameters. `config` holds only serialisable YAML data.
- **Fixed (MEDIUM)**: Exploration every `explore_every` iterations (default 3). `propose_architecture()` is forced instead of `refine_architecture()` on those iterations, preventing pure greedy convergence to local optima.

---

### 2.7 `src/models/cnn.py` — DynamicCNN ✅

- `nn.ModuleDict` per block for optional BN and Dropout is correct.
- Block order `Conv → BN → Act → Dropout → Pool` is standard.
- `AdaptiveAvgPool2d(1)` before classifier removes fixed spatial size constraints.
- **Fixed (MEDIUM)**: Skip connection channel alignment now uses a `nn.ModuleList` of learned `nn.Conv2d(kernel=1, bias=False)` projections registered in `__init__` (one per inter-block transition where channel sizes differ; `nn.Identity()` where they match). The old `_match_channels` zero-pad/slice path has been removed. Gradient flow through residual connections is now proper.

---

### 2.8 `src/models/model_builder.py` — ModelBuilder ✅

- `build_model` and `count_parameters` are correct.
- **Fixed (HIGH)**: `estimate_flops()` removed. The canonical implementation `flops_estimate()` in `src/evaluation/metrics.py` is the single source of truth. Both `Evaluator` and any future callers import from there.

---

### 2.9 `src/training/trainer.py` — Trainer ✅

- AdamW + CosineAnnealingLR/StepLR — production-appropriate choices.
- `zero_grad(set_to_none=True)` — PyTorch 2.x best practice.
- EarlyStopping integration is clean.
- Checkpoint saves `arch_config + state_dict + epoch + metrics`.
- Sample-weighted (not batch-count-weighted) loss/accuracy accumulation is correct.
- No changes required.

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
| `llm.provider` | `openai` or `deepseek` (validated at startup) |
| `llm.model` | Model name passed to the API (e.g. `deepseek-chat`, `gpt-4o`) |
| `llm.base_url` | Optional API base URL override (required for DeepSeek: `https://api.deepseek.com`) |
| `llm.api_key_env` | Name of the environment variable holding the key (never the key itself) |

Current `configs/agent.yaml`:
```yaml
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com
  temperature: 0.7
  max_tokens: 1024
  api_key_env: DEEPSEEK_API_KEY
```

To run:
```bash
export DEEPSEEK_API_KEY="sk-..."
python3 scripts/run_experiment.py
```

---

## 4. Test Coverage

| Test File | What is covered | Status |
|-----------|-----------------|--------|
| `test_memory.py` | Top-k ranking, save/load round-trip | ✅ Pass |
| `test_search_space.py` | Valid arch, invalid kernel, invalid filter | ✅ Pass |
| `test_model_builder.py` | Output shape, param count, 3 arch configs (incl. skip connections with learned projections) | ✅ Pass |
| `test_trainer.py` | 2-epoch smoke, loss trend, checkpoint | ✅ Pass |
| — | `LLMAgent` mock-mode propose/refine | ⚠️ Not yet written |
| — | `NASController` end-to-end loop | ⚠️ Not yet written |
| — | `ArchitectureGenerator.mutate()` | ⚠️ Not yet written |
| — | `EarlyStopping` max/min modes, patience boundary | ⚠️ Not yet written |
| — | `Evaluator` metric correctness | ⚠️ Not yet written |

---

## 5. Resolved Issues Tracker

| # | Severity | Location | Issue | Status |
|---|----------|----------|-------|--------|
| 1 | HIGH | `controller.py` | Memory never saved mid-run — crash = total data loss | ✅ Fixed |
| 2 | HIGH | `model_builder.py` | `estimate_flops` duplicated `flops_estimate` in `metrics.py` | ✅ Fixed |
| 3 | MEDIUM | `controller.py` | Runtime objects in config dict — un-serialisable anti-pattern | ✅ Fixed |
| 4 | MEDIUM | `controller.py` | Pure greedy refinement with no exploration — local optima risk | ✅ Fixed |
| 5 | MEDIUM | `llm_agent.py` | `PromptBuilder` constructed then immediately replaced externally | ✅ Fixed |
| 6 | MEDIUM | `llm_agent.py` | `feedback_strategy: all` had no token budget guard | ✅ Fixed |
| 7 | MEDIUM | `cnn.py` | Skip channel alignment via zero-pad/slice, not learned projection | ✅ Fixed |
| 8 | LOW | `memory.py` | `summary()` hardcoded `val_accuracy` metric | ✅ Fixed |
| 9 | LOW | `prompt_builder.py` | Schema text was a static string that could drift from search space | ✅ Fixed |
| 10 | LOW | `run_experiment.py` | `memory.save()` not called after experiment completes | ✅ Fixed |
| 11 | SECURITY | `configs/agent.yaml` | Raw API key committed to file instead of env-var reference | ✅ Fixed |

---

## 6. Architecture Strengths

- **Clean separation of concerns** — agent, memory, search space, model, training, evaluation are independent modules with well-defined interfaces.
- **Mock mode** — the full pipeline runs offline without an API key; essential for CI.
- **Reproducibility** — seeded RNG, `seed_everything`, deterministic DataLoader splits, config snapshot at experiment start.
- **Config-driven** — all hyperparameters live in YAML; no magic numbers in code.
- **Provider-agnostic LLM client** — `base_url` + env-var key pattern supports any OpenAI-compatible endpoint.
- **Crash-safe memory** — per-iteration disk persistence means partially completed runs are always recoverable.
- **Exploration-exploitation balance** — periodic forced re-proposals (`explore_every`) prevent greedy local optima.
- **Learned skip projections** — 1×1 `Conv2d` projections give residual connections proper gradient flow across channel-changing boundaries.
- **Correct checkpoint format** — `arch_config + state_dict + epoch + metrics` in one `.pt` file makes reconstruction unambiguous.

---
