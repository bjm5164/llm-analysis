# LLM Analysis

Mechanistic interpretability of LLM computations under out-of-distribution (OOD) token injection.
Built on [TransformerLens](https://github.com/neelnanda-io/TransformerLens) targeting Qwen3-4B on a single 24 GB GPU.

## What this does

The core question: **when an OOD token sequence is injected into a prompt, which components of the model's computation break, and which survive?**

The approach is direct logit attribution (DLA) + activation patching:

1. Run the clean prompt, decompose which heads/MLPs write the answer logit.
2. Inject whitespace or newline tokens at random positions to create corrupted variants.
3. Compare attribution before and after — which heads shift, which stay stable.
4. Activation-patch clean residual stream back into the corrupted run to localise the disruption.

---

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd llm-analysis
uv sync
```

For SAELens support (optional, has numpy version constraints):

```bash
uv sync --extra sae
```

---

## Configuration

Everything is controlled by `config.yaml`. Key fields:

```yaml
model:
  name: Qwen/Qwen3-4B       # any HuggingFace model ID TransformerLens supports
  dtype: float16
  cache_strategy: full       # full | resid | attn_pattern | minimal

prompts:
  clean: "1 + 1 = "
  corrupted: '1 + 1 = {"answer": '

tokens:
  answer: "2"
  distractor:                # optional — enables logit-diff metric

corruption_sweep:
  variants:
    - {label: "newline_3_s1", inject: ["\n"], count: 3, seed: 1}
    - {label: "mixed_5_s1",   inject: ["\n", " "], count: 5, seed: 1}
```

Cache strategies trade memory for coverage:

| Strategy | What is cached | Use case |
|---|---|---|
| `full` | everything | exploration, short prompts |
| `resid` | residual stream only | DLA without attention patterns |
| `attn_pattern` | attention patterns only | attention-only analysis |
| `minimal` | resid_pre + patterns | memory-constrained runs |

---

## Streamlit app

The interactive app is the primary interface. It exposes all four experiments as separate pages with live config editing, Plotly charts, and GPU memory management.

```bash
streamlit run app.py
```

| Page | Experiment |
|---|---|
| **Model** | Load model, sanity check, answer token verification |
| **Baseline** | Clean prompt DLA — component bar chart + head attribution heatmap |
| **OOD** | Clean vs corrupted — head diff heatmap, component attribution diff (raw + normalised), attention patterns |
| **Patching** | Activation patching sweep — resid_pre / attn_out / mlp_out / head_out rescue heatmaps |
| **Sweep** | Corruption sweep — P(answer) table + summary heatmap across all injection variants |

The sidebar on every page shows live GPU memory (driver-level, accounting for all processes). Two cache controls:

- **Clear Cache** — releases PyTorch reserved-but-unused memory back to the driver.
- **Hard Reset** — evicts the model from GPU entirely (`st.cache_resource.clear()`). Use before loading a different model or when you need the memory for another task. The model must be reloaded before running experiments again.

The **Config Editor** on the home page lets you edit `config.yaml` in-browser. Clicking **Apply** propagates changes to all pages and clears stale experiment results.

---

## Command-line experiments

The same experiments are available as standalone scripts that save static PNG plots to `plots/<experiment>/`.

```bash
# Experiment 01: clean baseline — component and head attribution
uv run python experiments/01_clean_baseline.py

# Experiment 02: OOD comparison — clean vs corrupted attribution diff
uv run python experiments/02_ood_comparison.py

# Experiment 03: activation patching — localise disruption by (layer, position)
uv run python experiments/03_activation_patching.py

# Experiment 04: corruption sweep — compare attribution across injection variants
uv run python experiments/04_corruption_sweep.py

# Use a different config
uv run python experiments/01_clean_baseline.py -c my_config.yaml
```

---

## Library

`src/Analysis/` is usable directly in notebooks:

```python
import sys
sys.path.insert(0, "src")

from Analysis.config import load_config
from Analysis.model import load_model, run_with_cache, corrupt_tokens, gpu_memory, free_memory
from Analysis.attribution import final_logit_margin, component_attribution, head_attribution
from Analysis.patching import make_logit_diff_metric, patch_resid_pre, compare_clean_corrupted

cfg = load_config()
model = load_model(cfg.model)

tokens = model.to_tokens("1 + 1 = ", prepend_bos=True)
logits, cache = run_with_cache(model, tokens, strategy="full")

margin = final_logit_margin(model, logits, "2")
# component_attribution sums to the answer logit
comp_attrs, labels = component_attribution(model, cache, "2", pos=-1)
# head_attribution is attention-heads only (subset of comp_attrs)
head_attrs = head_attribution(model, cache, "2", pos=-1)  # (n_layers, n_heads)

free_memory(logits, cache)   # delete tensors, gc.collect(), empty_cache()
gpu_memory()                 # print driver-level stats
```

### Key functions

**`model.py`**

| Function | Description |
|---|---|
| `load_model(cfg)` | Load a HookedTransformer from config |
| `run_with_cache(model, tokens, strategy)` | Forward pass under `torch.no_grad()` with selective caching |
| `corrupt_tokens(model, tokens, cfg)` | Inject whitespace/newline tokens at random positions |
| `gpu_memory()` | Print and return driver-level GPU memory stats |
| `free_memory(*args)` | Delete tensors, `gc.collect()`, `empty_cache()` |

**`attribution.py`**

| Function | Description |
|---|---|
| `final_logit_margin(model, logits, answer)` | P(answer), logit, rank |
| `component_attribution(model, cache, answer)` | Per-block DLA (embed + attn + MLP per layer) — **sums to the answer logit** |
| `head_attribution(model, cache, answer)` | Per-head DLA as `(n_layers, n_heads)` tensor — attention heads only |

**`patching.py`**

| Function | Description |
|---|---|
| `compare_clean_corrupted(...)` | Run both prompts, report baseline metric before patching |
| `patch_resid_pre / patch_attn_out / patch_mlp_out` | Layer × position rescue sweeps |
| `patch_head_out` | Layer × head × position sweep (slow — opt in explicitly) |

---

## Memory notes

Qwen3-4B in float16 uses ~8 GB for weights. A full activation cache on a short prompt adds 3–7 GB depending on sequence length.

- Use `cache_strategy: resid` for DLA-only runs (skips caching attention patterns).
- Call `free_memory(logits, cache)` in notebooks after each step.
- The `head_out` patching sweep is O(n_layers × n_heads × n_positions) forward passes — expect several minutes on Qwen3-4B.
- Hard Reset in the app sidebar is the fastest way to fully reclaim GPU memory between sessions.
