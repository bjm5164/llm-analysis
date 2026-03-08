# LLM Analysis

Mechanistic interpretability of LLM computations under OOD token injection, using TransformerLens.

## Project structure

```
src/Analysis/       # Core library modules
  model.py              # Model loading, tokenization, cache helpers (HookedTransformer)
  logit_lens.py         # Logit lens trajectories, readability curves (accumulated_resid)
  attribution.py        # Direct logit attribution (decompose_resid, logit_attrs, stack_head_results)
  patching.py           # Activation patching sweeps (TL patching module + manual for unequal lengths)
  visualization.py      # Matplotlib plots: tokenization panels, readability, heatmaps, attention

experiments/            # Python experiment scripts with argparse CLI
  01_clean_baseline.py  # Clean prompt analysis: logit lens + attribution + top heads
  02_ood_comparison.py  # Clean vs corrupted vs control comparison
  03_activation_patching.py  # Patching sweeps to localize OOD disruption

scripts/                # Bash wrappers with env-var defaults
  run_baseline.sh
  run_ood_comparison.sh
  run_patching.sh
  run_all.sh            # Full pipeline
```

## Key conventions

- **TransformerLens only** — no raw HuggingFace hooks. Use HookedTransformer, ActivationCache, TL patching.
- **Explicit BOS** — always pass `prepend_bos=True/False` explicitly. Never assume.
- **Token verification** — always call `verify_tokenization` and `verify_answer_token` before experiments.
- **Selective caching** — use `names_filter` for memory-efficient sweeps (see `model.py` helpers).
- **Bash wraps Python** — experiments are parameterized Python scripts; bash scripts set defaults and chain them.

## Running experiments

```bash
# Quick sanity check
python main.py --model gpt2

# Full pipeline with defaults (qwen2.5-1.5b)
./scripts/run_all.sh

# Custom model and prompts
MODEL=qwen2.5-0.5b CLEAN="1 + 1 =" CORRUPTED="1 + 1 = {" ANSWER=" 2" ./scripts/run_all.sh
```

## TransformerLens API patterns used

- `model.run_with_cache(tokens)` — forward pass with activation cache
- `cache.accumulated_resid(apply_ln=True, pos_slice=-1)` — layerwise readability
- `cache.decompose_resid(apply_ln=True, return_labels=True)` — component attribution
- `cache.logit_attrs(resid_stack, answer_token_id)` — project onto answer direction
- `cache.stack_head_results()` + `cache.apply_ln_to_stack()` — per-head analysis
- `patching.get_act_patch_resid_pre()` etc — activation patching sweeps
- `model.run_with_hooks(tokens, fwd_hooks=[...])` — targeted interventions

## Unequal-length patching

When corrupted prompt is longer than clean (OOD suffix), TL's built-in patching helpers fail because they try to patch positions that don't exist in the clean cache. The patching module falls back to manual per-(layer,position) sweeps over the shared prefix. For proper patching with suffix tokens, use matched-length controls or targeted interventions (zero_suffix_contribution, patch_position_at_layer).
