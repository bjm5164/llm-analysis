---
name: transformerlens-qwen-setup-and-model-loading
description: Load and validate a Qwen-family model in TransformerLens before mechanistic interpretability work, with special attention to tokenization, BOS handling, cache strategy, and Qwen-specific gotchas.
---

# TransformerLens Qwen Setup and Model Loading

Use this skill when the task is to get a Qwen-family model into a stable, inspectable TransformerLens workflow before deeper interpretability experiments.

## What this skill is for

This skill is the setup layer for every later analysis. It is for cases like:
- loading a Qwen, Qwen2, Qwen2.5, or Qwen3 family model into `HookedTransformer`
- checking that tokenization matches the exact strings you want to analyze
- deciding whether to prepend BOS or not
- confirming hook names and tensor shapes
- choosing a caching strategy that will not explode memory
- preparing a notebook environment for interactive exploration

## Why this matters

TransformerLens works by exposing a consistent `HookedTransformer` interface across many pretrained models, and it explicitly supports Qwen-family weight conversions in the docs. The model-properties table also lists many Qwen2, Qwen2.5, and Qwen3 checkpoints, which makes Qwen a first-class target for this workflow.

For your project, setup errors are especially costly because OOD-token studies are hypersensitive to tokenization and prompt formatting. A single unexpected BOS token, spacing difference, or tokenizer edge case can invalidate a tracing result.

## Core APIs to rely on

- `HookedTransformer.from_pretrained(...)`
- `model.to_tokens(...)`
- `model.to_str_tokens(...)`
- `model.to_single_token(...)`
- `model.to_single_str_token(...)`
- `model.run_with_cache(...)`
- `transformer_lens.utils.test_prompt(...)`
- `transformer_lens.utils.get_act_name(...)`

## Recommended workflow

### 1. Start from a supported Qwen checkpoint

Prefer a model that already appears in the TransformerLens model-properties table or release notes. Smaller Qwen-family models are best for rapid iteration, interactive plotting, and debugging OOD injections.

Good first-pass choices:
- a Qwen2.5 small/base model if you want strong modern behavior with moderate memory
- Qwen3-0.6B if you specifically want the newest officially mentioned Qwen3 base support

### 2. Fix tokenization before any interpretability claims

Always inspect:
- the exact token IDs for the clean prompt
- the exact token IDs for the OOD-injected prompt
- whether the answer token is a single token or multi-token string
- whether your arithmetic target is `" 2"`, `"2"`, or multiple tokens under this tokenizer

For arithmetic prompts, do not assume GPT-2-style token behavior carries over to Qwen.

### 3. Decide BOS handling explicitly

TransformerLens warns that BOS and tokenizer behavior can materially change tokenization. For comparison experiments, keep BOS handling fixed across all prompt variants.

Rule of thumb:
- if using string prompts, keep `prepend_bos` explicit everywhere
- if using pre-tokenized tensors, standardize whether BOS is already included

### 4. Run a minimal cache sanity check

Before a real experiment:
- tokenize a tiny prompt
- run `run_with_cache(..., remove_batch_dim=True)`
- inspect key activations like `resid_pre`, `attn.hook_pattern`, and `mlp.hook_post`
- verify layer and head counts match expectations

### 5. Use selective caching when scaling up

Full-cache runs are perfect for short prompts and exploratory analysis, but repeated multi-condition OOD experiments will become memory-bound. Use `names_filter` to cache only the activations needed for a given experiment.

Useful cache subsets:
- only residual-stream hooks for logit-lens and DLA work
- only attention patterns plus a few residual hooks for readability studies
- only one or two late layers when debugging localized effects

## Qwen-specific practical notes

### RMSNorm and rotary embeddings

The model-properties table shows Qwen-family models using RMS normalization and rotary positional embeddings. This matters because:
- residual-stream comparisons across positions should respect positional effects
- LN-style intuitions from older GPT-2 analyses do not port over perfectly
- when projecting to logits, use TransformerLens helpers rather than ad hoc assumptions

### Grouped-query / KV-head issues

Recent release notes mention fixes for key/value head patching in models where the number of attention heads differs from the number of key-value heads. This is directly relevant to modern architectures and is a reason to keep patching code close to official utilities rather than rolling your own blindly.

### Context length and vocabulary size

Several Qwen-family entries in the model-properties table have large vocabularies and longer contexts than older GPT-2-family defaults. This affects memory pressure during full caching and should influence how aggressively you filter cached activations.

## Suggested setup template

```python
import torch
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils

model_name = "qwen2.5-1.5b"  # example alias; verify the exact supported alias first
model = HookedTransformer.from_pretrained(
    model_name,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

prompt = "what is 1 + 1"
tokens = model.to_tokens(prompt, prepend_bos=True)
print(tokens)
print(model.to_str_tokens(tokens))

logits, cache = model.run_with_cache(tokens, remove_batch_dim=False)
print(logits.shape)
print(cache[utils.get_act_name("resid_pre", 0)].shape)
```

## What success looks like

You know setup is good when:
- the prompt and answer tokenize exactly as expected
- you can name the answer token(s) precisely
- `run_with_cache` works without shape surprises
- you know which hook names you need for the next experiment
- you have a small-model config for iteration and a larger-model config for confirmatory runs

## Common failure modes

- comparing prompts with inconsistent BOS settings
- forgetting that `" 2"` and `"2"` may tokenize differently
- assuming cache tensors are shape `[pos, d_model]` when batch dim is still present
- caching everything during large sweeps and running out of VRAM or RAM
- reusing custom hooks without clearing them

## Hand-off to the next skills

Once setup is stable:
- use the answer-tracing skill to find where the model builds evidence for `2`
- use the OOD-patching skill to test whether those features survive later injected tokens
- use the visualization skill to inspect heads and positions interactively

## References

- TransformerLens overview and purpose
- TransformerLens `HookedTransformer` docs
- TransformerLens `hook_points` docs
- TransformerLens model-properties table
- TransformerLens release notes mentioning Qwen3 support and KV-head patching fixes
