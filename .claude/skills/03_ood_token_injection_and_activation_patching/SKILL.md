---
name: transformerlens-ood-token-injection-and-activation-patching
description: Study whether answer-relevant features survive, degrade, or become unreadable after injecting out-of-distribution tokens, using clean/corrupted prompt design, activation patching, and targeted interventions.
---

# TransformerLens OOD Token Injection and Activation Patching

Use this skill when the task is to causally test how injected out-of-distribution tokens affect a computation that previously supported a target answer.

## What this skill is for

This skill is the core of your project. It is for questions like:
- after I inject strange or constrained-decoding tokens later in the sequence, does the model still preserve the `" 2"` feature?
- if the answer stops being predicted, where was the computation disrupted?
- was the feature erased, rotated away from the readout, or merely no longer attended to?
- which layers, heads, or positions are most vulnerable to OOD-token interference?

## Core idea

First establish a clean prompt that yields the expected answer. Then create one or more corrupted prompts that insert OOD tokens after the task-relevant content. Finally, use activation patching to ask which clean activations restore correct behavior when transplanted into the corrupted run.

TransformerLens provides a generic patching framework plus specialized helpers for common activation families.

## Main APIs

- `transformer_lens.patching.generic_activation_patch(...)`
- `get_act_patch_resid_pre(...)`
- `get_act_patch_resid_mid(...)`
- `get_act_patch_attn_out(...)`
- `get_act_patch_mlp_out(...)`
- `get_act_patch_attn_head_*` helpers
- `model.run_with_hooks(...)`
- `ActivationCache` from the clean run

## Experimental design

### Prompt families

Build at least three prompt classes:

1. **Clean**
   - `what is 1 + 1`

2. **OOD-suffix injected**
   - same base prompt followed by one or more strange tokens that could appear under constrained decoding
   - examples may include malformed fragments, impossible delimiter sequences, or tokenizer-rare suffixes

3. **Matched in-distribution suffix control**
   - same length and same approximate position structure, but using ordinary continuation tokens

This separation matters. Without a matched in-distribution control, you cannot tell whether the effect is due to OOD-ness or just due to extra sequence length.

### Metrics

Use at least two:
- answer-token logit or logit difference
- final loss or rank of the correct answer token

For arithmetic, a logit-difference metric is usually the most interpretable.

## Step-by-step workflow

### 1. Identify the corruption effect

Before patching anything, compare clean vs corrupted runs on:
- answer logit
- answer rank
- layerwise readability using `accumulated_resid`

This gives you a first sense of whether the feature is:
- absent entirely,
- present but weakened,
- or still present yet not effectively routed to the output.

### 2. Residual-stream patch sweeps

Start with `get_act_patch_resid_pre()` over layer and position. This is the fastest way to localize where restoring the clean residual stream rescues the answer.

Interpretation:
- a strong rescue at one late position suggests the clean answer representation was lost or overwritten there
- rescue spread over many early positions suggests the corruption perturbed information routing more globally

### 3. Split by sublayer type

Run:
- `get_act_patch_attn_out()`
- `get_act_patch_mlp_out()`
- `get_act_patch_resid_mid()`

This tells you whether the damage is mostly happening through attention routing, MLP writing, or intermediate residual composition.

### 4. Drill down to heads

If an attention block is implicated, use the head-level patching helpers:
- head outputs
- q/k/v vectors
- patterns, where relevant

This is especially important for your question about readability by later injected tokens. A feature may still exist in the residual stream, but later attention heads may stop reading it correctly.

### 5. Test "feature present but unreadable" explicitly

This is the key subtlety in your problem.

A practical test battery:
- compare clean vs corrupted readability of the answer direction in the residual stream
- patch only reader heads or later residual states
- patch only writer components identified from the clean analysis

Interpretive pattern:
- if the answer direction remains readable but prediction fails, suspect failed reading / routing
- if the answer direction becomes weak or absent, suspect feature damage or erasure
- if patching late reader heads rescues behavior more than patching early writers, the feature likely survives but is not being used correctly

## Custom hook strategy for OOD studies

The built-in patching utilities are excellent for sweeps. After you localize the issue, switch to hand-written hooks for precise hypotheses.

Examples:
- patch only the final-token residual stream at one layer
- zero out only the contribution from the injected suffix positions
- patch only the source positions that correspond to the original arithmetic expression
- intervene only on specific heads that previously wrote or read the answer feature

## Minimal patching pattern

```python
from transformer_lens import patching

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits = model(corrupted_tokens)

patched_scores = patching.get_act_patch_resid_pre(
    model,
    corrupted_tokens,
    clean_cache,
    patching_metric=metric_fn,
)
```

## A recommended analysis grid

For each corruption type, compute:
- clean performance
- corrupted performance
- layerwise answer readability curve
- residual-pre patch map over layer × position
- attn-out patch map
- mlp-out patch map
- top rescuing heads

Then compare across corruption types:
- random rare-token suffixes
- syntactically broken but common-token suffixes
- matched ordinary suffixes
- repeated single-token junk suffixes

## Important architectural gotchas

- on modern Qwen-like architectures, key-value head structure may differ from query-head structure, so use official patching helpers rather than assuming one-to-one head semantics
- longer suffixes alter position encodings as well as content, so keep suffix length controlled in comparisons
- instruct-tuned models may react strongly to chat-template-like tokens; not every odd token is truly OOD with respect to the tokenizer distribution

## What counts as a strong result

A compelling mechanistic result looks like this:
- clean prompt shows a localized set of writer heads or MLPs that create the `" 2"` feature
- injected OOD suffix preserves early writing but later patch analysis shows one or two reader heads fail to recover the feature
- targeted restoration of those reader activations rescues the correct answer

That would let you say the computation was not simply erased; it became unreadable or inaccessible downstream.

## Common failure modes

- using only clean vs corrupted logits and calling that an explanation
- failing to match suffix length and position shifts in the control prompt
- mixing tokenization changes with OOD effects
- interpreting a high-rescue patch as proof that the patched site is the only causal locus

## Hand-off to the next skills

After localization:
- use the visualization skill to inspect the implicated heads interactively
- use the dataset-and-sparse-models skill to build a larger corpus of clean/corrupted examples and train probes for feature persistence or readability

## References

- TransformerLens `patching` docs describing `generic_activation_patch` and specialized patching helpers
- TransformerLens main demo section on activation patching and causal intervention
- TransformerLens release notes mentioning fixes for KV-head patching in architectures with mismatched head counts
