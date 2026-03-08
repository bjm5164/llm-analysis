---
name: transformerlens-answer-tracing-and-logit-attribution
description: Trace where a model writes answer-relevant features into the residual stream and which components push the logits toward a target token such as " 2".
---

# TransformerLens Answer Tracing and Logit Attribution

Use this skill when the task is to explain why a model predicts a target token, such as why `"what is 1 + 1"` points toward `" 2"`.

## What this skill is for

This skill answers questions like:
- which layers and positions make `" 2"` more likely?
- which heads or MLPs write the decisive features?
- when does the answer become linearly readable in the residual stream?
- does the answer exist early and get refined, or only appear late?

For your project, this is the baseline clean-prompt analysis that must be done before introducing OOD tokens.

## Main TransformerLens tools

- `model.run_with_cache(...)`
- `ActivationCache.accumulated_resid(...)`
- `ActivationCache.decompose_resid(...)`
- `ActivationCache.get_full_resid_decomposition(...)`
- `ActivationCache.logit_attrs(...)`
- `model.tokens_to_residual_directions(...)`
- `cache.stack_head_results(...)`
- `cache.apply_ln_to_stack(...)`

## Conceptual framing

The central question is not just "what token is predicted," but "what vectors inside the residual stream are pointing in the answer direction, and which components wrote them?"

TransformerLens is especially strong here because it gives you:
- the full cached residual stream
- decompositions by block or by finer component
- direct logit attribution utilities
- logit-lens style readouts at many layers

## Recommended workflow for `what is 1 + 1 -> 2`

### 1. Pin down the answer token direction

First identify whether the answer is a single token under your tokenizer. Then get the residual direction associated with that token.

This is your target readout direction.

### 2. Measure the final logit margin

Before decomposing anything, record:
- the answer token logit
- top competing logits
- the logit difference between `" 2"` and distractors such as `" 1"`, `" 3"`, or other arithmetic completions

This gives a stable target metric for later comparisons.

### 3. Run accumulated-residual analysis

Use `accumulated_resid(..., apply_ln=True)` at the final prediction position to ask: after each layer, how much of the answer direction is already present?

This is the cleanest way to see when the answer becomes readable.

Interpretation:
- early rise means the model forms the answer direction quickly
- late rise means the model delays committing until later blocks
- oscillation means some layers add and others partially erase or rotate the signal

### 4. Run direct logit attribution over components

Use `decompose_resid(...)` first for coarse attribution:
- embed
- pos_embed
- each attention block output
- each MLP output

Then use `get_full_resid_decomposition()` when you want more granular breakdowns into individual heads or MLP-neuron-level components.

This tells you which components actually write toward the answer logit.

### 5. Drill down to per-head analysis

If a layer looks important, compute head results and attribute them individually. In many tasks, only a few heads carry the decisive evidence.

Good questions:
- which head most strongly pushes toward `" 2"`?
- does that head write at the final position, or route information from earlier positions?
- are there both positive and negative contributor heads?

### 6. Inspect position-specific writing

For arithmetic and short prompts, position matters a lot. The key question is whether the answer-relevant feature is written:
- at the final token position directly,
- copied from an earlier operand token,
- or assembled gradually across positions.

A useful habit is to inspect residual decompositions at every position, not just the last one.

## Minimal code skeleton

```python
import torch
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("qwen2.5-1.5b")
prompt = "what is 1 + 1"
answer = " 2"

tokens = model.to_tokens(prompt)
logits, cache = model.run_with_cache(tokens)

answer_dir = model.tokens_to_residual_directions(model.to_tokens(answer, prepend_bos=False))
accum, labels = cache.accumulated_resid(apply_ln=True, return_labels=True)
resid_stack, comp_labels = cache.decompose_resid(return_labels=True)
logit_attrs = cache.logit_attrs(resid_stack, answer)
```

## How to interpret the outputs

### `accumulated_resid`
Use this to answer: when is the answer readable?

### `decompose_resid`
Use this to answer: which blocks wrote the answer-relevant direction?

### `get_full_resid_decomposition`
Use this to answer: which exact heads or finer units wrote the signal?

### `logit_attrs`
Use this to answer: which components positively or negatively contribute to the target token’s logit?

## What to save for later OOD analysis

For each clean prompt, save:
- answer token(s)
- final-position residual stream
- layerwise accumulated readability curve
- component-wise attribution scores
- per-head positive and negative contributors
- the top few heads and MLPs you suspect encode the arithmetic feature

These become your "clean reference features" for later robustness tests.

## Good sanity checks

- replace `1 + 1` with `1 + 2` and confirm the attribution pattern changes meaningfully
- swap wording while preserving semantics and see whether the same heads are reused
- compare base vs instruct Qwen models to test whether chat tuning changes where arithmetic evidence is written

## Common failure modes

- forgetting that direct logit attribution is only meaningful after the correct normalization/readout treatment
- over-interpreting a coarse sublayer contribution without drilling down to heads or neurons
- treating a readable feature as necessarily causal; readability and causal necessity are different
- ignoring negative contributors that suppress distractor answers

## Best next step

After you know where the answer signal is written in the clean prompt, move to activation patching under OOD token injection. That is the right way to test whether those answer-relevant features remain present, get damaged, or become unreadable.

## References

- TransformerLens `ActivationCache` docs and examples for `decompose_resid`, `accumulated_resid`, and `logit_attrs`
- TransformerLens exploratory-analysis demo for per-head residual attribution and attention analysis
- TransformerLens `HookedTransformer` docs for token and residual-direction helpers
