---
name: transformerlens-interactive-visualization-for-qwen
description: Build interactive visual analysis workflows for Qwen-family models in TransformerLens using CircuitsVis, Plotly, residual-stream heatmaps, and token-level dashboards.
---

# TransformerLens Interactive Visualization for Qwen

Use this skill when the task is to inspect a Qwen-family model interactively instead of only producing static summary numbers.

## What this skill is for

This skill helps with:
- attention visualization by layer and head
- token-by-token dashboards for short prompts
- residual-stream attribution heatmaps
- patching heatmaps over layers and positions
- side-by-side clean vs OOD-corrupted prompt comparisons
- exploratory notebooks where you can click through hypotheses quickly

This is especially important for your project because OOD-token failures are often easiest to understand visually before they can be formalized.

## Main tools

- TransformerLens `ActivationCache`
- `cache["pattern", layer]` style attention access
- `cache.stack_head_results(...)`
- Plotly heatmaps / line charts
- CircuitsVis attention viewers
- notebook widgets or lightweight dashboard wrappers

The official exploratory-analysis demo explicitly uses CircuitsVis plus Plotly-style visual workflows for head-level interpretability.

## Recommended visual objects

### 1. Tokenization panel

Always begin each notebook with a tokenization display:
- token index
- token string
- token id
- whether it belongs to the clean prompt or injected suffix

This prevents silent tokenizer mistakes.

### 2. Attention-pattern viewer

For a selected layer or top-k heads:
- show destination-position × source-position attention maps
- label axes with `model.to_str_tokens(...)`
- add a clean/corrupted toggle

In OOD studies, pay special attention to whether late heads stop attending to the original arithmetic tokens and instead get distracted by suffix tokens.

### 3. Layerwise readability curve

Plot answer-direction readability across layers using accumulated residual projections. Overlay clean and corrupted curves.

This is one of the highest-value plots in the whole workflow.

### 4. Per-head direct-logit-attribution heatmap

Compute per-head contributions and display a layer × head heatmap.

Useful questions:
- which heads help `" 2"` in the clean run?
- which heads become negative or inactive after OOD injection?
- which heads should be targeted for patching or ablation?

### 5. Patching rescue heatmaps

Display patch scores over:
- layer × position for `resid_pre`
- layer × head for head-output patching

These often provide the quickest causal localization.

## Qwen-specific notebook recommendations

### Keep the first notebook small

Use a small Qwen-family checkpoint with short prompts and aggressive selective caching. A responsive notebook is worth more than a perfectly faithful but sluggish first setup.

### Label suffix positions explicitly

When studying injected tokens, the most useful visual convention is to color or annotate three regions:
- task tokens
n- separator or transition tokens
- injected OOD suffix tokens

That makes it much easier to see when attention shifts from the original arithmetic expression to junk suffix positions.

### Use consistent answer-centric views

Every dashboard should make it easy to answer one question:
"What changed with respect to the model’s ability to produce the target answer token?"

So each figure should connect back to one of:
- answer logit
- answer logit difference
- answer-direction readability
- causal rescue score

## Practical notebook architecture

### Notebook 1: clean prompt anatomy

Sections:
- model load
- tokenization display
- final logits table
- layerwise readability plot
- per-sublayer attribution chart
- per-head attribution heatmap
- top-head attention viewers

### Notebook 2: OOD injection comparison

Sections:
- clean vs corrupted token panels
- logits comparison
- readability overlay
- attention comparison for top implicated heads
- patching rescue maps
- summary cells of main hypotheses

### Notebook 3: hypothesis-driven interventions

Sections:
- targeted hooks
- ablations
- reader-vs-writer tests
- clean/corrupted/partially-restored comparisons

## Example visualization snippets

```python
str_tokens = model.to_str_tokens(tokens)
pattern = cache["pattern", layer]  # [head, dest_pos, src_pos]
```

```python
per_head_resid, labels = cache.stack_head_results(
    layer=-1,
    pos_slice=-1,
    return_labels=True,
)
```

Use these tensors to drive Plotly heatmaps or CircuitsVis renderers.

## What to visualize first for your exact problem

For `"what is 1 + 1"` plus injected suffixes, I would prioritize:
1. tokenization table
2. answer-logit comparison
3. accumulated-residual clean vs corrupted overlay
4. per-head attribution heatmap
5. rescue heatmap from `resid_pre` patching
6. attention maps for the top rescuing heads

That sequence usually narrows the search space very quickly.

## Common failure modes

- using unlabeled token axes, making attention maps impossible to read
- plotting every head at once instead of ranking and focusing on the top few
- building dashboards that are visually rich but not tied to an answer-centric metric
- mixing clean and corrupted runs with inconsistent prompt formatting

## Hand-off to the next skill

Once your dashboard identifies recurrent vulnerable positions or components, generate a larger activation dataset and train sparse probes or SAEs to characterize the feature family more systematically.

## References

- TransformerLens exploratory-analysis demo using CircuitsVis and head-level attribution
- TransformerLens main demo section on caching and attention visualization
- TransformerLens `ActivationCache` helpers for stacking head results and residual decompositions
