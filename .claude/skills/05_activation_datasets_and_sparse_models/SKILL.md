---
name: transformerlens-activation-datasets-and-sparse-models
description: Build activation datasets with TransformerLens and train sparse linear probes or hand off to SAELens for sparse autoencoder training, feature dashboards, and larger-scale representation analysis.
---

# TransformerLens Activation Datasets and Sparse Models

Use this skill when the task shifts from single-example interpretability to dataset-scale representation analysis.

## What this skill is for

This skill covers:
- building datasets of prompts and activations
- extracting only the activations needed for a study
- labeling examples as clean, OOD-corrupted, rescued, or failed
- training sparse linear probes or other sparse predictive models on those activations
- deciding when to move from TransformerLens to SAELens for sparse autoencoder work

## Key point

TransformerLens has training utilities for autoregressive language models, but it is not the main training framework for sparse autoencoders. For sparse feature learning, SAELens is the more appropriate companion tool.

That does not reduce TransformerLens’ importance. TransformerLens is still the easiest way to:
- generate the activation corpus,
- define exact hook points,
- and run the mechanistic analyses that motivate what to model sparsely.

## Main TransformerLens tools

- `model.run_with_cache(...)`
- selective caching through `names_filter`
- `transformer_lens.utils.get_dataset(...)`
- `transformer_lens.evals` utilities for synthetic or diagnostic tasks
- `HookedTransformer.load_sample_training_dataset()` for simple workflows
- `transformer_lens.train` if you really do need to train or fine-tune a small autoregressive model in-library

## When to use SAELens

Switch to SAELens when you want to:
- train sparse autoencoders on activations
- load or analyze pretrained SAEs
- generate feature dashboards
- use its deeper integration path via `HookedSAETransformer`

The SAELens README explicitly presents these as first-class use cases and states that SAEs can integrate deeply with TransformerLens workflows.

## Dataset design for your project

### Example schema

For each prompt instance, store:
- `prompt_clean`
- `prompt_corrupted`
- `answer_token`
- `corruption_type`
- `suffix_length`
- `logit_diff_clean`
- `logit_diff_corrupted`
- `rescue_score` from your best patch
- extracted activations from one or more hook points
- optional labels like `feature_survives`, `feature_unreadable`, `feature_erased`

### Best hook points to start with

For sparse linear models, I would begin with:
- `resid_pre` at late layers
- `resid_mid` at layers implicated by patching
- `attn_out` for the top rescuing attention blocks
- head results for the most important heads

Do not start by dumping every tensor from every layer unless you absolutely need it.

## Training sparse linear probes

A very effective first pass is not an SAE but a sparse supervised probe.

Examples:
- L1-regularized logistic regression predicting whether the answer feature remains readable
- Lasso predicting answer logit difference from residual activations
- group-sparse probes over per-head features

Why this helps:
- it gives a fast, interpretable baseline
- it identifies a compact subset of dimensions or heads worth further study
- it tells you whether the distinction you care about is linearly accessible at all

## Suggested dataset pipeline

### Stage 1: generate examples

Construct many prompt pairs covering:
- arithmetic variants
- multiple OOD suffix families
- matched in-distribution suffix controls
- multiple answer tokens and distractors

### Stage 2: run targeted extraction

For each example, run either:
- full cache for a small pilot dataset
- or selective cache for a larger production dataset

### Stage 3: compute supervision labels

Possible targets:
- binary: correct answer survives or not
- scalar: answer logit difference
- ternary: erased vs readable-but-unused vs preserved

### Stage 4: train sparse models

Start simple:
- sklearn L1 models on CPU for prototyping
- torch sparse linear layers for larger datasets
- SAELens when moving to unsupervised sparse feature learning

## Where TransformerLens training utilities do and do not fit

The `transformer_lens.train` module provides utilities for training `HookedTransformer` models on autoregressive language modeling tasks. That is useful if you want to train a tiny bespoke model for a toy experiment, such as a controlled arithmetic or synthetic OOD-decoding environment.

It is not the best entry point for SAE training.

Use `transformer_lens.train` when you want:
- a toy model trained from scratch under controlled data
- reproducible checkpoints for a synthetic mechanistic study
- to study how OOD robustness emerges during model training

Use SAELens when you want:
- sparse autoencoder training
- feature dashboards
- pretrained SAE analysis

## A strong end-to-end plan for your research

### Phase A: clean mechanistic baseline

Use TransformerLens to identify where `" 2"` is written and read.

### Phase B: OOD causal localization

Use patching to classify failure modes for many corrupted prompts.

### Phase C: activation dataset

Extract activations at the implicated sites across the whole corpus.

### Phase D: sparse modeling

Train sparse probes first, then SAEs if the representation appears rich and compositional.

### Phase E: feature interpretation

Use sparse features to ask whether OOD suffixes:
- suppress arithmetic features,
- activate competing junk features,
- or selectively break reader circuits.

## Common failure modes

- jumping straight to SAE training before the mechanistic question is crisply defined
- storing massive raw activation dumps with no label schema
- using activations from too many hook points at once and losing interpretability
- confusing predictive probe performance with causal explanation

## References

- TransformerLens `train` docs for autoregressive training utilities
- TransformerLens `utils.get_dataset(...)` docs for convenient dataset access
- TransformerLens `evals` docs for task and dataset helpers
- SAELens README on training sparse autoencoders, analyzing pretrained SAEs, and generating feature dashboards
