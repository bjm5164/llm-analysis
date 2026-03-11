# LLM Analysis

Mechanistic interpretability of LLM computations under out-of-distribution (OOD) token injection.
Built on [TransformerLens](https://github.com/neelnanda-io/TransformerLens) targeting Qwen3-4B on a single 24 GB GPU.

## What this does

The core question: **when an OOD token sequence is injected into a prompt, which components of the model's computation break, and which survive?**

A transformer processes a prompt by passing it through a series of layers. Each layer contains attention heads (which move information between token positions) and MLPs (which transform information at each position). The final output is a probability distribution over the vocabulary for the next token.

This tool lets you decompose that computation, compare it across prompt variants, and causally test which components matter — all through an interactive UI.

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

Run the app:

```bash
streamlit run src/AppConfig.py
```

---

## Key concepts

These concepts are used throughout the app. Understanding them makes each page more useful.

### Direct logit attribution (DLA)

A transformer's residual stream accumulates contributions from every component — the token embedding, positional embedding, each attention head, and each MLP. Because the final prediction is a linear function of the residual stream (LayerNorm + unembed matrix), you can project each component's contribution onto the direction for a specific answer token and measure how much it pushes the model toward or away from that answer.

The result is a per-component attribution score. Positive means the component pushes toward the answer; negative means it pushes away. These scores sum to the final answer logit, so you get a complete decomposition of *why* the model predicts what it does.

### Activation patching

DLA tells you *what* each component contributes, but not whether it's *necessary*. Activation patching answers this causally: run the model on a corrupted prompt, but at one specific (layer, position, component), swap in the activation from the clean prompt. If the correct answer recovers, that site is causally important for the computation the corruption disrupted.

### Logit lens

At each layer, the residual stream contains a partial computation. The logit lens projects this intermediate state through the final LayerNorm and unembedding matrix to see what the model would predict *if processing stopped at that layer*. This reveals when the answer token first emerges — early layers might not have it, and you can watch it appear and strengthen as layers progress.

---

## Pages

### Home — Configuration

The home page displays the current `config.yaml` and provides a live YAML editor. Changes are validated and propagated to all pages when you click Apply.

Key configuration options:
- **Model**: name, dtype, device, BOS handling, cache strategy
- **Corruption**: token types to inject, count, and random seed
- **Corruption sweep**: multiple named variants for batch comparison

### Model — Load and verify

Before any analysis, load the model here. The page shows architecture details (layers, d_model, heads, vocab size) and provides tools for verifying your setup:

- **Sanity check**: Enter any prompt, see the model's top-N next-token predictions with probabilities. Use this to confirm the model is working and to get intuition for what it predicts.
- **Answer token check**: Enter a token by string (e.g. `2` or `' 2'`) or by ID (e.g. `#220`). The app resolves it, confirms it maps to exactly one token, and shows the decoded result. This matters because tokenizers often have multiple tokens that look identical — `2` and ` 2` (with leading space) are different tokens with different IDs, and using the wrong one invalidates all downstream analysis.
- **Token lookup**: Search by ID to see what a token decodes to, or search by string to see all token IDs that produce it. Also automatically checks the leading-space variant, which is the most common source of token confusion.

### Prompts — Build and manage prompts

All analysis pages pull prompts from a shared prompt library. This page provides two ways to create them.

**Text Explorer** is the straightforward mode: type text, see it tokenized into chips (showing position index, token ID, and decoded string for each token), and save it with a label. You can also score the next token to see the model's top-K predictions. Use this for natural prompts where you don't need precise control over token boundaries.

**Token Builder** is for cases where token boundaries matter. When a tokenizer processes text, it merges characters into tokens using its learned vocabulary — `":"` might become a single token, but for your experiment you might need it split into `"`, `:`, `"` as three separate tokens. The token builder lets you construct prompts one token at a time with exact control over these boundaries.

Three ways to add tokens in the builder:
- **By text**: type a string, it gets tokenized naturally, and all resulting tokens are appended. Good for adding chunks of regular text.
- **By token search**: type a string and see what token ID(s) it maps to. If it maps to multiple tokens, you can add them individually or all at once. Handles escape sequences like `\n` and quoted strings with leading spaces like `' 2'`.
- **By token ID**: enter numeric IDs directly (e.g. `220` or `220,362,220`), useful when you know the exact token IDs you want.

The builder shows the current sequence as interactive chips, with undo, clear, and remove-at-position controls. Prompts built this way are saved with their exact token IDs so the tokenization is preserved when used in other pages.

**Saved prompts** appear at the bottom. Token-builder prompts show a badge indicating they carry token-level data. All prompts can be exported to JSON and imported back, which is useful for sharing prompt sets or resuming work.

### DLA — Direct logit attribution

Select one or two prompts from the library and an answer token. The page runs a forward pass on each prompt and decomposes the answer logit.

**Single-prompt mode** shows:
- **Logit margin**: the answer token's probability, raw logit, and rank among all vocabulary tokens.
- **Component attribution bar chart**: one bar per component (embed, pos_embed, each layer's attention block, each layer's MLP). The values sum to the final answer logit. Large positive bars are the components "writing" the answer; large negative bars are components pushing against it.
- **Head attribution heatmap**: a (layers x heads) grid where color intensity shows each attention head's contribution to the answer logit. This is the attention-head subset of the component attribution, broken out by individual head.
- **Top heads table**: heads ranked by absolute attribution, showing which specific heads contribute most.

**Comparison mode** (two prompts) adds:
- **Head attribution diff heatmap**: schema minus baseline. Red cells mean the head contributes *more* toward the answer under the modified prompt; blue means *less*. Large shifts indicate heads whose information routing changed.
- **Component diff charts**: raw difference and normalised difference (as fraction of baseline total attribution).
- **Top changed heads table**: heads ranked by how much their attribution shifted, with baseline and schema values side by side.

Set an optional **distractor token** to enable logit-difference metrics (logit(answer) - logit(distractor)), which is more stable than raw logit when comparing prompts that shift the overall logit scale.

### Attention — Where the model looks

Attention patterns show, for each head, which source tokens each destination token attends to. This page visualises those patterns using CircuitsVis interactive widgets.

**DLA-guided comparison** (when two prompts and an answer token are provided): heads are ranked by how much their DLA changed between prompts. The idea is that if a head's *contribution to the answer* shifted, its attention pattern likely changed too — so you want to look at those heads first. Select a head from the ranked list and see its attention pattern for both prompts side by side.

**Layer explorer**: browse all heads in a layer as a thumbnail grid (via CircuitsVis). Hover to preview, click to expand. Switch between baseline and schema prompts with a radio button. Good for surveying a full layer to spot unusual patterns.

**Position focus**: pick a specific (layer, head, destination position) and see a bar chart of where that position attends. The default destination is the last position (the one predicting the next token), which lets you check whether the answer-predicting position attends to schema tokens, content tokens, or structural delimiters like quotes and colons.

### Targeted Patching — Single-component interventions

This is the causal testing page. Instead of sweeping across all layers and positions, you pick one specific site and apply a surgical intervention to see what happens.

**Components you can target:**
- **resid_pre**: the full residual stream entering a layer. Intervening here affects everything downstream.
- **attn_head**: a single attention head's output (hook_z). Tests whether one specific head is necessary.
- **mlp_out**: a full MLP layer's output.
- **mlp_neuron**: a single neuron in the MLP's post-nonlinearity activation. The finest-grained intervention available.

**Intervention types:**
- **Zero ablation**: set the activation to zero. If the model breaks, the component was doing something important.
- **Mean ablation**: replace with the mean across sequence positions. Removes position-specific information while preserving the component's average contribution.
- **Noise injection**: add Gaussian noise (configurable std). Tests how robust the computation is to perturbation.
- **Activation patching**: replace with the activation from a source prompt at a chosen position. The classic causal intervention — tests whether this component carries the information that differs between the two prompts.

**Results shown:**
- **Top-k logits comparison**: grouped bar chart showing original vs patched next-token probabilities. Reveals whether the intervention changed the model's prediction and what it shifted to.
- **Logit lens at intervention layer**: what the residual stream "believes" at the intervention site, before and after. Shows how the intervention changed the intermediate computation.
- **Answer token tracking** (optional, requires answer token): the answer token's logit-lens score at every layer, original vs patched. Shows exactly where in the layer stack the intervention's effect propagates.
- **Residual stream L2 norm**: the magnitude of the residual stream at the target position across all layers, original vs patched. A sudden change in norm indicates where the intervention disrupted the computation's scale.
- **Attention patterns**: before vs after the intervention, at any layer/head you choose. Defaults to the intervention target. Shows whether the intervention changed where the model looks.

### Sweep — Corruption sweep

Tests how robust the model's answer is across multiple corruption variants simultaneously. Each variant injects a different combination of whitespace/newline tokens into the clean prompt (configured on the Home page).

The page runs DLA on each variant and compares against the clean baseline. Results are organised in three tabs:

- **P(answer) table**: probability of the answer token for clean and each variant. A quick overview of whether the model still gets the right answer.
- **Summary heatmap**: the top-K heads (ranked by clean attribution) as rows, corruption variants as columns. Cell color shows how much each head's attribution changed. Heads that shift consistently across variants are systematically disrupted by corruption; heads that only shift for some variants are placement-dependent.
- **Per-variant diffs**: select a variant to see its full head attribution diff heatmap, delta in P(answer), and a ranked table of the most-changed heads.

### Logit Lens — Track answer emergence

Runs the logit lens at every layer for a given prompt and answer token, then lets you iteratively generate tokens and re-run the lens after each step.

The initial run shows a table with the answer token's rank, probability, and logit at each layer. An expandable detail view for each step highlights which layer the answer token first enters the top-5. This reveals where in the network the answer "crystallises" — early layers often don't have it, and you can watch it appear.

**Generate + Re-check** appends the model's greedy next token to the sequence and re-runs the logit lens. This is useful for tracking how the answer direction evolves across multiple generation steps — does the model commit to the answer immediately, or does it take several tokens of generation before the answer direction stabilises?

The summary table across all steps shows whether the answer token is the model's top prediction at each step, making it easy to spot when the model "finds" the answer.

---

## Configuration

Everything is controlled by `src/config.yaml`. Key fields:

```yaml
model:
  name: Qwen/Qwen3-4B       # any HuggingFace model ID TransformerLens supports
  dtype: float16
  cache_strategy: full       # full | resid | attn_pattern | minimal

corruption:
  type: none
  inject: ["\n", " "]
  count: 3
  seed: 42

corruption_sweep:
  variants:
    - {label: "mixed_5_s1", inject: ["\n", " "], count: 5, seed: 1}
    - {label: "mixed_8_s1", inject: ["\n", " "], count: 8, seed: 1}
```

Prompts and answer tokens are managed interactively in the app rather than in the config file.

### Cache strategies

| Strategy | What is cached | Use case |
|---|---|---|
| `full` | everything | exploration, short prompts |
| `resid` | residual stream only | DLA without attention patterns |
| `attn_pattern` | attention patterns only | attention-only analysis |
| `minimal` | resid_pre + patterns | memory-constrained runs |

The Attention page requires `full` or `attn_pattern`. DLA only needs `resid`. Choose based on what you're doing and how much GPU memory you have.

---

## Project structure

```
src/
├── AppConfig.py              # Streamlit home page / config editor
├── config.py                 # Config dataclasses, YAML loading
├── config.yaml               # Single source of truth for settings
├── model.py                  # Model loading, tokenization, cache helpers
├── attribution.py            # Direct logit attribution (DLA)
├── patching.py               # Activation patching for causal analysis
├── visualization.py          # Matplotlib static plots
├── viz_interactive.py        # Plotly / CircuitsVis interactive charts
├── app_state.py              # Streamlit session state & caching
└── pages/
    ├── 0_Model.py
    ├── 1_Prompts.py
    ├── 2_DLA.py
    ├── 3_Attention.py
    ├── 4_Targeted_Patching.py
    ├── 5_Sweep.py
    └── 6_Logit_Lens.py
tests/
├── test_tokenization_edge_cases.py
└── test_token_normalization.py
```

---

## Memory notes

Qwen3-4B in float16 uses ~8 GB for weights. A full activation cache on a short prompt adds 3-7 GB depending on sequence length.

- Use `cache_strategy: resid` for DLA-only runs (skips caching attention patterns).
- The `head_out` patching sweep is O(n_layers x n_heads x n_positions) forward passes — expect several minutes on Qwen3-4B.
- Hard Reset in the app sidebar is the fastest way to fully reclaim GPU memory between sessions.
