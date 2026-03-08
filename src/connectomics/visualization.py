"""Visualization for mechanistic interpretability experiments.

Every figure should connect back to one answer-centric metric:
- answer logit / logit difference
- answer-direction readability
- causal rescue score

Prioritized visual objects (per the visualization skill):
1. Tokenization panel
2. Layerwise readability curve (clean vs corrupted overlay)
3. Per-head direct-logit-attribution heatmap
4. Patch rescue heatmaps (layer x position, layer x head)
5. Attention pattern viewers for top implicated heads

Uses matplotlib for script output. CircuitsVis and Plotly available
for notebook use via the circuitsviz_* helpers.
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer, ActivationCache


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------

def save_fig(fig, outdir: str | Path | None, filename: str):
    """Save figure to outdir (or cwd), then close."""
    if outdir:
        path = Path(outdir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path = Path(filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Tokenization panel
# ---------------------------------------------------------------------------

def plot_tokenization_panel(
    str_tokens: list[str],
    token_ids: torch.Tensor,
    suffix_start: int | None = None,
    title: str = "Tokenization",
    outdir: str | Path | None = None,
    filename: str = "tokenization_panel.png",
):
    """Visual tokenization table with optional suffix annotation.

    Highlights three regions when suffix_start is given:
    - task tokens (green)
    - suffix tokens (red)
    This prevents silent tokenizer mistakes.
    """
    n = len(str_tokens)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 1.5))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)

    for i, (stok, tid) in enumerate(zip(str_tokens, token_ids[0])):
        if suffix_start is not None and i >= suffix_start:
            color = "#ffcccc"
        elif i == 0 and "bos" in stok.lower() or "endoftext" in stok.lower():
            color = "#e0e0e0"
        else:
            color = "#ccffcc"
        ax.add_patch(plt.Rectangle((i - 0.45, -0.3), 0.9, 1.8,
                                   facecolor=color, edgecolor="gray", lw=0.5))
        ax.text(i, 0.9, repr(stok), ha="center", va="center", fontsize=7, rotation=30)
        ax.text(i, 0.2, str(tid.item()), ha="center", va="center", fontsize=6, color="gray")
        ax.text(i, -0.15, str(i), ha="center", va="center", fontsize=6, color="blue")

    plt.tight_layout()
    save_fig(fig, outdir, filename)


# ---------------------------------------------------------------------------
# 2. Layerwise readability curve (the highest-value plot)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3. Component attribution bar chart
# ---------------------------------------------------------------------------

def plot_component_attribution(
    attrs: torch.Tensor,
    labels: list[str],
    answer_label: str,
    outdir: str | Path | None = None,
    filename: str = "component_attribution.png",
):
    """Bar chart of per-component direct logit attribution."""
    data = attrs.detach().cpu().float().numpy()
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.4), 4))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in data]
    ax.bar(range(len(data)), data, color=colors, alpha=0.7)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, fontsize=6, rotation=60, ha="right")
    ax.set_ylabel(f"Attribution toward {answer_label}")
    ax.set_title(f"Component attribution: {answer_label}")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_fig(fig, outdir, filename)


# ---------------------------------------------------------------------------
# 5. Per-head attribution heatmap (layer x head)
# ---------------------------------------------------------------------------

def plot_head_attribution(
    attrs: torch.Tensor,
    answer_label: str,
    outdir: str | Path | None = None,
    filename: str = "head_attribution.png",
):
    """Heatmap of per-head direct logit attribution.

    Answers: which heads help the answer? Which become negative after OOD?
    """
    data = attrs.detach().cpu().float().numpy()
    n_layers, n_heads = data.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.5), max(4, n_layers * 0.35)))
    vmax = max(abs(data.min()), abs(data.max()), 1e-6)
    im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Per-head attribution: {answer_label}")
    plt.colorbar(im, ax=ax, label="attribution")
    plt.tight_layout()
    save_fig(fig, outdir, filename)


# ---------------------------------------------------------------------------
# 6. Patch rescue heatmaps
# ---------------------------------------------------------------------------

def plot_patch_heatmap(
    scores: torch.Tensor,
    str_tokens: list[str],
    title: str,
    outdir: str | Path | None = None,
    filename: str = "patch_heatmap.png",
):
    """Heatmap of patching rescue scores over (layer x position).

    The quickest way to localize where clean activations rescue
    the corrupted run's answer.
    """
    data = scores.detach().cpu().float().numpy()
    n_layers, seq_len = data.shape
    fig, ax = plt.subplots(figsize=(max(6, seq_len * 0.8), max(4, n_layers * 0.3)))
    im = ax.imshow(data, aspect="auto", cmap="RdBu")
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels([repr(t) for t in str_tokens[:seq_len]],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="metric after patching")
    plt.tight_layout()
    save_fig(fig, outdir, filename)


def plot_head_patch_heatmap(
    scores: torch.Tensor,
    title: str,
    outdir: str | Path | None = None,
    filename: str = "head_patch_heatmap.png",
):
    """Heatmap of per-head patching rescue scores (layer x head).

    Summed across positions to show which heads matter most.
    """
    # scores is (n_layers, n_heads, seq_len) — sum over positions
    data = scores.sum(dim=-1).detach().cpu().float().numpy()
    n_layers, n_heads = data.shape
    fig, ax = plt.subplots(figsize=(max(6, n_heads * 0.5), max(4, n_layers * 0.35)))
    im = ax.imshow(data, aspect="auto", cmap="RdBu")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="rescue score (summed over pos)")
    plt.tight_layout()
    save_fig(fig, outdir, filename)


# ---------------------------------------------------------------------------
# 7. Attention pattern viewer
# ---------------------------------------------------------------------------

def plot_attention_pattern(
    cache: ActivationCache,
    layer: int,
    head: int,
    str_tokens: list[str],
    title: str | None = None,
    outdir: str | Path | None = None,
    filename: str | None = None,
):
    """Plot attention pattern for a single (layer, head).

    In OOD studies, look for whether late heads stop attending to
    arithmetic tokens and get distracted by suffix tokens.
    """
    # cache["pattern", layer] shape: (batch, head, dest_pos, src_pos)
    pattern = cache["pattern", layer][0, head].detach().cpu().float().numpy()
    seq_len = pattern.shape[0]
    tok_labels = [repr(t) for t in str_tokens[:seq_len]]

    fig, ax = plt.subplots(figsize=(max(4, seq_len * 0.6), max(4, seq_len * 0.6)))
    im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tok_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tok_labels, fontsize=7)
    ax.set_xlabel("Source (key)")
    ax.set_ylabel("Destination (query)")
    if title is None:
        title = f"L{layer} H{head} attention"
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, label="attention weight")
    plt.tight_layout()

    if filename is None:
        filename = f"attention_L{layer}_H{head}.png"
    save_fig(fig, outdir, filename)


def plot_top_head_attention(
    model: HookedTransformer,
    cache: ActivationCache,
    head_attrs: torch.Tensor,
    str_tokens: list[str],
    k: int = 4,
    outdir: str | Path | None = None,
    filename_prefix: str = "top_head_attn",
):
    """Plot attention patterns for the top-k heads by attribution.

    Saves one file per head. Focus on the top few rather than plotting all.
    """
    n_heads = head_attrs.shape[1]
    flat = head_attrs.flatten()
    top_idx = flat.abs().argsort(descending=True)[:k]

    for rank, idx in enumerate(top_idx):
        li = idx.item() // n_heads
        hi = idx.item() % n_heads
        attr_val = flat[idx].item()
        plot_attention_pattern(
            cache, li, hi, str_tokens,
            title=f"L{li} H{hi} (attr={attr_val:+.3f})",
            outdir=outdir,
            filename=f"{filename_prefix}_{rank}_L{li}_H{hi}.png",
        )


# ---------------------------------------------------------------------------
# 8. Corruption sweep summary
# ---------------------------------------------------------------------------

def plot_sweep_summary(
    variant_labels: list[str],
    diffs: list[torch.Tensor],
    clean_attrs: torch.Tensor,
    answer_label: str,
    k: int = 12,
    outdir: str | Path | None = None,
    filename: str = "sweep_summary.png",
):
    """Heatmap: attribution change (vs clean) for top-k heads across all variants.

    Rows = corruption variants, columns = top-k heads by clean attribution magnitude.
    Shows which heads are consistently disrupted vs. placement-dependent.

    Args:
        variant_labels: names for each row.
        diffs:          list of (n_layers, n_heads) tensors (variant - clean).
        clean_attrs:    (n_layers, n_heads) clean attribution tensor for ranking heads.
        k:              number of top heads to show.
    """
    n_heads = clean_attrs.shape[1]
    flat_clean = clean_attrs.flatten()
    top_idx = flat_clean.abs().argsort(descending=True)[:k]
    head_labels = [
        f"L{idx.item() // n_heads} H{idx.item() % n_heads}" for idx in top_idx
    ]

    matrix = np.stack([
        diff.flatten()[top_idx].detach().cpu().float().numpy()
        for diff in diffs
    ])  # (n_variants, k)

    vmax = max(abs(matrix.min()), abs(matrix.max()), 1e-6)
    fig, ax = plt.subplots(figsize=(max(8, k * 0.7), max(3, len(variant_labels) * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(k))
    ax.set_xticklabels(head_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(variant_labels)))
    ax.set_yticklabels(variant_labels, fontsize=8)
    ax.set_title(f"Attribution change vs clean — top {k} heads: {answer_label}")
    plt.colorbar(im, ax=ax, label="attribution diff")
    plt.tight_layout()
    save_fig(fig, outdir, filename)
