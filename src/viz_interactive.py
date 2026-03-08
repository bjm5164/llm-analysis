"""Interactive (Plotly) visualizations for the Streamlit app.

Mirrors visualization.py but returns go.Figure objects instead of saving PNGs.
Every function follows the same data contract as its matplotlib counterpart.
"""
from __future__ import annotations

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from transformer_lens import ActivationCache


def head_attribution_heatmap(
    attrs: torch.Tensor,
    title: str = "Per-head attribution",
) -> go.Figure:
    """Interactive heatmap of per-head DLA (n_layers x n_heads)."""
    data = attrs.detach().cpu().float().numpy()
    n_layers, n_heads = data.shape
    vmax = max(float(np.abs(data).max()), 1e-6)
    fig = px.imshow(
        data,
        color_continuous_scale="RdBu",
        zmin=-vmax,
        zmax=vmax,
        labels={"x": "Head", "y": "Layer", "color": "Attribution"},
        title=title,
        aspect="auto",
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=max(1, n_heads // 16)),
        yaxis=dict(tickmode="linear", dtick=max(1, n_layers // 20)),
        coloraxis_colorbar_title="attribution",
    )
    return fig


def component_attribution_bar(
    attrs: torch.Tensor,
    labels: list[str],
    answer_label: str,
) -> go.Figure:
    """Bar chart of coarse per-component DLA."""
    data = attrs.detach().cpu().float().numpy()
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in data]
    fig = go.Figure(go.Bar(
        x=labels,
        y=data.tolist(),
        marker_color=colors,
        opacity=0.8,
    ))
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        title=f"Component attribution toward {answer_label}",
        xaxis_title="Component",
        yaxis_title="Attribution",
        xaxis_tickangle=-60,
        height=420,
    )
    return fig


def sweep_summary_heatmap(
    variant_labels: list[str],
    diffs: list[torch.Tensor],
    clean_attrs: torch.Tensor,
    answer_label: str,
    k: int = 12,
) -> go.Figure:
    """Attribution-change heatmap: rows = variants, columns = top-k heads."""
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
    vmax = max(float(np.abs(matrix).max()), 1e-6)
    fig = px.imshow(
        matrix,
        x=head_labels,
        y=variant_labels,
        color_continuous_scale="RdBu",
        zmin=-vmax,
        zmax=vmax,
        title=f"Attribution change vs clean — top {k} heads: {answer_label}",
        labels={"color": "attr diff"},
        aspect="auto",
    )
    return fig


def patch_heatmap(
    scores: torch.Tensor,
    str_tokens: list[str],
    title: str,
) -> go.Figure:
    """Patching rescue score heatmap (layer x position)."""
    data = scores.detach().cpu().float().numpy()
    seq_len = data.shape[1]
    tok_labels = [repr(t) for t in str_tokens[:seq_len]]
    vmax = float(np.abs(data).max()) or 1e-6
    fig = px.imshow(
        data,
        x=tok_labels,
        color_continuous_scale="RdBu",
        zmin=-vmax,
        zmax=vmax,
        title=title,
        labels={"x": "Token", "y": "Layer", "color": "Rescue score"},
        aspect="auto",
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def head_patch_heatmap(
    scores: torch.Tensor,
    title: str,
) -> go.Figure:
    """Per-head patching rescue heatmap (summed over positions)."""
    data = scores.sum(dim=-1).detach().cpu().float().numpy()
    n_layers, n_heads = data.shape
    vmax = float(np.abs(data).max()) or 1e-6
    fig = px.imshow(
        data,
        color_continuous_scale="RdBu",
        zmin=-vmax,
        zmax=vmax,
        title=title,
        labels={"x": "Head", "y": "Layer", "color": "Rescue score"},
        aspect="auto",
    )
    fig.update_layout(
        xaxis=dict(tickmode="linear", dtick=max(1, n_heads // 16)),
        yaxis=dict(tickmode="linear", dtick=max(1, n_layers // 20)),
    )
    return fig


def attention_pattern_heatmap(
    cache: ActivationCache,
    layer: int,
    head: int,
    str_tokens: list[str],
    title: str | None = None,
) -> go.Figure:
    """Interactive attention pattern for a single (layer, head)."""
    pattern = cache["pattern", layer][0, head].detach().cpu().float().numpy()
    seq_len = pattern.shape[0]
    tok_labels = [repr(t) for t in str_tokens[:seq_len]]
    fig = px.imshow(
        pattern,
        x=tok_labels,
        y=tok_labels,
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1,
        title=title or f"L{layer} H{head} attention",
        labels={"x": "Source (key)", "y": "Destination (query)", "color": "weight"},
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def tokenization_table(
    str_tokens: list[str],
    token_ids: torch.Tensor,
) -> go.Figure:
    """Table view of tokenization: index, token ID, string repr."""
    ids_list = token_ids[0].tolist()
    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Index</b>", "<b>Token ID</b>", "<b>Token</b>"],
            fill_color="#f0f2f6",
            align="center",
        ),
        cells=dict(
            values=[
                list(range(len(str_tokens))),
                ids_list,
                [repr(t) for t in str_tokens],
            ],
            align="center",
            height=28,
        ),
    ))
    fig.update_layout(
        height=min(600, max(200, len(str_tokens) * 32 + 80)),
        margin=dict(t=20, b=0, l=0, r=0),
    )
    return fig
