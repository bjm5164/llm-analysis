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
    tok_labels = [f"{i}:{repr(t)}" for i, t in enumerate(str_tokens[:seq_len])]
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
    """Plotly heatmap for a single (layer, head). Kept for non-interactive use."""
    pattern = (
        cache["pattern", layer][0, head].detach().cpu().float().numpy()
    )
    seq_q, seq_k = pattern.shape
    tok_labels_k = [
        f"{i}:{repr(str_tokens[i])}" if i < len(str_tokens) else str(i)
        for i in range(seq_k)
    ]
    tok_labels_q = [
        f"{i}:{repr(str_tokens[i])}" if i < len(str_tokens) else str(i)
        for i in range(seq_q)
    ]

    cell_px = max(22, min(40, 800 // max(seq_q, 1)))
    fig_h = cell_px * seq_q + 180

    fig = px.imshow(
        pattern,
        x=tok_labels_k,
        y=tok_labels_q,
        color_continuous_scale="Blues",
        zmin=0,
        zmax=1,
        title=title or f"L{layer} H{head} Attention Pattern",
        labels={
            "x": "Source (key)",
            "y": "Destination (query)",
            "color": "Attention",
        },
        aspect="auto",
    )
    fig.update_xaxes(
        tickangle=-45,
        tickmode="array",
        tickvals=list(range(seq_k)),
        ticktext=tok_labels_k,
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(seq_q)),
        ticktext=tok_labels_q,
        tickfont=dict(size=9),
    )
    fig.update_layout(height=fig_h, margin=dict(b=120, l=120))
    return fig


# ------------------------------------------------------------------
# CircuitsVis-based interactive attention visualizations
# ------------------------------------------------------------------

# Script injected after every CircuitsVis widget to auto-resize the
# Streamlit iframe to match the rendered content height.  Uses a
# ResizeObserver so the iframe grows/shrinks when the user clicks a
# thumbnail to expand or collapse a head.
_RESIZE_SCRIPT = """
<script>
(function() {
  function resize() {
    var h = document.body.scrollHeight;
    if (window.frameElement) {
      window.frameElement.style.height = h + "px";
    }
  }
  // Initial resize after CircuitsVis renders (async module load).
  setTimeout(resize, 500);
  setTimeout(resize, 1500);
  // Continuous resize on DOM changes (expand/collapse).
  new ResizeObserver(resize).observe(document.body);
})();
</script>
"""


def _wrap_cv(raw_html: str) -> str:
    """Wrap CircuitsVis HTML with auto-resize script."""
    return raw_html + _RESIZE_SCRIPT


def attention_heads_cv(
    cache: ActivationCache,
    layer: int,
    str_tokens: list[str],
    mask_upper_tri: bool = True,
) -> str:
    """CircuitsVis interactive multi-head attention widget.

    Returns an HTML string for embedding via st.components.v1.html().
    Click a thumbnail to expand the full pattern with hover values.
    """
    from circuitsvis.attention import attention_heads

    # attention shape expected: [n_heads, dest, src]
    patterns = (
        cache["pattern", layer][0].detach().cpu().float()
    )
    html_obj = attention_heads(
        attention=patterns,
        tokens=str_tokens,
        mask_upper_tri=mask_upper_tri,
    )
    return _wrap_cv(str(html_obj))


def attention_single_cv(
    cache: ActivationCache,
    layer: int,
    head: int,
    str_tokens: list[str],
    mask_upper_tri: bool = True,
    max_width: int | None = None,
    positive_color: str | None = None,
) -> str:
    """CircuitsVis single-head attention pattern.

    Returns an HTML string.  *max_width* (px) constrains the widget width.
    """
    from circuitsvis.attention import attention_pattern

    pattern = cache["pattern", layer][0, head].detach().cpu().float()
    kwargs: dict = dict(
        tokens=str_tokens,
        attention=pattern,
        mask_upper_tri=mask_upper_tri,
    )
    if positive_color is not None:
        kwargs["positive_color"] = positive_color
    html_obj = attention_pattern(**kwargs)
    html = _wrap_cv(str(html_obj))
    if max_width is not None:
        html = (
            f'<div style="max-width:{max_width}px;margin:0 auto">'
            + html
            + "</div>"
        )
    return html


def attention_source_row(
    attn_weights: "np.ndarray",
    str_tokens: list[str],
    dst_label: str = "",
    title: str | None = None,
) -> go.Figure:
    """Token sequence colored by attention weight from a single destination token.

    Shows a horizontal bar: [the] [cat] [crossed] [the] [road]
    each box colored by how much the destination token attends to it.
    """
    n_toks = len(attn_weights)
    tok_labels = [repr(t) for t in str_tokens[:n_toks]]
    fig = px.imshow(
        attn_weights[np.newaxis, :],
        x=tok_labels,
        y=[dst_label or "attn"],
        color_continuous_scale="Blues",
        zmin=0,
        zmax=float(attn_weights.max()) or 1.0,
        text_auto=".2f",
        labels={"color": "Attention"},
        aspect="auto",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_toks)),
        ticktext=tok_labels,
        tickangle=-45,
        tickfont=dict(size=9),
    )
    fig.update_layout(
        title=title or f"Attention from {dst_label}",
        height=160,
        margin=dict(t=40, b=80, l=60, r=20),
    )
    fig.update_yaxes(showticklabels=True, tickfont_size=10)
    fig.update_traces(textfont_size=10)
    return fig


def topk_logits_comparison(
    model,
    orig_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    pos: int = -1,
    k: int = 10,
    title: str = "Top-k final logits: original vs patched",
    rank_by: str = "original",
) -> go.Figure:
    """Grouped bar chart of top-k tokens, before and after intervention.

    Args:
        rank_by: Which logits determine the top-k ranking.
            "original" ranks by pre-intervention logits,
            "patched" ranks by post-intervention logits.
    """
    orig = orig_logits[0, pos, :].detach().cpu().float()
    patched = patched_logits[0, pos, :].detach().cpu().float()

    ranking = orig if rank_by == "original" else patched
    top_ids = ranking.topk(k).indices
    labels = [
        f"{i.item()}:{repr(model.tokenizer.decode([i.item()]))}"
        for i in top_ids
    ]

    fig = go.Figure([
        go.Bar(
            name="Original", x=labels,
            y=orig[top_ids].tolist(), marker_color="#3498db",
        ),
        go.Bar(
            name="Patched", x=labels,
            y=patched[top_ids].tolist(), marker_color="#e74c3c",
        ),
    ])
    fig.update_layout(
        barmode="group",
        title=title,
        xaxis_title="Token",
        xaxis_type="category",
        yaxis_title="Logit",
        xaxis_tickangle=-40,
        height=420,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60, b=80),
    )
    return fig


def logit_lens_at_layer(
    model,
    orig_cache,
    patched_cache,
    layer: int,
    pos: int,
    k: int = 10,
    title: str | None = None,
) -> go.Figure:
    """Grouped bar chart: top-k tokens from logit lens at resid_post of given layer."""

    def _lens(cache):
        resid = cache["resid_post", layer][:, pos : pos + 1, :]  # [1,1,d]
        with torch.no_grad():
            ln = model.ln_final(resid)              # [1,1,d]
            logits = model.unembed(ln)[0, 0]        # [vocab]
        return logits.detach().cpu().float()

    orig_l   = _lens(orig_cache)
    patched_l = _lens(patched_cache)

    top_ids = orig_l.topk(k).indices
    labels  = [
        repr(model.tokenizer.decode([i.item()])) or f"[{i.item()}]"
        for i in top_ids
    ]
    ttl = title or f"Logit lens at L{layer} resid_post — pos {pos}"

    fig = go.Figure([
        go.Bar(name="Original", x=labels, y=orig_l[top_ids].tolist(),   marker_color="#3498db"),
        go.Bar(name="Patched",  x=labels, y=patched_l[top_ids].tolist(), marker_color="#e74c3c"),
    ])
    fig.update_layout(
        barmode="group",
        title=ttl,
        xaxis_title="Token",
        xaxis_type="category",
        yaxis_title="Logit (logit lens)",
        xaxis_tickangle=-40,
        height=380,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60),
    )
    return fig


def answer_logit_across_layers(
    model,
    orig_cache,
    patched_cache,
    pos: int,
    answer_token_id: int,
    title: str = "Answer logit across layers (logit lens)",
) -> go.Figure:
    """Line chart: logit-lens score for the answer token at every layer, before vs after."""
    n_layers = model.cfg.n_layers
    orig_scores    = []
    patched_scores = []

    for layer in range(n_layers):
        for scores, cache in [(orig_scores, orig_cache), (patched_scores, patched_cache)]:
            resid = cache["resid_post", layer][:, pos : pos + 1, :]
            with torch.no_grad():
                ln     = model.ln_final(resid)
                logits = model.unembed(ln)[0, 0]
            scores.append(logits[answer_token_id].item())

    layers = list(range(n_layers))
    fig = go.Figure([
        go.Scatter(x=layers, y=orig_scores,    name="Original", mode="lines+markers",
                   line=dict(color="#3498db", width=2)),
        go.Scatter(x=layers, y=patched_scores, name="Patched",  mode="lines+markers",
                   line=dict(color="#e74c3c", width=2)),
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Logit (logit lens)",
        height=340,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60),
    )
    return fig


def residual_norm_across_layers(
    model,
    orig_cache,
    patched_cache,
    pos: int,
    title: str = "Residual stream L2 norm across layers",
) -> go.Figure:
    """Line chart: ||resid_post||₂ at each layer for the target position."""
    n_layers = model.cfg.n_layers
    orig_norms    = []
    patched_norms = []

    for layer in range(n_layers):
        o = orig_cache["resid_post", layer][0, pos, :].detach().cpu().float()
        p = patched_cache["resid_post", layer][0, pos, :].detach().cpu().float()
        orig_norms.append(o.norm().item())
        patched_norms.append(p.norm().item())

    layers = list(range(n_layers))
    fig = go.Figure([
        go.Scatter(x=layers, y=orig_norms,    name="Original", mode="lines+markers",
                   line=dict(color="#3498db", width=2)),
        go.Scatter(x=layers, y=patched_norms, name="Patched",  mode="lines+markers",
                   line=dict(color="#e74c3c", width=2)),
    ])
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="‖resid_post‖₂",
        height=320,
        legend=dict(orientation="h", y=1.02, x=0),
        margin=dict(t=60),
    )
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
