"""Attention Maps.

How does structured schema enforcement change where the model looks?

Workflow:
  1. Run Prompt A (baseline) and Prompt B (with schema / OOD suffix).
  2. Compare attention patterns head-by-head and layer-by-layer.
  3. Look for heads that redirect attention toward (or away from) schema
     tokens, answer-relevant positions, or structural delimiters.
"""

import streamlit as st
import streamlit.components.v1 as components

from app_state import (
    get_config, get_model, render_sidebar_memory,
    prompt_selector, prompt_tokenize, token_id_input,
)
from attribution import head_attribution
from model import run_with_cache
from ov_circuits import (
    head_activity,
    ov_eigenvalues_single,
    copying_score,
    composition_from_act,
    composition_to_act,
    trace_circuit_act,
)
from viz_interactive import (
    activity_heatmap,
    attention_heads_cv,
    attention_single_cv,
    attention_source_row,
    eigenvalue_spectrum,
    circuit_graph,
)

st.set_page_config(page_title="Attention — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Attention Maps")
st.caption(
    "Where does the model look, and how does schema enforcement redirect attention flow?"
)

cfg = get_config()

# ---------------------------------------------------------------------------
# Prompt inputs (from saved library)
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2, gap="large")
with col_a:
    prompt_a = prompt_selector(
        "attn_prompt_a", label="Prompt A (baseline)",
        allow_empty=False, sync_slot="a",
    )
with col_b:
    prompt_b = prompt_selector(
        "attn_prompt_b", label="Prompt B (schema / OOD)",
        sync_slot="b",
    )

try:
    _model = get_model()
except Exception:
    _model = None

col_ans, _ = st.columns([1, 3])
with col_ans:
    if _model:
        answer_id = token_id_input(
            _model, "Answer token (for DLA-guided head ranking)",
            key="attn_answer",
            help="Optional. If set, heads are ranked by DLA change to surface routing-relevant heads.",
        )
    else:
        st.text_input("Answer token", key="attn_answer", help="Load a model first.")
        answer_id = None

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Run", type="primary")
with col2:
    if st.button("Clear"):
        st.session_state.pop("attn_results", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _run_one(model, tokens, str_tokens, answer_token_id: int | None) -> dict:
    _, cache = run_with_cache(model, tokens, strategy="full")
    result = {
        "tokens": tokens,
        "str_tokens": str_tokens,
        "cache": cache,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
    }
    if answer_token_id is not None:
        result["head_attrs"] = head_attribution(model, cache, answer_token_id, pos=-1)
    return result


if run:
    if not prompt_a:
        st.warning("Prompt A is required.")
        st.stop()
    with st.spinner("Caching activations…"):
        try:
            model = get_model()
            bos = cfg.model.prepend_bos
            toks_a, st_a = prompt_tokenize(
                model, "attn_prompt_a", bos,
            )
            store = {
                "A": _run_one(model, toks_a, st_a, answer_id),
            }
            if prompt_b and prompt_b.strip():
                toks_b, st_b = prompt_tokenize(
                    model, "attn_prompt_b", bos,
                )
                store["B"] = _run_one(
                    model, toks_b, st_b, answer_id,
                )
            st.session_state["attn_results"] = store
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

if "attn_results" not in st.session_state:
    st.stop()

results = st.session_state["attn_results"]
is_comparison = "B" in results
n_layers = results["A"]["n_layers"]
n_heads = results["A"]["n_heads"]

# ---------------------------------------------------------------------------
# DLA-guided head ranking (if answer token provided)
# ---------------------------------------------------------------------------
ranked_heads: list[tuple[int, int, float]] | None = None
if is_comparison and "head_attrs" in results["A"] and "head_attrs" in results["B"]:
    diff = results["B"]["head_attrs"] - results["A"]["head_attrs"]
    flat = diff.flatten()
    order = flat.abs().argsort(descending=True)
    ranked_heads = [
        (idx.item() // n_heads, idx.item() % n_heads, flat[idx].item())
        for idx in order
    ]

st.divider()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
if is_comparison:
    tab_dla, tab_explorer, tab_focus, tab_ov = st.tabs([
        "DLA-Guided Comparison",
        "Layer Explorer",
        "Position Focus",
        "OV Circuits",
    ])
else:
    tab_explorer, tab_focus, tab_ov = st.tabs([
        "Layer Explorer",
        "Position Focus",
        "OV Circuits",
    ])
    tab_dla = None

# ---------------------------------------------------------------------------
# DLA-Guided Comparison (comparison only)
# ---------------------------------------------------------------------------
if is_comparison:
    with tab_dla:
        if ranked_heads is None:
            st.info(
                "Provide an answer token to enable DLA-guided head ranking. "
                "Without it, we can't automatically identify which heads' "
                "routing changed most under schema enforcement."
            )
        else:
            st.markdown(
                "Heads ranked by **change in DLA** under schema enforcement. "
                "Large DLA shifts suggest the head's information routing was "
                "disrupted or redirected by schema tokens."
            )

            n_show = st.slider(
                "Heads to show", 1, min(12, len(ranked_heads)), 4,
                key="attn_n_ranked",
            )

            head_options = [
                f"L{li} H{hi} — {'gained' if dv > 0 else 'lost'} {abs(dv):.4f}"
                for li, hi, dv in ranked_heads[:n_show]
            ]

            dla_head_idx = st.radio(
                "Head", head_options,
                key="attn_dla_head",
            )

            sel = head_options.index(dla_head_idx)
            li, hi, diff_val = ranked_heads[sel]

            col_base, col_schema = st.columns(2)
            for col, (label, label_long, color) in zip(
                [col_base, col_schema],
                [("A", "Baseline", "#2563eb"), ("B", "Schema", "#dc2626")],
            ):
                with col:
                    st.markdown(f"**{label_long}** — L{li} H{hi}")
                    res = results[label]
                    try:
                        html = attention_single_cv(
                            res["cache"], li, hi,
                            res["str_tokens"],
                            max_width=688,
                            positive_color=color,
                        )
                        components.html(
                            html, height=688, scrolling=True,
                        )
                    except Exception as e:
                        st.warning(f"Cannot plot: {e}")

# ---------------------------------------------------------------------------
# Layer Explorer
# ---------------------------------------------------------------------------
with tab_explorer:
    st.markdown(
        "Browse attention patterns layer-by-layer. "
        "Hover over a head thumbnail to preview; click to lock. "
        "Toggle between Baseline and Schema with the radio button."
    )

    # --- Controls row ---
    if is_comparison:
        ctrl_l, ctrl_p = st.columns([3, 2])
    else:
        ctrl_l = st.container()
    with ctrl_l:
        layer = st.slider("Layer", 0, n_layers - 1, 0, key="attn_exp_layer")
    if is_comparison:
        with ctrl_p:
            exp_prompt = st.radio(
                "Prompt", ["Baseline", "Schema"],
                horizontal=True, key="attn_exp_prompt",
            )
        exp_label = "A" if exp_prompt == "Baseline" else "B"
    else:
        exp_label = "A"

    res = results[exp_label]
    cache = res["cache"]
    str_toks = res["str_tokens"]
    seq_len = len(str_toks)

    # Full-width CircuitsVis widget — it manages its own
    # thumbnail grid + expanded detail view internally.
    try:
        html = attention_heads_cv(cache, layer, str_toks)
        # Large initial height — the injected ResizeObserver will
        # shrink the iframe to fit once CircuitsVis renders.
        components.html(html, height=1500, scrolling=False)
    except Exception as e:
        st.warning(f"Cannot plot: {e}")

# ---------------------------------------------------------------------------
# Position Focus — attention to/from a specific token
# ---------------------------------------------------------------------------
with tab_focus:
    st.markdown(
        "Pick a destination token and see where it attends (its source row). "
        "Useful for checking whether the answer-predicting position attends to "
        "schema tokens, content tokens, or structural delimiters."
    )

    n_ctrl = 4 if is_comparison else 3
    focus_cols = st.columns(n_ctrl)
    with focus_cols[0]:
        focus_layer = st.slider(
            "Layer", 0, n_layers - 1, 0, key="attn_focus_layer",
        )
    with focus_cols[1]:
        focus_head = st.slider(
            "Head", 0, n_heads - 1, 0, key="attn_focus_head",
        )
    with focus_cols[2]:
        max_pos = len(results["A"]["str_tokens"]) - 1
        focus_pos = st.slider(
            "Destination position",
            0, max_pos, max_pos,
            key="attn_focus_pos",
            help="Which query position's attention to show",
        )
    if is_comparison:
        with focus_cols[3]:
            focus_prompt = st.radio(
                "Prompt", ["Baseline", "Schema"],
                horizontal=True, key="attn_focus_prompt",
            )
        focus_label = "A" if focus_prompt == "Baseline" else "B"
        focus_label_long = focus_prompt
    else:
        focus_label = "A"
        focus_label_long = "Baseline"

    res = results[focus_label]
    cache = res["cache"]
    str_toks = res["str_tokens"]
    pos = min(focus_pos, len(str_toks) - 1)

    st.markdown(
        f"**{focus_label_long}** — L{focus_layer} H{focus_head}, "
        f"attention from pos {pos} `{repr(str_toks[pos])}`"
    )

    try:
        pattern = (
            cache["pattern", focus_layer][0, focus_head]
            .detach().cpu().float().numpy()
        )
        attn_row = pattern[pos, :len(str_toks)]
        st.plotly_chart(
            attention_source_row(
                attn_row, str_toks,
                dst_label=repr(str_toks[pos]),
                title=(
                    f"L{focus_layer} H{focus_head} "
                    f"({focus_label_long}) — pos {pos}"
                ),
            ),
            use_container_width=True,
        )

        # Full pattern via CircuitsVis
        html = attention_single_cv(
            cache, focus_layer, focus_head, str_toks,
        )
        components.html(html, height=1000, scrolling=False)
    except Exception as e:
        st.warning(f"Cannot plot ({focus_label_long}): {e}")

# ---------------------------------------------------------------------------
# OV Circuits — activity trace (neuroscience-style) with A/B comparison
# ---------------------------------------------------------------------------
with tab_ov:
    st.markdown(
        "**Neuroscience-style activity trace.** "
        "Given the stimulus (input tokens), see which heads fired strongest "
        "at a position, pick a seed head, and trace its signal forward and "
        "backward through the model. "
        + ("Compare baseline vs constrained to see where circuits break."
           if is_comparison else "")
    )

    import numpy as np
    import plotly.express as px

    model = get_model()

    ov_cache_a = results["A"]["cache"]
    ov_str_tokens_a = results["A"]["str_tokens"]
    seq_len_a = len(ov_str_tokens_a)

    if is_comparison:
        ov_cache_b = results["B"]["cache"]
        ov_str_tokens_b = results["B"]["str_tokens"]
        seq_len_b = len(ov_str_tokens_b)

    # --- Step 1: Position selector + activity map ---
    st.subheader("1. Activity map — which heads fired?")
    st.caption(
        "Per-head output norm at the selected position: "
        "‖z @ W_O‖. Bright = strong contribution to the residual stream."
    )

    col_pos_a = st.columns(2 if is_comparison else 1)
    with col_pos_a[0]:
        ov_pos_a = st.slider(
            "Position (Baseline)" if is_comparison else "Token position",
            0, seq_len_a - 1, seq_len_a - 1,
            key="ov_pos_a",
            help="Position to analyze in the baseline prompt.",
        )
        st.caption(
            f"Baseline pos {ov_pos_a}: `{repr(ov_str_tokens_a[ov_pos_a])}`"
        )
    if is_comparison:
        with col_pos_a[1]:
            ov_pos_b = st.slider(
                "Position (Schema)",
                0, seq_len_b - 1, seq_len_b - 1,
                key="ov_pos_b",
                help="Position to analyze in the schema prompt.",
            )
            st.caption(
                f"Schema pos {ov_pos_b}: `{repr(ov_str_tokens_b[ov_pos_b])}`"
            )

    act_a = head_activity(model, ov_cache_a, pos=ov_pos_a)
    if is_comparison:
        act_b = head_activity(model, ov_cache_b, pos=ov_pos_b)

    # Show top-firing heads from baseline as quick-pick options
    flat_act = act_a.flatten()
    top_indices = flat_act.argsort()[::-1][:8]
    top_heads = [
        (int(idx) // n_heads, int(idx) % n_heads) for idx in top_indices
    ]
    top_labels = [
        f"L{l} H{h} ({flat_act[l * n_heads + h]:.1f})"
        for l, h in top_heads
    ]

    col_pick, col_manual_l, col_manual_h = st.columns([2, 1, 1])
    with col_pick:
        pick = st.selectbox(
            "Quick-pick from top-firing heads (baseline)",
            top_labels,
            key="ov_quick_pick",
        )
        pick_idx = top_labels.index(pick)
        default_layer, default_head = top_heads[pick_idx]
    with col_manual_l:
        seed_layer = st.number_input(
            "Seed layer", 0, n_layers - 1, default_layer, key="ov_seed_layer",
        )
    with col_manual_h:
        seed_head = st.number_input(
            "Seed head", 0, n_heads - 1, default_head, key="ov_seed_head",
        )

    if is_comparison:
        col_act_a, col_act_b, col_act_diff = st.columns(3)
        with col_act_a:
            st.plotly_chart(
                activity_heatmap(
                    act_a,
                    title=f"Baseline — pos {ov_pos_a}",
                    highlight=(seed_layer, seed_head),
                ),
                use_container_width=True,
            )
        with col_act_b:
            st.plotly_chart(
                activity_heatmap(
                    act_b,
                    title=f"Schema — pos {ov_pos_b}",
                    highlight=(seed_layer, seed_head),
                ),
                use_container_width=True,
            )
        with col_act_diff:
            diff_act = act_b - act_a
            vabs = max(float(np.abs(diff_act).max()), 1e-6)
            fig_diff = px.imshow(
                diff_act,
                color_continuous_scale="RdBu",
                zmin=-vabs, zmax=vabs,
                labels={"x": "Head", "y": "Layer", "color": "Δ‖output‖"},
                title="Schema − Baseline",
                aspect="auto",
            )
            fig_diff.update_layout(
                xaxis=dict(tickmode="linear", dtick=max(1, n_heads // 16)),
                yaxis=dict(tickmode="linear", dtick=max(1, n_layers // 20)),
                height=max(300, n_layers * 22 + 100),
            )
            fig_diff.add_shape(
                type="rect",
                x0=seed_head - 0.5, x1=seed_head + 0.5,
                y0=seed_layer - 0.5, y1=seed_layer + 0.5,
                line=dict(color="black", width=3),
            )
            st.plotly_chart(fig_diff, use_container_width=True)
    else:
        st.plotly_chart(
            activity_heatmap(
                act_a,
                title=f"Head activity at pos {ov_pos_a}: "
                      f"`{repr(ov_str_tokens_a[ov_pos_a])}`",
                highlight=(seed_layer, seed_head),
            ),
            use_container_width=True,
        )

    # --- Step 2: Eigenvalue spectrum of the seed head ---
    st.subheader(f"2. Eigenvalue spectrum — L{seed_layer} H{seed_head}")
    st.caption(
        "Top eigenvalues of W_OV in the complex plane. "
        "Near +1 = copying, near -1 = negation, small |λ| = suppression. "
        "(Weight-based — same for both prompts.)"
    )

    eigs = ov_eigenvalues_single(model, seed_layer, seed_head)
    cscore = copying_score(eigs)
    col_metric, col_slider = st.columns([1, 3])
    with col_metric:
        st.metric("Copying score", f"{cscore:.3f}")
    with col_slider:
        eig_k = st.slider(
            "Top-k eigenvalues", 5, min(50, len(eigs)), 20, key="ov_eig_k",
        )
    st.plotly_chart(
        eigenvalue_spectrum(eigs, seed_layer, seed_head, top_k=eig_k),
        use_container_width=True,
    )

    # --- Step 3: Signal trace — who reads from / writes to the seed? ---
    st.subheader("3. Signal trace — composition on this input")
    if is_comparison:
        st.caption(
            "How much of the seed head's output reaches each downstream "
            "head (readers) and how much each upstream head contributes to "
            "the seed (writers). Side-by-side: does constrained decoding "
            "break the circuit edges?"
        )
    else:
        st.caption(
            "How much of the seed head's actual output reaches each "
            "downstream head's value space (readers), and how much of each "
            "upstream head's output reaches the seed (writers)."
        )

    readers_a = composition_from_act(
        model, ov_cache_a, seed_layer, seed_head, pos=ov_pos_a,
    )
    writers_a = composition_to_act(
        model, ov_cache_a, seed_layer, seed_head, pos=ov_pos_a,
    )
    if is_comparison:
        readers_b = composition_from_act(
            model, ov_cache_b, seed_layer, seed_head, pos=ov_pos_b,
        )
        writers_b = composition_to_act(
            model, ov_cache_b, seed_layer, seed_head, pos=ov_pos_b,
        )

    def _comp_heatmap(data, title, cmap="Plasma"):
        vmax = float(data.max()) or 1e-6
        fig = px.imshow(
            data, color_continuous_scale=cmap,
            zmin=0, zmax=vmax,
            labels={"x": "Head", "y": "Layer", "color": "‖contrib‖"},
            title=title, aspect="auto",
        )
        fig.update_layout(
            xaxis=dict(tickmode="linear", dtick=max(1, n_heads // 16)),
            yaxis=dict(tickmode="linear", dtick=max(1, n_layers // 20)),
            height=350,
        )
        return fig

    def _diff_heatmap(data, title):
        vabs = max(float(np.abs(data).max()), 1e-6)
        fig = px.imshow(
            data, color_continuous_scale="RdBu",
            zmin=-vabs, zmax=vabs,
            labels={"x": "Head", "y": "Layer", "color": "Δ‖contrib‖"},
            title=title, aspect="auto",
        )
        fig.update_layout(
            xaxis=dict(tickmode="linear", dtick=max(1, n_heads // 16)),
            yaxis=dict(tickmode="linear", dtick=max(1, n_layers // 20)),
            height=350,
        )
        return fig

    # --- Readers ---
    st.markdown(f"**Reads from L{seed_layer} H{seed_head}** (downstream)")
    if is_comparison:
        cr_a, cr_b, cr_d = st.columns(3)
        with cr_a:
            st.plotly_chart(
                _comp_heatmap(readers_a, "Baseline"), use_container_width=True,
            )
        with cr_b:
            st.plotly_chart(
                _comp_heatmap(readers_b, "Schema"), use_container_width=True,
            )
        with cr_d:
            st.plotly_chart(
                _diff_heatmap(readers_b - readers_a, "Schema − Baseline"),
                use_container_width=True,
            )
    else:
        st.plotly_chart(
            _comp_heatmap(readers_a, "Readers"), use_container_width=True,
        )

    # --- Writers ---
    st.markdown(f"**Writes to L{seed_layer} H{seed_head}** (upstream)")
    if is_comparison:
        cw_a, cw_b, cw_d = st.columns(3)
        with cw_a:
            st.plotly_chart(
                _comp_heatmap(writers_a, "Baseline"), use_container_width=True,
            )
        with cw_b:
            st.plotly_chart(
                _comp_heatmap(writers_b, "Schema"), use_container_width=True,
            )
        with cw_d:
            st.plotly_chart(
                _diff_heatmap(writers_b - writers_a, "Schema − Baseline"),
                use_container_width=True,
            )
    else:
        st.plotly_chart(
            _comp_heatmap(writers_a, "Writers"), use_container_width=True,
        )

    # --- Step 4: Circuit graph ---
    st.subheader("4. Circuit graph")
    st.caption(
        "BFS from the seed head using activation-based composition. "
        "Edges show how much of a reader's value came from the writer."
    )
    col_depth, col_topk = st.columns(2)
    with col_depth:
        trace_depth = st.slider("Trace depth", 1, 4, 2, key="ov_trace_depth")
    with col_topk:
        trace_k = st.slider("Top-k per hop", 2, 10, 5, key="ov_trace_k")

    edges_a = trace_circuit_act(
        model, ov_cache_a, seed_layer, seed_head,
        pos=ov_pos_a, depth=trace_depth, top_k=trace_k,
    )

    if is_comparison:
        edges_b = trace_circuit_act(
            model, ov_cache_b, seed_layer, seed_head,
            pos=ov_pos_b, depth=trace_depth, top_k=trace_k,
        )
        col_g_a, col_g_b = st.columns(2)
        with col_g_a:
            st.plotly_chart(
                circuit_graph(
                    edges_a, n_layers, n_heads,
                    title=f"Baseline — L{seed_layer} H{seed_head} "
                          f"at pos {ov_pos_a}",
                ),
                use_container_width=True,
            )
        with col_g_b:
            st.plotly_chart(
                circuit_graph(
                    edges_b, n_layers, n_heads,
                    title=f"Schema — L{seed_layer} H{seed_head} "
                          f"at pos {ov_pos_b}",
                ),
                use_container_width=True,
            )

        # Summary: edges that weakened or disappeared
        edge_map_a = {(la, ha, lb, hb): s for la, ha, lb, hb, s in edges_a}
        edge_map_b = {(la, ha, lb, hb): s for la, ha, lb, hb, s in edges_b}
        all_edges = set(edge_map_a) | set(edge_map_b)
        edge_diffs = []
        for e in all_edges:
            sa = edge_map_a.get(e, 0.0)
            sb = edge_map_b.get(e, 0.0)
            edge_diffs.append((*e, sa, sb, sb - sa))
        edge_diffs.sort(key=lambda x: x[-1])  # most weakened first

        st.markdown("**Circuit edge changes** (sorted by largest drop)")
        rows = []
        for la, ha, lb, hb, sa, sb, delta in edge_diffs[:15]:
            rows.append({
                "Edge": f"L{la}H{ha} → L{lb}H{hb}",
                "Baseline": f"{sa:.3f}",
                "Schema": f"{sb:.3f}",
                "Δ": f"{delta:+.3f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.plotly_chart(
            circuit_graph(
                edges_a, n_layers, n_heads,
                title=f"Circuit from L{seed_layer} H{seed_head} "
                      f"at pos {ov_pos_a}",
            ),
            use_container_width=True,
        )
