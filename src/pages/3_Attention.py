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
from viz_interactive import (
    attention_heads_cv,
    attention_single_cv,
    attention_source_row,
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
    tab_dla, tab_explorer, tab_focus = st.tabs([
        "DLA-Guided Comparison",
        "Layer Explorer",
        "Position Focus",
    ])
else:
    tab_explorer, tab_focus = st.tabs([
        "Layer Explorer",
        "Position Focus",
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
