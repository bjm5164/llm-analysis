"""Attention Maps.

How does structured schema enforcement change where the model looks?

Workflow:
  1. Run Prompt A (baseline) and Prompt B (with schema / OOD suffix).
  2. Compare attention patterns head-by-head and layer-by-layer.
  3. Look for heads that redirect attention toward (or away from) schema
     tokens, answer-relevant positions, or structural delimiters.
"""

import streamlit as st

from app_state import get_config, get_model, render_sidebar_memory, prompt_selector, token_id_input
from attribution import head_attribution
from model import run_with_cache, tokenize
from viz_interactive import (
    attention_pattern_heatmap,
    attention_all_heads_heatmap,
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
        "attn_prompt_a", label="Prompt A (baseline)", allow_empty=False,
    )
with col_b:
    prompt_b = prompt_selector(
        "attn_prompt_b", label="Prompt B (schema / OOD)",
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
def _run_one(model, prompt: str, answer_token_id: int | None) -> dict:
    tokens, str_tokens = tokenize(model, prompt, prepend_bos=cfg.model.prepend_bos)
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
            store = {"A": _run_one(model, prompt_a, answer_id)}
            if prompt_b and prompt_b.strip():
                store["B"] = _run_one(model, prompt_b, answer_id)
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

            n_show = st.slider("Heads to show", 1, min(12, len(ranked_heads)), 4, key="attn_n_ranked")

            for li, hi, diff_val in ranked_heads[:n_show]:
                direction = "gained" if diff_val > 0 else "lost"
                st.markdown(
                    f"### L{li} H{hi} — {direction} {abs(diff_val):.4f} attribution"
                )

                # Full-width stacked layout so tokens aren't cropped
                for label, label_long in [("A", "Baseline"), ("B", "Schema")]:
                    res = results[label]
                    try:
                        fig = attention_pattern_heatmap(
                            res["cache"], li, hi, res["str_tokens"],
                            title=f"L{li} H{hi} — {label_long}",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Cannot plot ({label_long}): {e}")

                # Source-row comparison at the final token position
                st.markdown(
                    f"**Attention from final token** — where does L{li} H{hi} look "
                    f"when predicting the next token?"
                )
                for label, label_long in [("A", "Baseline"), ("B", "Schema")]:
                    res = results[label]
                    str_toks = res["str_tokens"]
                    try:
                        pattern = (
                            res["cache"]["pattern", li][0, hi]
                            .detach().cpu().float().numpy()
                        )
                        last = len(str_toks) - 1
                        attn_row = pattern[last, :len(str_toks)]
                        st.plotly_chart(
                            attention_source_row(
                                attn_row, str_toks,
                                dst_label=repr(str_toks[last]),
                                title=f"{label_long} — final token attention",
                            ),
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"Cannot plot source row ({label_long}): {e}")

                st.divider()

# ---------------------------------------------------------------------------
# Layer Explorer
# ---------------------------------------------------------------------------
with tab_explorer:
    st.markdown(
        "Browse attention patterns layer-by-layer. "
        "In comparison mode, patterns are shown side-by-side."
    )

    c1, c2 = st.columns([2, 2])
    with c1:
        layer = st.slider("Layer", 0, n_layers - 1, 0, key="attn_exp_layer")
    with c2:
        head_opts = ["All heads"] + [f"Head {h}" for h in range(n_heads)]
        head_sel = st.selectbox("Head", head_opts, key="attn_exp_head")

    prompt_labels = [("A", "Baseline")] + ([("B", "Schema")] if is_comparison else [])

    if is_comparison:
        cols = st.columns(2)
    else:
        cols = [st.container()]

    for col, (label, label_long) in zip(cols, prompt_labels):
        with col:
            st.markdown(f"#### {label_long}")
            res = results[label]
            cache = res["cache"]
            str_toks = res["str_tokens"]

            try:
                if head_sel == "All heads":
                    st.plotly_chart(
                        attention_all_heads_heatmap(cache, layer, str_toks),
                        use_container_width=True,
                    )
                else:
                    head = int(head_sel.split()[1])
                    st.plotly_chart(
                        attention_pattern_heatmap(
                            cache, layer, head, str_toks,
                            title=f"L{layer} H{head} ({label_long})",
                        ),
                        use_container_width=True,
                    )
            except Exception as e:
                st.warning(f"Cannot plot ({label_long}): {e}")

# ---------------------------------------------------------------------------
# Position Focus — attention to/from a specific token
# ---------------------------------------------------------------------------
with tab_focus:
    st.markdown(
        "Pick a destination token and see where it attends (its source row). "
        "Useful for checking whether the answer-predicting position attends to "
        "schema tokens, content tokens, or structural delimiters."
    )

    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        focus_layer = st.slider("Layer", 0, n_layers - 1, 0, key="attn_focus_layer")
    with c2:
        focus_head = st.slider("Head", 0, n_heads - 1, 0, key="attn_focus_head")
    with c3:
        # Use baseline prompt length for position slider
        max_pos = len(results["A"]["str_tokens"]) - 1
        focus_pos = st.slider(
            "Destination position",
            0, max_pos, max_pos,
            key="attn_focus_pos",
            help="Which query position's attention to show",
        )

    prompt_labels = [("A", "Baseline")] + ([("B", "Schema")] if is_comparison else [])

    for label, label_long in prompt_labels:
        res = results[label]
        cache = res["cache"]
        str_toks = res["str_tokens"]
        pos = min(focus_pos, len(str_toks) - 1)

        st.markdown(
            f"**{label_long}** — L{focus_layer} H{focus_head}, "
            f"attention from pos {pos} `{repr(str_toks[pos])}`"
        )

        try:
            pattern = cache["pattern", focus_layer][0, focus_head].detach().cpu().float().numpy()
            attn_row = pattern[pos, :len(str_toks)]
            st.plotly_chart(
                attention_source_row(
                    attn_row, str_toks,
                    dst_label=repr(str_toks[pos]),
                    title=f"L{focus_layer} H{focus_head} ({label_long}) — pos {pos}",
                ),
                use_container_width=True,
            )

            # Full pattern for context
            st.plotly_chart(
                attention_pattern_heatmap(
                    cache, focus_layer, focus_head, str_toks,
                    title=f"L{focus_layer} H{focus_head} ({label_long})",
                ),
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Cannot plot ({label_long}): {e}")
