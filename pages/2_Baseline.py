"""Experiment 01: Clean baseline attribution.

Which components (heads, MLPs, embeddings) directly contribute to the answer logit
on the clean prompt?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from connectomics.app_state import get_config, get_model, render_sidebar_memory
from connectomics.viz_interactive import (
    tokenization_table,
    component_attribution_bar,
    head_attribution_heatmap,
)

st.set_page_config(page_title="Baseline — LLM Connectomics", layout="wide")
render_sidebar_memory()

st.title("Baseline")
st.caption("Clean prompt direct logit attribution — which components write the answer?")

cfg = get_config()

# --- Sidebar config ---
with st.sidebar:
    st.subheader("Run Config")
    clean_prompt = st.text_input("Clean prompt", value=cfg.prompts.clean)
    answer_tok = st.text_input("Answer token", value=cfg.tokens.answer)
    strategy = st.selectbox(
        "Cache strategy",
        ["full", "resid", "minimal"],
        index=0,
        help="'full' caches everything. Use 'resid' for DLA only (saves memory).",
    )

# --- Controls ---
col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Run Baseline", type="primary")
with col2:
    if st.button("Clear Results"):
        st.session_state.pop("baseline_results", None)
        st.rerun()

# --- Run ---
if run:
    with st.spinner("Running baseline…"):
        try:
            model = get_model()
            from connectomics.model import tokenize, run_with_cache
            from connectomics.attribution import (
                final_logit_margin,
                component_attribution,
                head_attribution,
            )

            tokens, str_tokens = tokenize(model, clean_prompt, prepend_bos=cfg.model.prepend_bos)
            logits, cache = run_with_cache(model, tokens, strategy=strategy)
            margin = final_logit_margin(model, logits, answer_tok)
            comp_attrs, comp_labels = component_attribution(model, cache, answer_tok, pos=-1)
            head_attrs = head_attribution(model, cache, answer_tok, pos=-1)

            st.session_state["baseline_results"] = {
                "tokens": tokens,
                "str_tokens": str_tokens,
                "margin": margin,
                "comp_attrs": comp_attrs,
                "comp_labels": comp_labels,
                "head_attrs": head_attrs,
                "answer_tok": answer_tok,
                "prompt": clean_prompt,
            }
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# --- Display ---
if "baseline_results" in st.session_state:
    r = st.session_state["baseline_results"]
    ans_repr = repr(r["answer_tok"])

    st.divider()
    st.subheader(f"Results — `{r['prompt']}`")

    c1, c2, c3 = st.columns(3)
    m = r["margin"]
    c1.metric("P(answer)", f"{m['prob']:.5f}")
    c2.metric("Answer logit", f"{m['logit']:.3f}")
    c3.metric("Rank", m["rank"])

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Tokenization", "Component Attribution", "Head Attribution", "Top Heads"]
    )

    with tab1:
        st.plotly_chart(
            tokenization_table(r["str_tokens"], r["tokens"]),
            use_container_width=True,
        )

    with tab2:
        st.plotly_chart(
            component_attribution_bar(r["comp_attrs"], r["comp_labels"], ans_repr),
            use_container_width=True,
        )

    with tab3:
        st.plotly_chart(
            head_attribution_heatmap(
                r["head_attrs"],
                title=f"Per-head DLA toward {ans_repr}",
            ),
            use_container_width=True,
        )

    with tab4:
        n_heads = r["head_attrs"].shape[1]
        flat = r["head_attrs"].flatten()
        k = st.slider("Show top K heads", 5, 30, 15, key="baseline_topk")
        top_idx = flat.abs().argsort(descending=True)[:k]
        rows = [
            {
                "Rank": rank + 1,
                "Layer": idx.item() // n_heads,
                "Head": idx.item() % n_heads,
                "Attribution": round(flat[idx].item(), 5),
            }
            for rank, idx in enumerate(top_idx)
        ]
        st.dataframe(rows, use_container_width=False, hide_index=True)
