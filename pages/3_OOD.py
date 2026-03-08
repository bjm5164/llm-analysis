"""Experiment 02: OOD comparison.

Compare head attribution between the clean and corrupted prompts.
Shows where the model's computation changes when the OOD suffix is injected.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from app_state import get_config, get_model, render_sidebar_memory
from viz_interactive import (
    tokenization_table,
    component_attribution_bar,
    head_attribution_heatmap,
    attention_pattern_heatmap,
)

st.set_page_config(page_title="OOD — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("OOD Comparison")
st.caption("Compare head attribution: clean vs corrupted prompt.")

cfg = get_config()

# --- Sidebar config ---
with st.sidebar:
    st.subheader("Run Config")
    clean_prompt = st.text_input("Clean prompt", value=cfg.prompts.clean)
    corrupted_prompt = st.text_input("Corrupted prompt", value=cfg.prompts.corrupted)
    answer_tok = st.text_input("Answer token", value=cfg.tokens.answer)
    n_attn_heads = st.slider(
        "Attention patterns for top N changed heads", 1, 8, 4,
        help="Number of heads (by |attribution diff|) to show attention patterns for.",
    )

# --- Controls ---
col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Run OOD", type="primary")
with col2:
    if st.button("Clear Results"):
        st.session_state.pop("ood_results", None)
        st.rerun()

# --- Run ---
if run:
    with st.spinner("Running OOD comparison…"):
        try:
            model = get_model()
            from model import tokenize, run_with_cache
            from attribution import final_logit_margin, head_attribution

            from attribution import component_attribution

            results = {}
            for label, prompt in [("clean", clean_prompt), ("corrupted", corrupted_prompt)]:
                tokens, str_tokens = tokenize(model, prompt, prepend_bos=cfg.model.prepend_bos)
                logits, cache = run_with_cache(model, tokens, strategy=cfg.model.cache_strategy)
                margin = final_logit_margin(model, logits, answer_tok)
                ha = head_attribution(model, cache, answer_tok, pos=-1)
                ca, ca_labels = component_attribution(model, cache, answer_tok, pos=-1)
                results[label] = {
                    "tokens": tokens,
                    "str_tokens": str_tokens,
                    "margin": margin,
                    "head_attrs": ha,
                    "comp_attrs": ca,
                    "comp_labels": ca_labels,
                    "cache": cache,
                }

            head_diff = results["corrupted"]["head_attrs"] - results["clean"]["head_attrs"]

            # Component diff — labels must match (same model structure, pos=-1)
            comp_diff = results["corrupted"]["comp_attrs"] - results["clean"]["comp_attrs"]
            clean_scale = results["clean"]["comp_attrs"].abs().sum().item()
            comp_diff_norm = comp_diff / max(clean_scale, 1e-8)

            st.session_state["ood_results"] = {
                "results": results,
                "diff": head_diff,
                "comp_diff": comp_diff,
                "comp_diff_norm": comp_diff_norm,
                "answer_tok": answer_tok,
            }
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

# --- Display ---
if "ood_results" in st.session_state:
    r = st.session_state["ood_results"]
    results = r["results"]
    diff = r["diff"]
    ans_repr = repr(r["answer_tok"])

    st.divider()

    # Logit margin comparison
    st.subheader("Logit Margin")
    cols = st.columns(len(results))
    for col, (label, res) in zip(cols, results.items()):
        m = res["margin"]
        col.metric(f"{label}", f"P={m['prob']:.5f}")
        col.caption(f"logit={m['logit']:.3f}  rank={m['rank']}")

    st.divider()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Head Attribution Diff",
        "Component Attribution",
        "Clean Heads",
        "Corrupted Heads",
        "Tokenization",
        "Attention Patterns",
    ])

    with tab1:
        st.plotly_chart(
            head_attribution_heatmap(
                diff,
                title=f"Head DLA diff: corrupted - clean ({ans_repr})",
            ),
            use_container_width=True,
        )
        # Top changed heads table
        n_heads = diff.shape[1]
        flat = diff.flatten()
        k = st.slider("Show top K", 5, 20, 10, key="ood_topk")
        top_idx = flat.abs().argsort(descending=True)[:k]
        rows = [
            {
                "Rank": rank + 1,
                "Layer": idx.item() // n_heads,
                "Head": idx.item() % n_heads,
                "Attribution diff": round(flat[idx].item(), 5),
            }
            for rank, idx in enumerate(top_idx)
        ]
        st.dataframe(rows, use_container_width=False, hide_index=True)

    with tab2:
        labels = results["clean"]["comp_labels"]
        st.markdown("**Clean**")
        st.plotly_chart(
            component_attribution_bar(
                results["clean"]["comp_attrs"], labels, f"{ans_repr} (clean)"
            ),
            use_container_width=True,
        )
        st.markdown("**Corrupted**")
        st.plotly_chart(
            component_attribution_bar(
                results["corrupted"]["comp_attrs"], labels, f"{ans_repr} (corrupted)"
            ),
            use_container_width=True,
        )
        st.markdown(
            "**Difference** — raw (corrupted − clean) and normalised by "
            "Σ|clean attribution| so values are fractions of the clean total"
        )
        col_raw, col_norm = st.columns(2)
        with col_raw:
            st.plotly_chart(
                component_attribution_bar(
                    r["comp_diff"], labels, f"{ans_repr} diff (raw)"
                ),
                use_container_width=True,
            )
        with col_norm:
            st.plotly_chart(
                component_attribution_bar(
                    r["comp_diff_norm"], labels, f"{ans_repr} diff (normalised)"
                ),
                use_container_width=True,
            )

    with tab3:
        st.plotly_chart(
            head_attribution_heatmap(
                results["clean"]["head_attrs"],
                title=f"Clean DLA: {ans_repr}",
            ),
            use_container_width=True,
        )

    with tab4:
        st.plotly_chart(
            head_attribution_heatmap(
                results["corrupted"]["head_attrs"],
                title=f"Corrupted DLA: {ans_repr}",
            ),
            use_container_width=True,
        )

    with tab5:
        for label, res in results.items():
            st.markdown(f"**{label}**")
            st.plotly_chart(
                tokenization_table(res["str_tokens"], res["tokens"]),
                use_container_width=True,
            )

    with tab6:
        n_heads = diff.shape[1]
        flat = diff.flatten()
        top_idx = flat.abs().argsort(descending=True)[:n_attn_heads]

        for rank, idx in enumerate(top_idx):
            li = idx.item() // n_heads
            hi = idx.item() % n_heads
            attr_val = flat[idx].item()
            st.markdown(f"**L{li} H{hi}** — attribution change: {attr_val:+.5f}")

            cols = st.columns(2)
            for col, label in zip(cols, ["clean", "corrupted"]):
                with col:
                    st.markdown(f"*{label}*")
                    try:
                        fig = attention_pattern_heatmap(
                            results[label]["cache"],
                            li, hi,
                            results[label]["str_tokens"],
                            title=f"L{li} H{hi} ({label})",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Cannot plot attention ({label}): {e}")
            st.divider()
