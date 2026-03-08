"""Experiment 04: Corruption sweep.

Run multiple whitespace/newline injection variants against the clean prompt
and compare the head attribution diffs across all of them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from app_state import get_config, get_model, render_sidebar_memory
from viz_interactive import sweep_summary_heatmap, head_attribution_heatmap

st.set_page_config(page_title="Sweep — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Corruption Sweep")
st.caption(
    "Inject whitespace/newline tokens into the clean prompt across multiple variants. "
    "Compare which heads are consistently disrupted vs. placement-dependent."
)

cfg = get_config()
variants = cfg.corruption_sweep.named_configs()

# --- Sidebar config ---
with st.sidebar:
    st.subheader("Run Config")
    clean_prompt = st.text_input("Clean prompt", value=cfg.prompts.clean)
    answer_tok = st.text_input("Answer token", value=cfg.tokens.answer)
    top_k = st.slider(
        "Top K heads in summary heatmap", 4, 32, 12,
        help="Heads ranked by |clean attribution|.",
    )

    st.subheader("Variants")
    if variants:
        st.caption(f"{len(variants)} variants from config.yaml")
        for label, v in variants:
            st.caption(f"• **{label}**: inject={v.inject!r}  count={v.count}  seed={v.seed}")
    else:
        st.warning("No variants configured. Add them to `corruption_sweep.variants` on the Home page.")

# --- Controls ---
col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Run Sweep", type="primary", disabled=not variants)
with col2:
    if st.button("Clear Results"):
        st.session_state.pop("sweep_results", None)
        st.rerun()

if not variants:
    st.info("Configure sweep variants on the Home page under `corruption_sweep.variants`.")

# --- Run ---
if run and variants:
    progress = st.progress(0, text="Running clean baseline…")
    status = st.empty()

    try:
        model = get_model()
        from model import tokenize, run_with_cache, corrupt_tokens as _corrupt
        from attribution import final_logit_margin, head_attribution

        # Clean baseline
        tokens, str_tokens = tokenize(model, clean_prompt, prepend_bos=cfg.model.prepend_bos)
        logits, cache = run_with_cache(model, tokens, strategy=cfg.model.cache_strategy)
        clean_margin = final_logit_margin(model, logits, answer_tok)
        clean_attrs = head_attribution(model, cache, answer_tok, pos=-1)

        variant_labels, diffs, probs = [], [], []
        n_total = len(variants) + 1

        for i, (label, vcfg) in enumerate(variants):
            frac = (i + 1) / n_total
            progress.progress(frac, text=f"[{label}] inject={vcfg.inject!r} count={vcfg.count}…")

            inj_tokens = _corrupt(model, tokens, vcfg, prepend_bos=cfg.model.prepend_bos)
            inj_logits, inj_cache = run_with_cache(
                model, inj_tokens, strategy=cfg.model.cache_strategy
            )
            margin = final_logit_margin(model, inj_logits, answer_tok)
            attrs = head_attribution(model, inj_cache, answer_tok, pos=-1)
            diff = attrs - clean_attrs

            variant_labels.append(label)
            diffs.append(diff)
            probs.append(margin["prob"])

        progress.empty()
        status.empty()

        st.session_state["sweep_results"] = {
            "clean_margin": clean_margin,
            "clean_attrs": clean_attrs,
            "variant_labels": variant_labels,
            "diffs": diffs,
            "probs": probs,
            "answer_tok": answer_tok,
            "prompt": clean_prompt,
        }
        st.success(f"Done. {len(variants)} variants completed.")

    except Exception as e:
        progress.empty()
        status.empty()
        st.error(f"Error: {e}")
        st.exception(e)

# --- Display ---
if "sweep_results" in st.session_state:
    r = st.session_state["sweep_results"]
    ans_repr = repr(r["answer_tok"])

    st.divider()
    st.subheader(f"Results — `{r['prompt']}`")

    # P(answer) table
    tab1, tab2, tab3 = st.tabs(["P(answer) Table", "Summary Heatmap", "Per-Variant Diffs"])

    with tab1:
        rows = [
            {"Variant": "clean (baseline)", "P(answer)": round(r["clean_margin"]["prob"], 5)}
        ]
        rows += [
            {"Variant": label, "P(answer)": round(prob, 5)}
            for label, prob in zip(r["variant_labels"], r["probs"])
        ]
        st.dataframe(rows, use_container_width=False, hide_index=True)

    with tab2:
        st.plotly_chart(
            sweep_summary_heatmap(
                r["variant_labels"],
                r["diffs"],
                r["clean_attrs"],
                answer_label=ans_repr,
                k=top_k,
            ),
            use_container_width=True,
        )

    with tab3:
        selected = st.selectbox("Select variant", r["variant_labels"], key="sweep_select")
        if selected:
            idx = r["variant_labels"].index(selected)
            prob_val = r["probs"][idx]
            clean_prob = r["clean_margin"]["prob"]
            delta_pct = (prob_val - clean_prob) / max(clean_prob, 1e-8) * 100

            col1, col2, col3 = st.columns(3)
            col1.metric("Clean P(answer)", f"{clean_prob:.5f}")
            col2.metric(f"{selected} P(answer)", f"{prob_val:.5f}")
            col3.metric("Delta", f"{delta_pct:+.1f}%")

            st.plotly_chart(
                head_attribution_heatmap(
                    r["diffs"][idx],
                    title=f"Attribution diff: {selected} - clean ({ans_repr})",
                ),
                use_container_width=True,
            )

            # Top changed heads
            n_heads = r["diffs"][idx].shape[1]
            flat = r["diffs"][idx].flatten()
            k_heads = st.slider("Show top K heads", 5, 20, 10, key="sweep_topk")
            top_idx = flat.abs().argsort(descending=True)[:k_heads]
            head_rows = [
                {
                    "Rank": rank + 1,
                    "Layer": i.item() // n_heads,
                    "Head": i.item() % n_heads,
                    "Attribution diff": round(flat[i].item(), 5),
                }
                for rank, i in enumerate(top_idx)
            ]
            st.dataframe(head_rows, use_container_width=False, hide_index=True)
