"""Experiment 03: Activation patching.

Sweep clean activations into the corrupted run at each (layer, position)
to localize where the model's computation is disrupted.

Note: head_out sweep is O(n_layers * n_heads * n_positions) forward passes
and can be slow on large models. Use the checkbox to opt in.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from app_state import get_config, get_model, render_sidebar_memory
from viz_interactive import patch_heatmap, head_patch_heatmap

st.set_page_config(page_title="Patching — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Activation Patching")
st.caption(
    "Patch clean activations into the corrupted run to find which "
    "(layer, position) sites rescue the answer."
)

cfg = get_config()

# --- Sidebar config ---
with st.sidebar:
    st.subheader("Run Config")
    clean_prompt = st.text_input("Clean prompt", value=cfg.prompts.clean)
    corrupted_prompt = st.text_input("Corrupted prompt", value=cfg.prompts.corrupted)
    answer_tok = st.text_input("Answer token", value=cfg.tokens.answer)
    distractor_tok = st.text_input(
        "Distractor token (optional)",
        value=cfg.tokens.distractor or "",
        help="If set, uses logit-diff metric instead of raw logit.",
    )

    st.subheader("Sweeps")
    sweep_resid = st.checkbox("resid_pre", value=True)
    sweep_attn = st.checkbox("attn_out", value=True)
    sweep_mlp = st.checkbox("mlp_out", value=True)
    sweep_head = st.checkbox(
        "head_out (slow)",
        value=False,
        help="O(n_layers × n_heads × n_positions) forward passes.",
    )

# --- Controls ---
col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Run Patching", type="primary")
with col2:
    if st.button("Clear Results"):
        st.session_state.pop("patching_results", None)
        st.rerun()

selected_sweeps = [
    name for name, enabled in [
        ("resid_pre", sweep_resid),
        ("attn_out", sweep_attn),
        ("mlp_out", sweep_mlp),
        ("head_out", sweep_head),
    ] if enabled
]

if not selected_sweeps and run:
    st.warning("No sweeps selected.")
    run = False

# --- Run ---
if run:
    from model import tokenize, run_with_cache
    from patching import (
        make_logit_diff_metric,
        compare_clean_corrupted,
        patch_resid_pre,
        patch_attn_out,
        patch_mlp_out,
        patch_head_out,
    )

    n_sweeps = len(selected_sweeps)
    progress = st.progress(0, text="Setting up…")

    try:
        model = get_model()
        clean_tokens, clean_str = tokenize(model, clean_prompt, prepend_bos=cfg.model.prepend_bos)
        corrupt_toks, corrupt_str = tokenize(model, corrupted_prompt, prepend_bos=cfg.model.prepend_bos)

        distractor = distractor_tok.strip() or None
        metric_fn = make_logit_diff_metric(model, answer_tok, distractor)

        progress.progress(0.1, text="Running baseline comparison…")
        baseline = compare_clean_corrupted(model, clean_tokens, corrupt_toks, metric_fn)
        clean_cache = baseline["clean_cache"]

        n_shared = min(clean_tokens.shape[1], corrupt_toks.shape[1])
        shared_str = corrupt_str[:n_shared]

        sweep_results = {}
        sweep_fns = {
            "resid_pre": (patch_resid_pre, "Resid-pre patching"),
            "attn_out":  (patch_attn_out,  "Attn-out patching"),
            "mlp_out":   (patch_mlp_out,   "MLP-out patching"),
            "head_out":  (patch_head_out,  "Per-head patching"),
        }

        for i, sweep_name in enumerate(selected_sweeps):
            fn, title = sweep_fns[sweep_name]
            frac = 0.1 + 0.9 * (i / n_sweeps)
            progress.progress(frac, text=f"{title}…")
            scores = fn(model, corrupt_toks, clean_cache, metric_fn)
            sweep_results[sweep_name] = (scores, title, shared_str)

        progress.empty()

        st.session_state["patching_results"] = {
            "baseline": baseline,
            "sweep_results": sweep_results,
            "answer_tok": answer_tok,
            "clean_prompt": clean_prompt,
            "corrupted_prompt": corrupted_prompt,
        }
        st.success("Done.")

    except Exception as e:
        progress.empty()
        st.error(f"Error: {e}")
        st.exception(e)

# --- Display ---
if "patching_results" in st.session_state:
    r = st.session_state["patching_results"]
    baseline = r["baseline"]

    st.divider()
    st.subheader("Baseline Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Clean score", f"{baseline['clean_score']:.4f}")
    col2.metric("Corrupted score", f"{baseline['corrupted_score']:.4f}")
    col3.metric("Delta", f"{baseline['delta']:.4f}")

    st.divider()
    st.subheader("Patching Results")

    for sweep_name, (scores, title, shared_str) in r["sweep_results"].items():
        if sweep_name == "head_out":
            with st.expander(f"{title} (summed over positions)"):
                st.plotly_chart(
                    head_patch_heatmap(scores, title=title),
                    use_container_width=True,
                )
                # Top rescuing heads
                summed = scores.sum(dim=-1)
                n_heads = summed.shape[1]
                flat = summed.flatten()
                top_idx = flat.argsort(descending=True)[:10]
                rows = [
                    {
                        "Rank": rank + 1,
                        "Layer": idx.item() // n_heads,
                        "Head": idx.item() % n_heads,
                        "Rescue score (sum)": round(flat[idx].item(), 5),
                    }
                    for rank, idx in enumerate(top_idx)
                ]
                st.dataframe(rows, use_container_width=False, hide_index=True)
        else:
            with st.expander(title, expanded=True):
                st.plotly_chart(
                    patch_heatmap(scores, shared_str, title=title),
                    use_container_width=True,
                )
                # Top rescuing positions
                n_pos = scores.shape[1]
                flat = scores.flatten()
                top_idx = flat.argsort(descending=True)[:10]
                rows = [
                    {
                        "Rank": rank + 1,
                        "Layer": idx.item() // n_pos,
                        "Position": idx.item() % n_pos,
                        "Token": (
                            repr(shared_str[idx.item() % n_pos])
                            if idx.item() % n_pos < len(shared_str) else "?"
                        ),
                        "Rescue score": round(flat[idx].item(), 5),
                    }
                    for rank, idx in enumerate(top_idx)
                ]
                st.dataframe(rows, use_container_width=False, hide_index=True)
