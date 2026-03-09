"""Direct Logit Attribution.

Which components and heads write the answer token, and how does structured
schema enforcement redistribute that credit?

Workflow:
  1. Run Prompt A (baseline) and Prompt B (with schema / OOD suffix).
  2. Compare logit margins, component attribution, and per-head DLA.
  3. Identify heads whose contribution shifts most — these are candidates
     for information-routing changes under schema enforcement.
"""

import streamlit as st

from app_state import get_config, get_model, render_sidebar_memory, prompt_selector, token_id_input
from attribution import component_attribution, final_logit_margin, head_attribution
from model import run_with_cache, tokenize
from viz_interactive import (
    tokenization_table,
    component_attribution_bar,
    head_attribution_heatmap,
)

st.set_page_config(page_title="DLA — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Direct Logit Attribution")
st.caption(
    "Which components write the answer? How does schema enforcement "
    "redistribute credit across heads and MLPs?"
)

cfg = get_config()

# ---------------------------------------------------------------------------
# Prompt inputs (from saved library)
# ---------------------------------------------------------------------------
col_a, col_b = st.columns(2, gap="large")
with col_a:
    prompt_a = prompt_selector("dla_prompt_a", label="Prompt A (baseline)", allow_empty=False)
with col_b:
    prompt_b = prompt_selector("dla_prompt_b", label="Prompt B (schema / OOD)")

try:
    _model = get_model()
except Exception:
    _model = None

col_ans, col_dist, _ = st.columns([1, 1, 2])
with col_ans:
    if _model:
        answer_id = token_id_input(_model, "Answer token", key="dla_answer")
    else:
        st.text_input("Answer token", key="dla_answer", help="Load a model first to resolve tokens.")
        answer_id = None
with col_dist:
    if _model:
        distractor_id = token_id_input(
            _model, "Distractor (optional)", key="dla_distractor",
            help="If set, metrics include logit(answer) − logit(distractor).",
        )
    else:
        st.text_input("Distractor (optional)", key="dla_distractor")
        distractor_id = None

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
col1, col2 = st.columns([1, 5])
with col1:
    run = st.button("Run DLA", type="primary")
with col2:
    if st.button("Clear"):
        st.session_state.pop("dla_results", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_one(model, prompt: str, answer_id: int, strategy: str) -> dict:
    """Run a single prompt and collect DLA results."""
    tokens, str_tokens = tokenize(model, prompt, prepend_bos=cfg.model.prepend_bos)
    logits, cache = run_with_cache(model, tokens, strategy=strategy)
    margin = final_logit_margin(model, logits, answer_id)
    comp_attrs, comp_labels = component_attribution(model, cache, answer_id, pos=-1)
    head_attrs = head_attribution(model, cache, answer_id, pos=-1)

    answer_decoded = model.tokenizer.decode(answer_id)

    return {
        "tokens": tokens,
        "str_tokens": str_tokens,
        "margin": margin,
        "comp_attrs": comp_attrs,
        "comp_labels": comp_labels,
        "head_attrs": head_attrs,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "answer_id": answer_id,
        "answer_decoded": answer_decoded,
        "prepend_bos": cfg.model.prepend_bos,
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run:
    if not prompt_a or answer_id is None:
        st.warning("Prompt A and a valid answer token are required.")
        st.stop()
    with st.spinner("Running DLA…"):
        try:
            model = get_model()
            answer_str = model.tokenizer.decode(answer_id)
            strategy = cfg.model.cache_strategy
            results = {"A": _run_one(model, prompt_a, answer_id, strategy)}
            if prompt_b and prompt_b.strip():
                results["B"] = _run_one(model, prompt_b, answer_id, strategy)

            store = {"results": results, "answer_tok": answer_str}

            # Compute diffs when we have both prompts
            if "B" in results:
                head_diff = results["B"]["head_attrs"] - results["A"]["head_attrs"]
                comp_diff = results["B"]["comp_attrs"] - results["A"]["comp_attrs"]
                clean_scale = results["A"]["comp_attrs"].abs().sum().item()
                store["head_diff"] = head_diff
                store["comp_diff"] = comp_diff
                store["comp_diff_norm"] = comp_diff / max(clean_scale, 1e-8)

            st.session_state["dla_results"] = store
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
if "dla_results" not in st.session_state:
    st.stop()

r = st.session_state["dla_results"]
results = r["results"]
ans_repr = repr(r["answer_tok"])
is_comparison = "B" in results

# --- Token verification ---
st.divider()
_margin = results["A"]["margin"]
st.caption(
    f"Answer token: `{repr(r['answer_tok'])}` → "
    f"id **{_margin['answer_id']}** → decoded `{repr(_margin['answer_token'])}` "
    f"· prepend_bos={results['A'].get('prepend_bos', cfg.model.prepend_bos)}"
)

# --- Logit margin ---
st.subheader("Logit Margin")

if is_comparison:
    cols = st.columns(2)
    for col, (label, label_long) in zip(cols, [("A", "Baseline"), ("B", "Schema")]):
        m = results[label]["margin"]
        col.metric(f"{label_long} P(answer)", f"{m['prob']:.5f}")
        col.caption(f"logit = {m['logit']:.3f}  ·  rank = {m['rank']}")
else:
    m = results["A"]["margin"]
    c1, c2, c3 = st.columns(3)
    c1.metric("P(answer)", f"{m['prob']:.5f}")
    c2.metric("Answer logit", f"{m['logit']:.3f}")
    c3.metric("Rank", m["rank"])

# --- Tabs ---
if is_comparison:
    tab_diff, tab_comp, tab_heads_a, tab_heads_b, tab_toks = st.tabs([
        "Head Attribution Diff",
        "Component Attribution",
        "Baseline Heads",
        "Schema Heads",
        "Tokenization",
    ])
else:
    tab_heads_a, tab_comp, tab_topk, tab_toks = st.tabs([
        "Head Attribution",
        "Component Attribution",
        "Top Heads",
        "Tokenization",
    ])

# --- Head attribution diff (comparison only) ---
if is_comparison:
    with tab_diff:
        st.markdown(
            "Positive (red) = head contributes **more** toward the answer under schema "
            "enforcement. Negative (blue) = head contributes **less**. "
            "Large shifts indicate heads whose information routing changed."
        )
        st.plotly_chart(
            head_attribution_heatmap(
                r["head_diff"],
                title=f"Head DLA diff: schema − baseline ({ans_repr})",
            ),
            use_container_width=True,
        )

        # Top changed heads
        diff = r["head_diff"]
        n_heads = diff.shape[1]
        flat = diff.flatten()
        k = st.slider("Show top K changed heads", 5, 30, 15, key="dla_diff_topk")
        top_idx = flat.abs().argsort(descending=True)[:k]
        rows = []
        for rank, idx in enumerate(top_idx):
            li = idx.item() // n_heads
            hi = idx.item() % n_heads
            baseline_val = results["A"]["head_attrs"][li, hi].item()
            schema_val = results["B"]["head_attrs"][li, hi].item()
            rows.append({
                "Rank": rank + 1,
                "Layer": li,
                "Head": hi,
                "Baseline": round(baseline_val, 5),
                "Schema": round(schema_val, 5),
                "Diff": round(flat[idx].item(), 5),
            })
        st.dataframe(rows, use_container_width=False, hide_index=True)

# --- Component attribution ---
with tab_comp:
    if is_comparison:
        labels = results["A"]["comp_labels"]
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("**Baseline**")
            st.plotly_chart(
                component_attribution_bar(
                    results["A"]["comp_attrs"], labels, f"{ans_repr} (baseline)"
                ),
                use_container_width=True,
            )
        with col_r:
            st.markdown("**Schema**")
            st.plotly_chart(
                component_attribution_bar(
                    results["B"]["comp_attrs"], labels, f"{ans_repr} (schema)"
                ),
                use_container_width=True,
            )

        st.markdown(
            "**Difference** (schema − baseline). "
            "Normalised values are fractions of the baseline total attribution."
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
    else:
        st.plotly_chart(
            component_attribution_bar(
                results["A"]["comp_attrs"],
                results["A"]["comp_labels"],
                ans_repr,
            ),
            use_container_width=True,
        )

# --- Per-prompt head heatmaps ---
with tab_heads_a:
    st.plotly_chart(
        head_attribution_heatmap(
            results["A"]["head_attrs"],
            title=f"Baseline head DLA: {ans_repr}",
        ),
        use_container_width=True,
    )

if is_comparison:
    with tab_heads_b:
        st.plotly_chart(
            head_attribution_heatmap(
                results["B"]["head_attrs"],
                title=f"Schema head DLA: {ans_repr}",
            ),
            use_container_width=True,
        )

# --- Top heads (single-prompt only) ---
if not is_comparison:
    with tab_topk:
        ha = results["A"]["head_attrs"]
        n_heads = ha.shape[1]
        flat = ha.flatten()
        k = st.slider("Show top K heads", 5, 30, 15, key="dla_topk")
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

# --- Tokenization ---
with tab_toks:
    for label, label_long in [("A", "Baseline")] + ([("B", "Schema")] if is_comparison else []):
        res = results[label]
        st.markdown(f"**{label_long}**")
        st.plotly_chart(
            tokenization_table(res["str_tokens"], res["tokens"]),
            use_container_width=True,
        )
