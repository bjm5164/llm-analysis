"""Targeted intervention dashboard.

Pick a specific component (residual stream, attention head, MLP output, or
a single MLP neuron), choose source and target prompts from the library,
select positions in each, apply an intervention (zero ablation, mean ablation,
noise injection, or activation patching), and inspect the effect on:
  - final next-token logits
  - logit lens at the target layer
  - answer-token logit across all layers (optional — requires answer token)
  - residual stream L2 norm across all layers
"""

import torch
import streamlit as st

from app_state import get_config, get_model, render_sidebar_memory, prompt_selector, token_id_input
from model import tokenize
from patching import make_logit_diff_metric, targeted_intervention
from viz_interactive import (
    answer_logit_across_layers,
    attention_pattern_heatmap,
    logit_lens_at_layer,
    residual_norm_across_layers,
    topk_logits_comparison,
)

st.set_page_config(page_title="Targeted Intervention — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Targeted Intervention")
st.caption(
    "Apply zero ablation, mean ablation, noise injection, or activation patching "
    "to a single layer/head/neuron and inspect the effect on the residual stream and final logits."
)

cfg = get_config()

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
try:
    model = get_model()
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_mlp = model.cfg.d_mlp
    model_loaded = True
except Exception as _e:
    st.warning(f"Model not loaded yet ({_e}). Load it on the Model page first.")
    model_loaded = False
    n_layers, n_heads, d_mlp = 28, 16, 14336

# ---------------------------------------------------------------------------
# Prompt selection from library
# ---------------------------------------------------------------------------
st.subheader("Prompts")
col_target, col_source = st.columns(2, gap="large")

with col_target:
    target_text = prompt_selector(
        "tp_target_prompt",
        label="Target prompt (run through model)",
        allow_empty=False, sync_slot="a",
    )
    if model_loaded and target_text:
        _tt, _st_toks = tokenize(
            model, target_text, prepend_bos=cfg.model.prepend_bos,
        )
        st.caption(
            f"{_tt.shape[1]} tokens: "
            f"{' '.join(repr(t) for t in _st_toks)}"
        )

with col_source:
    source_text = prompt_selector(
        "tp_source_prompt",
        label="Source prompt (for patch intervention)",
        sync_slot="b",
    )
    if model_loaded and source_text:
        _ts, _ss_toks = tokenize(model, source_text, prepend_bos=cfg.model.prepend_bos)
        st.caption(f"{_ts.shape[1]} tokens: {' '.join(repr(t) for t in _ss_toks)}")

st.divider()

# ---------------------------------------------------------------------------
# Main layout: controls (left) | results (right)
# ---------------------------------------------------------------------------
col_ctrl, col_results = st.columns([1, 2], gap="large")

with col_ctrl:
    st.subheader("Intervention Setup")

    component = st.selectbox(
        "Component",
        ["resid_pre", "attn_head", "mlp_out", "mlp_neuron"],
        help=(
            "**resid_pre** — full residual stream before a layer  \n"
            "**attn_head** — single attention head output (hook_z)  \n"
            "**mlp_out** — full MLP output for a layer  \n"
            "**mlp_neuron** — single post-nonlinearity MLP neuron"
        ),
    )

    layer = st.slider("Layer", 0, max(0, n_layers - 1), value=n_layers // 2)

    head = None
    neuron = None
    if component == "attn_head":
        head = st.slider("Head", 0, max(0, n_heads - 1), value=0)
    elif component == "mlp_neuron":
        neuron = int(st.number_input(
            "Neuron index", min_value=0, max_value=max(0, d_mlp - 1), value=0, step=1
        ))

    intervention = st.selectbox(
        "Intervention",
        ["zero", "mean", "noise", "patch"],
        help=(
            "**zero** — set activation to 0  \n"
            "**mean** — replace with mean over sequence positions  \n"
            "**noise** — add Gaussian noise  \n"
            "**patch** — replace with activation from the source prompt"
        ),
    )

    noise_std = 1.0
    if intervention == "noise":
        noise_std = st.slider("Noise std", 0.01, 10.0, value=1.0, step=0.05)

    # Position selector for target prompt
    source_pos_idx = None
    if model_loaded and target_text:
        _tokens, _str_toks = tokenize(model, target_text, prepend_bos=cfg.model.prepend_bos)
        n_pos = _tokens.shape[1]
        pos_labels = [f"[{i}] {repr(t)}" for i, t in enumerate(_str_toks)]
        pos_idx = st.selectbox(
            "Target position",
            range(n_pos),
            index=n_pos - 1,
            format_func=lambda i: pos_labels[i],
        )
    else:
        pos_idx = int(st.number_input("Target position", min_value=0, value=0, step=1))

    # Source position selector (for patch intervention)
    if intervention == "patch":
        if not source_text:
            st.warning("Select a source prompt for patch intervention.")
        elif model_loaded:
            _src_tokens, _src_str_toks = tokenize(
                model, source_text, prepend_bos=cfg.model.prepend_bos,
            )
            _src_n_pos = _src_tokens.shape[1]
            _src_labels = [f"[{i}] {repr(t)}" for i, t in enumerate(_src_str_toks)]
            _default = min(pos_idx if model_loaded and target_text else 0, _src_n_pos - 1)
            source_pos_idx = st.selectbox(
                "Source position",
                range(_src_n_pos),
                index=_default,
                format_func=lambda i: _src_labels[i],
            )
        else:
            source_pos_idx = int(st.number_input("Source position", min_value=0, value=0, step=1))

    top_k = st.slider("Top-k tokens to display", 5, 25, value=10)

    st.divider()
    run = st.button("Run Intervention", type="primary", disabled=not model_loaded)
    if st.button("Clear Results"):
        st.session_state.pop("targeted_results", None)
        st.rerun()

# ---------------------------------------------------------------------------
# Run the intervention
# ---------------------------------------------------------------------------
if run:
    if not target_text:
        st.warning("Select a target prompt.")
        st.stop()
    if intervention == "patch" and not source_text:
        st.warning("Select a source prompt for patch intervention.")
        st.stop()

    progress = st.progress(0, text="Tokenizing...")
    try:
        tokens, str_toks = tokenize(model, target_text, prepend_bos=cfg.model.prepend_bos)

        source_cache = None
        if intervention == "patch":
            progress.progress(0.2, text="Caching source activations...")
            src_tokens, _ = tokenize(model, source_text, prepend_bos=cfg.model.prepend_bos)
            with torch.no_grad():
                _, source_cache = model.run_with_cache(src_tokens)

        progress.progress(0.4, text="Running intervention...")

        orig_logits, patched_logits, orig_cache, patched_cache = targeted_intervention(
            model=model,
            tokens=tokens,
            layer=layer,
            component=component,
            pos=pos_idx,
            intervention=intervention,
            head=head,
            neuron=neuron,
            noise_std=noise_std,
            source_cache=source_cache,
            source_pos=source_pos_idx,
        )

        progress.empty()

        st.session_state["targeted_results"] = dict(
            orig_logits=orig_logits,
            patched_logits=patched_logits,
            orig_cache=orig_cache,
            patched_cache=patched_cache,
            str_toks=str_toks,
            layer=layer,
            pos_idx=pos_idx,
            component=component,
            intervention=intervention,
            head=head,
            neuron=neuron,
            top_k=top_k,
        )
        st.success("Done.")

    except Exception as exc:
        progress.empty()
        st.error(f"Error: {exc}")
        st.exception(exc)

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "targeted_results" in st.session_state:
    r = st.session_state["targeted_results"]

    with col_results:
        # --- Header metrics ---
        m1, m2 = st.columns(2)
        m1.metric("Layer", r["layer"])
        comp_label = r["component"]
        if r["head"] is not None:
            comp_label += f" H{r['head']}"
        if r["neuron"] is not None:
            comp_label += f" N{r['neuron']}"
        m2.metric("Target", comp_label)

        if r.get("intervention") == "patch":
            st.caption("Patching: source → target")

        st.divider()

        k = r["top_k"]
        tok_label = repr(r["str_toks"][r["pos_idx"]])

        # --- Top-k logits comparison (always shown) ---
        st.subheader("Final logits (last position)")
        st.plotly_chart(
            topk_logits_comparison(
                model, r["orig_logits"], r["patched_logits"],
                pos=-1, k=k,
                title=(
                    f"Top-{k} next-token logits at last position"
                    f" — intervened at pos {r['pos_idx']} ({tok_label})"
                ),
            ),
            use_container_width=True,
        )

        # --- Logit lens at intervention layer (always shown) ---
        st.subheader(f"Logit lens at layer {r['layer']} (resid_post)")
        st.plotly_chart(
            logit_lens_at_layer(
                model, r["orig_cache"], r["patched_cache"],
                layer=r["layer"], pos=r["pos_idx"], k=k,
                title=(
                    f"Logit lens — L{r['layer']} resid_post"
                    f" at pos {r['pos_idx']} ({tok_label})"
                ),
            ),
            use_container_width=True,
        )

        # --- Answer-token tracking (optional) ---
        st.subheader("Per-token tracking (optional)")
        st.caption(
            "Enter an answer token to track its logit-lens score across all layers. "
            "Also enables a logit-difference metric in the header."
        )
        _tok_col1, _tok_col2 = st.columns(2)
        with _tok_col1:
            answer_id = token_id_input(
                model, "Answer token", key="tp_answer_tok",
            ) if model_loaded else None
        with _tok_col2:
            distractor_id = token_id_input(
                model, "Distractor (optional)", key="tp_distractor",
                help="If set, metric = logit(answer) − logit(distractor).",
            ) if model_loaded else None

        if answer_id is not None:
            # Show logit-diff metric
            metric_fn = make_logit_diff_metric(model, answer_id, distractor_id)
            m_orig = metric_fn(r["orig_logits"]).item()
            m_patched = metric_fn(r["patched_logits"]).item()
            delta = m_patched - m_orig

            answer_str = model.tokenizer.decode(answer_id)
            mc1, mc2 = st.columns(2)
            mc1.metric("Metric — original", f"{m_orig:.4f}")
            mc2.metric("Metric — patched", f"{m_patched:.4f}", delta=f"{delta:+.4f}")

            st.plotly_chart(
                answer_logit_across_layers(
                    model, r["orig_cache"], r["patched_cache"],
                    pos=r["pos_idx"], answer_token_id=answer_id,
                    title=f"Logit-lens score for {repr(answer_str)} at pos {r['pos_idx']}",
                ),
                use_container_width=True,
            )

        # --- Residual stream norm (always shown) ---
        st.subheader("Residual stream norm — all layers")
        st.plotly_chart(
            residual_norm_across_layers(
                model, r["orig_cache"], r["patched_cache"],
                pos=r["pos_idx"],
            ),
            use_container_width=True,
        )

        # --- Attention patterns (always shown) ---
        st.divider()
        st.subheader("Attention patterns")
        st.caption(
            "Before vs after intervention. "
            "Select any layer and head — defaults to the intervention target."
        )

        _ap_col1, _ap_col2 = st.columns(2)
        with _ap_col1:
            _ap_layer = st.slider(
                "Attn layer", 0, max(0, n_layers - 1),
                value=r["layer"],
                key="ap_layer",
            )
        with _ap_col2:
            _ap_head = st.slider(
                "Attn head", 0, max(0, n_heads - 1),
                value=r["head"] if r["head"] is not None else 0,
                key="ap_head",
            )

        st.markdown("**Before intervention**")
        try:
            st.plotly_chart(
                attention_pattern_heatmap(
                    r["orig_cache"], _ap_layer, _ap_head, r["str_toks"],
                    title=f"L{_ap_layer} H{_ap_head} — original",
                ),
                use_container_width=True,
            )
        except Exception as _e:
            st.warning(f"Cannot plot: {_e}")

        st.markdown("**After intervention**")
        try:
            st.plotly_chart(
                attention_pattern_heatmap(
                    r["patched_cache"], _ap_layer, _ap_head, r["str_toks"],
                    title=f"L{_ap_layer} H{_ap_head} — patched",
                ),
                use_container_width=True,
            )
        except Exception as _e:
            st.warning(f"Cannot plot: {_e}")
