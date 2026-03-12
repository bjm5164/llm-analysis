"""Patch Screen — sweep an intervention across many components at once.

Configure what to screen (all heads in a layer, heads across layers,
MLP outputs across layers, neurons within a single MLP, etc.), pick an
intervention type, and see which components matter most for a given metric.
"""

import torch
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from app_state import (
    get_config, get_model, render_sidebar_memory,
    prompt_selector, prompt_tokenize, token_id_input,
)
from patching import make_logit_diff_metric

st.set_page_config(page_title="Patch Screen — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Patch Screen")
st.caption(
    "Sweep an intervention (zero / mean / noise / patch) across many components "
    "and rank them by their effect on a metric."
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
# Prompt selection
# ---------------------------------------------------------------------------
st.subheader("Prompts")
col_target, col_source = st.columns(2, gap="large")

with col_target:
    target_text = prompt_selector(
        "ps_target_prompt",
        label="Target prompt (run through model)",
        allow_empty=False, sync_slot="a",
    )
    if model_loaded and target_text:
        _tt, _st_toks = prompt_tokenize(
            model, "ps_target_prompt", cfg.model.prepend_bos,
        )
        st.caption(
            f"{_tt.shape[1]} tokens: "
            f"{' '.join(repr(t) for t in _st_toks)}"
        )

with col_source:
    source_text = prompt_selector(
        "ps_source_prompt",
        label="Source prompt (for patch intervention)",
        sync_slot="b",
    )
    if model_loaded and source_text:
        _ts, _ss_toks = prompt_tokenize(
            model, "ps_source_prompt", cfg.model.prepend_bos,
        )
        st.caption(
            f"{_ts.shape[1]} tokens: "
            f"{' '.join(repr(t) for t in _ss_toks)}"
        )

st.divider()

# ---------------------------------------------------------------------------
# Screen configuration
# ---------------------------------------------------------------------------
col_ctrl, col_results = st.columns([1, 2], gap="large")

with col_ctrl:
    st.subheader("Screen Setup")

    screen_mode = st.selectbox(
        "Screen mode",
        [
            "Heads in layer",
            "Heads across layers",
            "MLP out across layers",
            "Neurons in MLP layer",
            "resid_pre across layers",
        ],
        help=(
            "**Heads in layer** — all attention heads at a fixed layer  \n"
            "**Heads across layers** — all (layer, head) pairs in a range  \n"
            "**MLP out across layers** — full MLP output per layer  \n"
            "**Neurons in MLP layer** — individual neurons in one MLP  \n"
            "**resid_pre across layers** — residual stream before each layer"
        ),
    )

    # --- Layer / range config ---
    if screen_mode in ("Heads in layer", "Neurons in MLP layer"):
        screen_layer = st.slider(
            "Layer", 0, max(0, n_layers - 1),
            value=n_layers // 2, key="ps_screen_layer",
        )
    elif screen_mode in ("Heads across layers", "MLP out across layers", "resid_pre across layers"):
        layer_range = st.slider(
            "Layer range", 0, max(0, n_layers - 1),
            value=(0, n_layers - 1), key="ps_layer_range",
        )

    # --- Neuron range (can be huge, so allow sub-ranges) ---
    if screen_mode == "Neurons in MLP layer":
        neuron_range = st.slider(
            "Neuron range", 0, max(0, d_mlp - 1),
            value=(0, min(255, d_mlp - 1)), key="ps_neuron_range",
            help="d_mlp can be very large — select a sub-range to keep the screen tractable.",
        )

    # --- Intervention ---
    intervention = st.selectbox(
        "Intervention",
        ["zero", "mean", "noise", "patch"],
        key="ps_intervention",
        help=(
            "**zero** — set activation to 0  \n"
            "**mean** — replace with mean over positions  \n"
            "**noise** — add Gaussian noise  \n"
            "**patch** — replace with source prompt activation"
        ),
    )

    noise_std = 1.0
    noise_scale = 1.0
    if intervention == "noise":
        noise_std = st.slider("Noise std", 0.01, 10.0, value=1.0, step=0.05, key="ps_noise_std")
        noise_scale = st.slider("Noise magnitude", 0.01, 10.0, value=1.0, step=0.05, key="ps_noise_scale",
                                help="Multiplier applied to the noise vector after sampling")

    # --- Position selectors ---
    source_pos_idx = None
    if model_loaded and target_text:
        _tokens, _str_toks = prompt_tokenize(
            model, "ps_target_prompt", cfg.model.prepend_bos,
        )
        n_pos = _tokens.shape[1]
        pos_labels = [f"[{i}] {repr(t)}" for i, t in enumerate(_str_toks)]
        pos_idx = st.selectbox(
            "Target position",
            range(n_pos),
            index=n_pos - 1,
            format_func=lambda i: pos_labels[i],
            key="ps_target_pos",
        )
    else:
        pos_idx = int(st.number_input("Target position", min_value=0, value=0, step=1, key="ps_target_pos_num"))

    if intervention == "patch":
        if not source_text:
            st.warning("Select a source prompt for patch intervention.")
        elif model_loaded:
            _src_tokens, _src_str_toks = prompt_tokenize(
                model, "ps_source_prompt", cfg.model.prepend_bos,
            )
            _src_n_pos = _src_tokens.shape[1]
            _src_labels = [f"[{i}] {repr(t)}" for i, t in enumerate(_src_str_toks)]
            _default = min(pos_idx if model_loaded and target_text else 0, _src_n_pos - 1)
            source_pos_idx = st.selectbox(
                "Source position",
                range(_src_n_pos),
                index=_default,
                format_func=lambda i: _src_labels[i],
                key="ps_source_pos",
            )
        else:
            source_pos_idx = int(st.number_input("Source position", min_value=0, value=0, step=1, key="ps_source_pos_num"))

    # --- Metric ---
    st.divider()
    st.subheader("Metric")
    st.caption("The screen ranks components by their effect on this metric.")

    _m_col1, _m_col2 = st.columns(2)
    with _m_col1:
        answer_id = token_id_input(
            model, "Answer token", key="ps_answer_tok",
        ) if model_loaded else None
    with _m_col2:
        distractor_id = token_id_input(
            model, "Distractor (optional)", key="ps_distractor",
            help="If set, metric = logit(answer) − logit(distractor).",
        ) if model_loaded else None

    metric_mode = st.radio(
        "Metric type", ["logit_diff", "prob_delta"],
        format_func=lambda x: "Logit / logit-diff" if x == "logit_diff" else "Probability delta",
        key="ps_metric_mode",
        help="logit_diff uses raw logit (or logit difference if distractor set). "
             "prob_delta uses softmax probability change.",
    )

    top_k_display = st.slider("Top-k to highlight in table", 5, 50, value=15, key="ps_topk")

    st.divider()
    run = st.button("Run Screen", type="primary", disabled=not model_loaded)
    if st.button("Clear Results", key="ps_clear"):
        st.session_state.pop("patch_screen_results", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Helper: build the list of (component, layer, head, neuron) to screen
# ---------------------------------------------------------------------------
def _build_screen_jobs(mode):
    """Return a list of dicts with keys for targeted_intervention kwargs."""
    jobs = []
    if mode == "Heads in layer":
        for h in range(n_heads):
            jobs.append(dict(component="attn_head", layer=screen_layer, head=h, neuron=None))
    elif mode == "Heads across layers":
        lo, hi = layer_range
        for l in range(lo, hi + 1):
            for h in range(n_heads):
                jobs.append(dict(component="attn_head", layer=l, head=h, neuron=None))
    elif mode == "MLP out across layers":
        lo, hi = layer_range
        for l in range(lo, hi + 1):
            jobs.append(dict(component="mlp_out", layer=l, head=None, neuron=None))
    elif mode == "Neurons in MLP layer":
        lo, hi = neuron_range
        for n in range(lo, hi + 1):
            jobs.append(dict(component="mlp_neuron", layer=screen_layer, head=None, neuron=n))
    elif mode == "resid_pre across layers":
        lo, hi = layer_range
        for l in range(lo, hi + 1):
            jobs.append(dict(component="resid_pre", layer=l, head=None, neuron=None))
    return jobs


def _job_label(job):
    """Human-readable label for a screen job."""
    c = job["component"]
    l = job["layer"]
    if c == "attn_head":
        return f"L{l}H{job['head']}"
    elif c == "mlp_neuron":
        return f"L{l}N{job['neuron']}"
    elif c == "mlp_out":
        return f"L{l} mlp_out"
    elif c == "resid_pre":
        return f"L{l} resid_pre"
    return f"L{l} {c}"


# ---------------------------------------------------------------------------
# Run screen
# ---------------------------------------------------------------------------
if run:
    if not target_text:
        st.warning("Select a target prompt.")
        st.stop()
    if intervention == "patch" and not source_text:
        st.warning("Select a source prompt for patch intervention.")
        st.stop()
    if answer_id is None:
        st.warning("Enter a valid answer token for the metric.")
        st.stop()

    jobs = _build_screen_jobs(screen_mode)
    if not jobs:
        st.warning("No components to screen — check your configuration.")
        st.stop()

    progress = st.progress(0, text="Preparing...")

    try:
        tokens, str_toks = prompt_tokenize(
            model, "ps_target_prompt", cfg.model.prepend_bos,
        )

        # Source cache (for patch intervention)
        source_cache = None
        if intervention == "patch":
            progress.progress(0.05, text="Caching source activations...")
            src_tokens, _ = prompt_tokenize(
                model, "ps_source_prompt", cfg.model.prepend_bos,
            )
            with torch.no_grad():
                _, source_cache = model.run_with_cache(src_tokens)

        # Build metric
        if metric_mode == "logit_diff":
            metric_fn = make_logit_diff_metric(model, answer_id, distractor_id)
        else:
            from patching import make_prob_metric
            metric_fn = make_prob_metric(model, answer_id)

        # --- Run clean baseline once (with cache for mean/patch) ---
        progress.progress(0.1, text="Running clean baseline...")
        with torch.no_grad():
            orig_logits, orig_cache = model.run_with_cache(tokens)
        baseline_score = metric_fn(orig_logits).item()

        # --- Screen loop (one patched forward pass per job) ---
        import transformer_lens.utils as tl_utils
        from patching import _TARGETED_HOOK_MAP

        scores = []
        for i, job in enumerate(jobs):
            frac = 0.1 + 0.9 * (i / len(jobs))
            label = _job_label(job)
            progress.progress(frac, text=f"[{i+1}/{len(jobs)}] {label}")

            comp = job["component"]
            hook_name = tl_utils.get_act_name(_TARGETED_HOOK_MAP[comp], job["layer"])
            clean_act = orig_cache[hook_name]
            _head = job["head"]
            _neuron = job["neuron"]
            _src_pos = source_pos_idx if source_pos_idx is not None else pos_idx

            def _make_hook(comp, clean_act, hook_name, _head, _neuron, _src_pos):
                def hook_fn(activation, hook):
                    act = activation.clone()
                    if comp == "attn_head":
                        if intervention == "zero":
                            act[:, pos_idx, _head, :] = 0.0
                        elif intervention == "mean":
                            act[:, pos_idx, _head, :] = clean_act[:, :, _head, :].mean(dim=1)
                        elif intervention == "noise":
                            act[:, pos_idx, _head, :] += torch.randn_like(act[:, pos_idx, _head, :]) * noise_std * noise_scale
                        elif intervention == "patch":
                            act[:, pos_idx, _head, :] = source_cache[hook.name][:, _src_pos, _head, :]
                    elif comp == "mlp_neuron":
                        if intervention == "zero":
                            act[:, pos_idx, _neuron] = 0.0
                        elif intervention == "mean":
                            act[:, pos_idx, _neuron] = clean_act[:, :, _neuron].mean(dim=1)
                        elif intervention == "noise":
                            act[:, pos_idx, _neuron] += torch.randn(act.shape[0], device=act.device) * noise_std * noise_scale
                        elif intervention == "patch":
                            act[:, pos_idx, _neuron] = source_cache[hook.name][:, _src_pos, _neuron]
                    else:
                        if intervention == "zero":
                            act[:, pos_idx, :] = 0.0
                        elif intervention == "mean":
                            act[:, pos_idx, :] = clean_act.mean(dim=1)
                        elif intervention == "noise":
                            act[:, pos_idx, :] += torch.randn_like(act[:, pos_idx, :]) * noise_std * noise_scale
                        elif intervention == "patch":
                            act[:, pos_idx, :] = source_cache[hook.name][:, _src_pos, :]
                    return act
                return hook_fn

            hook_fn = _make_hook(comp, clean_act, hook_name, _head, _neuron, _src_pos)

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, hook_fn)],
                )
            score = metric_fn(patched_logits).item()
            scores.append(score)

        progress.empty()

        st.session_state["patch_screen_results"] = dict(
            screen_mode=screen_mode,
            jobs=jobs,
            scores=scores,
            baseline=baseline_score,
            metric_mode=metric_mode,
            answer_id=answer_id,
            distractor_id=distractor_id,
            intervention=intervention,
            pos_idx=pos_idx,
            top_k=top_k_display,
        )
        st.success(f"Done — screened {len(jobs)} components.")

    except Exception as exc:
        progress.empty()
        st.error(f"Error: {exc}")
        st.exception(exc)


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if "patch_screen_results" in st.session_state:
    r = st.session_state["patch_screen_results"]
    jobs = r["jobs"]
    scores = np.array(r["scores"])
    baseline = r["baseline"]
    deltas = scores - baseline
    labels = [_job_label(j) for j in jobs]

    with col_results:
        st.subheader("Screen Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline metric", f"{baseline:.4f}")
        m2.metric("Components screened", len(jobs))
        m3.metric("Intervention", r["intervention"])

        st.divider()

        # --- Choose visualization based on screen mode ---
        mode = r["screen_mode"]

        if mode == "Heads across layers":
            # Reshape into (n_layers_range, n_heads) heatmap
            layer_set = sorted(set(j["layer"] for j in jobs))
            n_l = len(layer_set)
            heatmap_data = deltas.reshape(n_l, n_heads)

            fig = px.imshow(
                heatmap_data,
                x=[f"H{h}" for h in range(n_heads)],
                y=[f"L{l}" for l in layer_set],
                color_continuous_scale="RdBu",
                color_continuous_midpoint=0,
                aspect="auto",
                title="Metric delta per (layer, head)",
                labels=dict(color="Δ metric"),
            )
            fig.update_layout(height=max(300, n_l * 25 + 100))
            st.plotly_chart(fig, use_container_width=True)

        else:
            # Bar chart for 1D screens
            sort_idx = np.argsort(np.abs(deltas))[::-1]
            sorted_labels = [labels[i] for i in sort_idx]
            sorted_deltas = deltas[sort_idx]

            colors = ["#d73027" if d < 0 else "#4575b4" for d in sorted_deltas]

            fig = go.Figure(go.Bar(
                x=sorted_labels,
                y=sorted_deltas,
                marker_color=colors,
            ))
            fig.update_layout(
                title="Metric delta (sorted by |Δ|)",
                xaxis_title="Component",
                yaxis_title="Δ metric (patched − baseline)",
                height=450,
            )
            # For large screens, limit the x-axis to top-k
            if len(sorted_labels) > 60:
                fig.update_layout(
                    xaxis=dict(range=[-0.5, 59.5]),
                    title=f"Metric delta — showing top 60 of {len(sorted_labels)} (sorted by |Δ|)",
                )
            st.plotly_chart(fig, use_container_width=True)

        # --- Ranked table ---
        st.divider()
        st.subheader("Ranked components")

        rank_idx = np.argsort(np.abs(deltas))[::-1]
        k = min(r["top_k"], len(jobs))
        table_rows = []
        for rank, idx in enumerate(rank_idx[:k]):
            j = jobs[idx]
            table_rows.append({
                "Rank": rank + 1,
                "Component": labels[idx],
                "Layer": j["layer"],
                "Head": j["head"] if j["head"] is not None else "—",
                "Neuron": j["neuron"] if j["neuron"] is not None else "—",
                "Patched metric": round(scores[idx], 5),
                "Δ metric": round(deltas[idx], 5),
            })
        st.dataframe(table_rows, use_container_width=False, hide_index=True)
