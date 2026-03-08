"""LLM Connectomics — Home page.

Displays current config and provides a live YAML editor.
Navigate to experiment pages via the sidebar.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from connectomics.app_state import get_config, apply_config_yaml, render_sidebar_memory
from connectomics.config import load_config, DEFAULT_CONFIG

st.set_page_config(
    page_title="LLM Connectomics",
    page_icon="🔬",
    layout="wide",
)

render_sidebar_memory()

st.title("LLM Connectomics")
st.caption(
    "Mechanistic interpretability experiments for Qwen3 using TransformerLens. "
    "Edit config here; navigate experiments via the sidebar."
)

cfg = get_config()

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Model")
    st.code(
        f"name:     {cfg.model.name}\n"
        f"dtype:    {cfg.model.dtype}\n"
        f"strategy: {cfg.model.cache_strategy}",
        language="yaml",
    )
with col2:
    st.subheader("Task")
    st.code(
        f"clean:      {repr(cfg.prompts.clean)}\n"
        f"corrupted:  {repr(cfg.prompts.corrupted)}\n"
        f"answer:     {repr(cfg.tokens.answer)}\n"
        f"distractor: {repr(cfg.tokens.distractor)}",
        language="yaml",
    )
with col3:
    st.subheader("Corruption Sweep")
    variants = cfg.corruption_sweep.named_configs()
    if variants:
        lines = [f"{label}: inject={v.inject} count={v.count}" for label, v in variants]
        st.code("\n".join(lines), language="text")
    else:
        st.info("No sweep variants configured.")

st.divider()
st.subheader("Config Editor")
st.caption(
    "Edit YAML directly. Click **Apply** to update the session config. "
    "Changes clear any cached experiment results."
)

new_yaml = st.text_area(
    "config.yaml",
    value=st.session_state.get("config_yaml", ""),
    height=520,
    label_visibility="collapsed",
)

col_a, col_b, col_c = st.columns([1, 1, 6])
with col_a:
    if st.button("Apply", type="primary"):
        cfg_new, err = apply_config_yaml(new_yaml)
        if err:
            st.error(f"Config error: {err}")
        else:
            st.session_state["config"] = cfg_new
            st.session_state["config_yaml"] = new_yaml
            for key in ["baseline_results", "ood_results", "patching_results", "sweep_results"]:
                st.session_state.pop(key, None)
            st.success("Config applied. Stale experiment results cleared.")
            st.rerun()

with col_b:
    if st.button("Reset to file"):
        st.session_state["config"] = load_config(DEFAULT_CONFIG)
        with open(DEFAULT_CONFIG) as f:
            st.session_state["config_yaml"] = f.read()
        for key in ["baseline_results", "ood_results", "patching_results", "sweep_results"]:
            st.session_state.pop(key, None)
        st.rerun()
