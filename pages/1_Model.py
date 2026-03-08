"""Model loader and sanity check page."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st
import torch

from app_state import get_config, get_model, render_sidebar_memory

st.set_page_config(page_title="Model — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Model")

cfg = get_config()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Current Config")
    st.code(
        f"name:              {cfg.model.name}\n"
        f"dtype:             {cfg.model.dtype}\n"
        f"device:            {cfg.model.device or 'auto'}\n"
        f"trust_remote_code: {cfg.model.trust_remote_code}\n"
        f"prepend_bos:       {cfg.model.prepend_bos}\n"
        f"cache_strategy:    {cfg.model.cache_strategy}",
        language="yaml",
    )
    st.caption("Edit on the Home page.")

st.divider()
st.subheader("Load Model")
st.caption(
    "The model is pinned in GPU memory via `@st.cache_resource`. "
    "Re-loading only triggers if the model config changes."
)

if st.button("Load / Verify Model", type="primary"):
    with st.spinner(f"Loading {cfg.model.name}…"):
        try:
            model = get_model()
            mcfg = model.cfg
            st.session_state["model_loaded"] = True
            st.success(
                f"Loaded **{cfg.model.name}** — "
                f"layers={mcfg.n_layers}  d_model={mcfg.d_model}  "
                f"heads={mcfg.n_heads}  d_head={mcfg.d_head}  vocab={mcfg.d_vocab}"
            )
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.session_state["model_loaded"] = False

if st.session_state.get("model_loaded"):
    model = get_model()
    mcfg = model.cfg

    st.subheader("Architecture")
    cols = st.columns(5)
    cols[0].metric("Layers", mcfg.n_layers)
    cols[1].metric("d_model", mcfg.d_model)
    cols[2].metric("Heads", mcfg.n_heads)
    cols[3].metric("d_head", mcfg.d_head)
    cols[4].metric("Vocab", mcfg.d_vocab)
    if hasattr(mcfg, "n_key_value_heads") and mcfg.n_key_value_heads != mcfg.n_heads:
        st.info(f"GQA: n_kv_heads = {mcfg.n_key_value_heads}")

    st.divider()
    st.subheader("Sanity Check")

    prompt = st.text_input("Prompt", value=cfg.prompts.clean, key="sanity_prompt")
    top_n = st.slider("Top-N predictions", 5, 20, 10)

    if st.button("Run Sanity Check"):
        from model import run_with_cache

        with st.spinner("Running…"):
            tokens = model.to_tokens(prompt, prepend_bos=cfg.model.prepend_bos)
            logits, _ = run_with_cache(model, tokens, strategy="minimal")
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            top_vals, top_ids = probs.topk(top_n)

        rows = [
            {
                "Rank": i + 1,
                "Token": repr(model.tokenizer.decode(tid.item())),
                "Token ID": tid.item(),
                "Probability": f"{p.item():.5f}",
            }
            for i, (tid, p) in enumerate(zip(top_ids, top_vals))
        ]
        st.dataframe(rows, use_container_width=False, hide_index=True)

    st.divider()
    st.subheader("Answer Token Check")
    answer = st.text_input("Answer token", value=cfg.tokens.answer, key="answer_check")
    if st.button("Verify Answer Token"):
        try:
            answer_id = model.to_single_token(answer)
            answer_str = model.to_single_str_token(answer_id)
            st.success(f"`{repr(answer)}` → id={answer_id}, decoded={repr(answer_str)}")
        except Exception as e:
            st.error(f"Token error: {e} — is this a single token in the tokenizer?")
