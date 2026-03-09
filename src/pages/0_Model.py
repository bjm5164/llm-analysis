"""Model loader and sanity check page."""

import streamlit as st
import torch

from app_state import get_config, get_model, render_sidebar_memory, token_id_input
from model import run_with_cache

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

    prompt = st.text_input("Prompt", value="", key="sanity_prompt")
    top_n = st.slider("Top-N predictions", 5, 20, 10)

    if st.button("Run Sanity Check"):
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
    check_id = token_id_input(model, "Answer token", key="answer_check")

    st.divider()
    st.subheader("Token Lookup")
    st.caption(
        "Look up a token by ID, or search for all token IDs that decode to a given string. "
        "Useful when two tokens look identical but have different IDs (e.g. with/without leading space)."
    )
    col_by_id, col_by_str = st.columns(2, gap="large")

    with col_by_id:
        st.markdown("**By ID**")
        tok_id_input = st.text_input(
            "Token ID(s)", placeholder="e.g. 220 or 220,362",
            key="tok_lookup_id",
        )
        if tok_id_input and st.button("Lookup ID", key="btn_lookup_id"):
            for part in tok_id_input.split(","):
                part = part.strip()
                if not part.isdigit():
                    st.warning(f"Skipping non-integer: {part}")
                    continue
                tid = int(part)
                if tid >= mcfg.d_vocab:
                    st.warning(f"ID {tid} out of range (vocab size {mcfg.d_vocab})")
                    continue
                decoded = model.tokenizer.decode(tid)
                raw_bytes = decoded.encode("unicode_escape").decode("ascii")
                st.code(
                    f"id={tid}  repr={repr(decoded)}  bytes={raw_bytes}",
                    language=None,
                )

    with col_by_str:
        st.markdown("**By string**")
        tok_str_input = st.text_input(
            "Token string", placeholder="e.g. 2",
            key="tok_lookup_str",
        )
        if tok_str_input and st.button("Search", key="btn_lookup_str"):
            # Encode without special tokens to find the "natural" token
            natural_ids = model.tokenizer.encode(tok_str_input, add_special_tokens=False)
            st.markdown(f"Tokenizer encodes `{repr(tok_str_input)}` as: **{natural_ids}**")
            for tid in natural_ids:
                decoded = model.tokenizer.decode(tid)
                st.code(f"id={tid}  repr={repr(decoded)}", language=None)

            # Also try with a leading space — common source of confusion
            with_space = " " + tok_str_input
            space_ids = model.tokenizer.encode(with_space, add_special_tokens=False)
            if space_ids != natural_ids:
                st.markdown(f"With leading space `{repr(with_space)}`: **{space_ids}**")
                for tid in space_ids:
                    decoded = model.tokenizer.decode(tid)
                    st.code(f"id={tid}  repr={repr(decoded)}", language=None)

