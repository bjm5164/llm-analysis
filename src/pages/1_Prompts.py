"""Prompt Library.

Explore tokenization and next-token predictions, then save prompts
for use in DLA, Attention, and other analysis pages.
"""

import html as html_mod

import torch
import streamlit as st

from app_state import (
    get_config,
    get_model,
    render_sidebar_memory,
    get_saved_prompts,
    save_prompt,
    delete_prompt,
    clear_prompts,
    export_prompts_json,
    import_prompts_json,
)
from model import run_with_cache, tokenize

st.set_page_config(page_title="Prompts — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Prompt Library")
st.caption("Explore prompts, then save them for use across analysis pages.")

cfg = get_config()

try:
    model = get_model()
    model_loaded = True
except Exception:
    model_loaded = False
    st.info("Load a model on the **Model** page to enable tokenization and scoring.")


def _chips(text: str, prepend_bos: bool) -> None:
    """Render token chips with position indices."""
    if not model_loaded or not text.strip():
        return
    try:
        toks, str_toks = tokenize(model, text, prepend_bos=prepend_bos)
        ids = toks[0].tolist()
        n = len(str_toks)
        count_html = (
            f'<span style="font-size:0.85em;color:#555;font-weight:600">'
            f'{n} token{"s" if n != 1 else ""}</span>'
        )
        chips = []
        for i, (tok_str, tok_id) in enumerate(zip(str_toks, ids)):
            is_bos = i == 0 and prepend_bos
            bg = "#e0e0e0" if is_bos else "#e8eaf6"
            color = "#888" if is_bos else "#1a237e"
            chips.append(
                f'<span title="pos {i} · id {tok_id}" style="'
                f"display:inline-block;margin:2px 3px;padding:3px 9px;"
                f"border-radius:4px;background:{bg};color:{color};"
                f'font-family:monospace;font-size:0.83em;white-space:pre;">'
                f'<sup style="color:#bbb;font-size:0.7em;margin-right:2px">{i}</sup>'
                f"{html_mod.escape(repr(tok_str))}</span>"
            )
        st.html(
            f"<div style='margin-bottom:4px'>{count_html}</div>"
            f"<div style='line-height:2.3;padding:4px 0'>"
            + "".join(chips)
            + "</div>"
        )
    except Exception as exc:
        st.error(f"Tokenization error: {exc}")


# ---------------------------------------------------------------------------
# Explorer
# ---------------------------------------------------------------------------
st.subheader("Explorer")

col_bos, col_topk, _ = st.columns([1, 1, 2])
prepend_bos = col_bos.checkbox(
    "Prepend BOS",
    value=cfg.model.prepend_bos,
    key="explorer_bos",
)
top_k = col_topk.slider("Top-K tokens", 5, 25, 10, key="explorer_topk")

prompt = st.text_area(
    "Prompt",
    height=100,
    label_visibility="collapsed",
    placeholder="Type a prompt...",
    key="explorer_prompt",
)

if prompt:
    _chips(prompt, prepend_bos)

# --- Save prompt ---
col_label, col_save = st.columns([3, 1])
with col_label:
    save_label = st.text_input(
        "Label",
        placeholder="e.g. clean, schema-json, baseline-1+1",
        key="explorer_save_label",
        label_visibility="collapsed",
    )
with col_save:
    if st.button("Save Prompt", type="primary", disabled=not prompt):
        if not save_label or not save_label.strip():
            st.warning("Enter a label for the prompt.")
        else:
            save_prompt(save_label.strip(), prompt)
            st.success(f"Saved as **{save_label.strip()}**")
            st.rerun()

# --- Next-token scoring ---
if model_loaded and prompt and st.button("Score next token"):
    with st.spinner("Running..."):
        tokens, _ = tokenize(model, prompt, prepend_bos=prepend_bos)
        logits, _ = run_with_cache(model, tokens, strategy="minimal")
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_p, top_ids = probs.topk(top_k)

    rows = [
        {
            "Rank": i + 1,
            "Token": repr(model.tokenizer.decode(tid.item())),
            "Token ID": tid.item(),
            "Probability": f"{p.item():.5f}",
            "Logit": f"{logits[0, -1, tid].item():.3f}",
        }
        for i, (tid, p) in enumerate(zip(top_ids, top_p))
    ]
    st.dataframe(rows, use_container_width=False, hide_index=True)

# ---------------------------------------------------------------------------
# Saved Prompts
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Saved Prompts")

prompts = get_saved_prompts()

if not prompts:
    st.info("No saved prompts yet. Use the explorer above to add some.")
else:
    # Clear all button
    if st.button("Clear All Prompts"):
        clear_prompts()
        st.rerun()

    for label, text in list(prompts.items()):
        with st.container(border=True):
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(f"**{label}**")
                st.code(text, language=None)
            with col_del:
                if st.button("Delete", key=f"del_{label}"):
                    delete_prompt(label)
                    st.rerun()

# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Export / Import")

col_export, col_import = st.columns(2, gap="large")

with col_export:
    st.markdown("**Export**")
    if get_saved_prompts():
        st.download_button(
            "Download prompts.json",
            data=export_prompts_json(),
            file_name="prompts.json",
            mime="application/json",
        )
    else:
        st.caption("Nothing to export.")

with col_import:
    st.markdown("**Import**")
    uploaded = st.file_uploader(
        "Upload prompts JSON",
        type=["json"],
        key="import_prompts_file",
        label_visibility="collapsed",
    )
    merge = st.checkbox("Merge with existing prompts", value=True)
    if uploaded is not None:
        if st.button("Import", type="primary"):
            try:
                raw = uploaded.read().decode("utf-8")
                n = import_prompts_json(raw, merge=merge)
                st.success(f"Imported {n} prompt(s).")
                st.rerun()
            except Exception as exc:
                st.error(f"Import failed: {exc}")
