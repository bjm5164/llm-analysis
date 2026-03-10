"""Prompt Library.

Explore tokenization and next-token predictions, then save prompts
for use in DLA, Attention, and other analysis pages.

Two ways to build prompts:
  - **Text mode**: type text, it gets tokenized naturally.
  - **Token builder**: construct prompts one token at a time, choosing
    exact token boundaries (e.g. three tokens '"', ':', '"' instead of
    the merged '":"' the tokenizer would produce).
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
    normalize_token_str,
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
    st.info(
        "Load a model on the **Model** page to enable "
        "tokenization and scoring."
    )


# ---------------------------------------------------------------------------
# Shared chip renderer
# ---------------------------------------------------------------------------
def _render_chips(
    str_toks: list[str],
    tok_ids: list[int],
    prepend_bos: bool = False,
) -> None:
    """Render token chips with position indices."""
    n = len(str_toks)
    count_html = (
        f'<span style="font-size:0.85em;color:#555;font-weight:600">'
        f'{n} token{"s" if n != 1 else ""}</span>'
    )
    chips = []
    for i, (tok_str, tok_id) in enumerate(zip(str_toks, tok_ids)):
        is_bos = i == 0 and prepend_bos
        bg = "#e0e0e0" if is_bos else "#e8eaf6"
        color = "#888" if is_bos else "#1a237e"
        chips.append(
            f'<span title="pos {i} · id {tok_id}" style="'
            f"display:inline-block;margin:2px 3px;padding:3px 9px;"
            f"border-radius:4px;background:{bg};color:{color};"
            f'font-family:monospace;font-size:0.83em;white-space:pre;">'
            f'<sup style="color:#bbb;font-size:0.7em;margin-right:2px">'
            f"{i}</sup>"
            f"{html_mod.escape(repr(tok_str))}</span>"
        )
    st.html(
        f"<div style='margin-bottom:4px'>{count_html}</div>"
        f"<div style='line-height:2.3;padding:4px 0'>"
        + "".join(chips)
        + "</div>"
    )


# ===================================================================
# Tab layout: Text Explorer | Token Builder
# ===================================================================
tab_text, tab_builder = st.tabs(["Text Explorer", "Token Builder"])


# ---------------------------------------------------------------------------
# Text Explorer (original functionality)
# ---------------------------------------------------------------------------
with tab_text:
    st.caption(
        "Type text — it gets tokenized naturally. Good for clean prompts."
    )

    col_bos, col_topk, _ = st.columns([1, 1, 2])
    prepend_bos = col_bos.checkbox(
        "Prepend BOS",
        value=cfg.model.prepend_bos,
        key="explorer_bos",
    )
    top_k = col_topk.slider(
        "Top-K predictions", 5, 25, 10, key="explorer_topk",
    )

    prompt = st.text_area(
        "Prompt",
        height=100,
        label_visibility="collapsed",
        placeholder="Type a prompt...",
        key="explorer_prompt",
    )

    if prompt and model_loaded:
        try:
            toks, str_toks = tokenize(
                model, prompt, prepend_bos=prepend_bos,
            )
            _render_chips(str_toks, toks[0].tolist(), prepend_bos)
        except Exception as exc:
            st.error(f"Tokenization error: {exc}")

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
                save_prompt(save_label.strip(), prompt, token_ids=None)
                st.success(f"Saved as **{save_label.strip()}**")
                st.rerun()

    # --- Next-token scoring ---
    if model_loaded and prompt and st.button("Score next token"):
        with st.spinner("Running..."):
            tokens, _ = tokenize(model, prompt, prepend_bos=prepend_bos)
            logits, _ = run_with_cache(
                model, tokens, strategy="minimal",
            )
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
# Token Builder — construct prompts token by token
# ---------------------------------------------------------------------------
_BUILDER_KEY = "builder_token_ids"

with tab_builder:
    if not model_loaded:
        st.warning("Load a model first to use the token builder.")
        st.stop()

    st.caption(
        "Build a prompt with exact token boundaries. "
        "Useful for studying constrained decoding where `'\":\"'` "
        "(one token) differs from `'\"'`, `':'`, `'\"'` (three tokens)."
    )

    # Initialise builder state
    if _BUILDER_KEY not in st.session_state:
        st.session_state[_BUILDER_KEY] = []

    builder_ids: list[int] = st.session_state[_BUILDER_KEY]

    # --- Current sequence display ---
    if builder_ids:
        str_toks = [model.tokenizer.decode(t) for t in builder_ids]
        _render_chips(str_toks, builder_ids)

        # Decoded text preview
        full_text = "".join(str_toks)
        st.code(full_text, language=None)
    else:
        st.info("Empty — add tokens below.")

    # --- Add tokens: three methods side by side ---
    st.markdown("##### Add tokens")
    col_text, col_search, col_id = st.columns(3, gap="medium")

    # 1. Add by text (natural tokenization)
    with col_text:
        st.markdown("**By text**")
        add_text = st.text_input(
            "Text to tokenize and append",
            key="builder_add_text",
            placeholder='e.g. What is 1+1?',
            label_visibility="collapsed",
        )
        if st.button(
            "Add text", key="builder_btn_text", disabled=not add_text,
        ):
            ids = model.tokenizer.encode(
                add_text, add_special_tokens=False,
            )
            if ids:
                builder_ids.extend(ids)
                decoded = [
                    model.tokenizer.decode(t) for t in ids
                ]
                st.caption(
                    f"Added {len(ids)} token(s): "
                    + ", ".join(
                        f"`{repr(d)}` (#{t})"
                        for t, d in zip(ids, decoded)
                    )
                )
                st.rerun()

    # 2. Add by token string search
    with col_search:
        st.markdown("**By token search**")
        search_str = st.text_input(
            "Search for token",
            key="builder_search_str",
            placeholder='e.g. " or :" or \\n',
            label_visibility="collapsed",
        )

        if search_str:
            s = normalize_token_str(search_str)
            # Show how this string tokenizes
            natural_ids = model.tokenizer.encode(
                s, add_special_tokens=False,
            )

            if len(natural_ids) == 1:
                tid = natural_ids[0]
                decoded = model.tokenizer.decode(tid)
                st.caption(
                    f"Single token: `{repr(decoded)}` → **#{tid}**"
                )
                if st.button(
                    f"Add #{tid}", key="builder_add_search_single",
                ):
                    builder_ids.append(tid)
                    st.rerun()
            elif len(natural_ids) > 1:
                st.caption(
                    f"`{repr(s)}` tokenizes as "
                    f"{len(natural_ids)} tokens:"
                )
                for tid in natural_ids:
                    decoded = model.tokenizer.decode(tid)
                    bcol1, bcol2 = st.columns([3, 1])
                    bcol1.caption(
                        f"`{repr(decoded)}` → **#{tid}**"
                    )
                    if bcol2.button(
                        f"Add", key=f"builder_add_search_{tid}",
                    ):
                        builder_ids.append(tid)
                        st.rerun()

                # Option to add all at once
                if st.button(
                    f"Add all {len(natural_ids)} tokens",
                    key="builder_add_search_all",
                ):
                    builder_ids.extend(natural_ids)
                    st.rerun()

    # 3. Add by token ID
    with col_id:
        st.markdown("**By token ID**")
        add_id_str = st.text_input(
            "Token ID(s)",
            key="builder_add_id",
            placeholder="e.g. 220 or 220,362,220",
            label_visibility="collapsed",
        )
        if st.button(
            "Add ID(s)", key="builder_btn_id", disabled=not add_id_str,
        ):
            added = []
            for part in add_id_str.split(","):
                part = part.strip()
                if not part:
                    continue
                if part.startswith("#"):
                    part = part[1:].strip()
                if not part.isdigit():
                    st.warning(f"Skipping non-integer: {part}")
                    continue
                tid = int(part)
                if tid >= model.cfg.d_vocab:
                    st.warning(
                        f"ID {tid} out of range "
                        f"(vocab size {model.cfg.d_vocab})"
                    )
                    continue
                builder_ids.append(tid)
                added.append(tid)
            if added:
                decoded = [
                    model.tokenizer.decode(t) for t in added
                ]
                st.caption(
                    f"Added: "
                    + ", ".join(
                        f"`{repr(d)}` (#{t})"
                        for t, d in zip(added, decoded)
                    )
                )
                st.rerun()

    # --- Edit sequence ---
    if builder_ids:
        st.markdown("##### Edit sequence")
        edit_cols = st.columns([1, 1, 1, 3])
        with edit_cols[0]:
            if st.button("Undo last", key="builder_undo"):
                builder_ids.pop()
                st.rerun()
        with edit_cols[1]:
            if st.button("Clear all", key="builder_clear"):
                st.session_state[_BUILDER_KEY] = []
                st.rerun()
        with edit_cols[2]:
            remove_pos = st.number_input(
                "Remove at position",
                min_value=0,
                max_value=max(0, len(builder_ids) - 1),
                value=max(0, len(builder_ids) - 1),
                step=1,
                key="builder_remove_pos",
                label_visibility="collapsed",
            )
        with edit_cols[3]:
            if st.button(
                f"Remove pos {remove_pos}", key="builder_remove_btn",
            ):
                if 0 <= remove_pos < len(builder_ids):
                    builder_ids.pop(remove_pos)
                    st.rerun()

        # --- Save builder prompt ---
        st.markdown("##### Save")
        col_blabel, col_bsave = st.columns([3, 1])
        with col_blabel:
            builder_label = st.text_input(
                "Label",
                placeholder="e.g. schema-split-tokens",
                key="builder_save_label",
                label_visibility="collapsed",
            )
        with col_bsave:
            if st.button("Save Prompt", type="primary",
                         key="builder_save_btn"):
                if not builder_label or not builder_label.strip():
                    st.warning("Enter a label.")
                else:
                    str_toks = [
                        model.tokenizer.decode(t) for t in builder_ids
                    ]
                    text = "".join(str_toks)
                    save_prompt(
                        builder_label.strip(),
                        text,
                        token_ids=list(builder_ids),
                    )
                    st.success(
                        f"Saved **{builder_label.strip()}** "
                        f"({len(builder_ids)} tokens)"
                    )
                    st.rerun()

        # --- Next-token scoring ---
        st.markdown("##### Score next token")
        builder_topk = st.slider(
            "Top-K predictions", 5, 25, 10, key="builder_topk",
        )
        if st.button(
            "Score next token",
            key="builder_score_btn",
            disabled=not builder_ids,
        ):
            with st.spinner("Running..."):
                tokens = torch.tensor([builder_ids], device=model.cfg.device)
                logits, _ = run_with_cache(
                    model, tokens, strategy="minimal",
                )
                probs = torch.softmax(logits[0, -1], dim=-1)
                top_p, top_ids = probs.topk(builder_topk)

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

    for label, entry in list(prompts.items()):
        text = entry["text"] if isinstance(entry, dict) else entry
        token_ids = (
            entry.get("token_ids")
            if isinstance(entry, dict)
            else None
        )
        with st.container(border=True):
            col_info, col_del = st.columns([5, 1])
            with col_info:
                badge = ""
                if token_ids is not None:
                    badge = (
                        f" · *{len(token_ids)} tokens "
                        f"(token-level)*"
                    )
                st.markdown(f"**{label}**{badge}")
                st.code(text, language=None)
                if model_loaded and token_ids is not None:
                    str_toks = [
                        model.tokenizer.decode(t) for t in token_ids
                    ]
                    _render_chips(str_toks, token_ids)
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
