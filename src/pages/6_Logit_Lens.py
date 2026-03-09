"""Logit lens + iterative greedy generation to track answer token emergence."""

import streamlit as st
import torch

from app_state import get_config, get_model, render_sidebar_memory, token_id_input
from model import answer_token_logit_lens, generate_next_token_greedy

st.set_page_config(page_title="Logit Lens — LLM Analysis", layout="wide")
render_sidebar_memory()

st.title("Logit Lens")
st.caption(
    "Track when an answer token first appears in the residual stream. "
    "Generate tokens one at a time and re-run logit lens after each step "
    "to see whether the answer direction is already present or needs more processing."
)

cfg = get_config()

try:
    model = get_model()
except Exception:
    st.warning("Load a model on the **Model** page first.")
    st.stop()

# --- Session state for iterative generation ---
if "ll_token_ids" not in st.session_state:
    st.session_state["ll_token_ids"] = None
if "ll_steps" not in st.session_state:
    st.session_state["ll_steps"] = []

# --- Inputs ---
col1, col2 = st.columns([3, 1])
with col1:
    ll_prompt = st.text_input("Prompt", value="", key="ll_prompt")
with col2:
    ll_answer_id = token_id_input(model, "Answer token", key="ll_answer")

# --- Controls ---
c1, c2, c3 = st.columns(3)

with c1:
    run_initial = st.button("Run Logit Lens", type="primary")
with c2:
    gen_step = st.button("Generate + Re-check")
with c3:
    if st.button("Reset"):
        st.session_state["ll_token_ids"] = None
        st.session_state["ll_steps"] = []
        st.rerun()


def _run_logit_lens(tokens: torch.Tensor, answer_id: int, label: str) -> dict:
    """Run logit lens and return a step record."""
    rows = answer_token_logit_lens(model, tokens, answer_id)
    final_layer = rows[-1]
    greedy_id, greedy_str, greedy_prob = generate_next_token_greedy(model, tokens)
    prompt_str = model.tokenizer.decode(tokens[0].tolist())
    return {
        "label": label,
        "prompt": prompt_str,
        "tokens": tokens,
        "lens_rows": rows,
        "final_rank": final_layer["rank"],
        "final_prob": final_layer["prob"],
        "greedy_id": greedy_id,
        "greedy_str": greedy_str,
        "greedy_prob": greedy_prob,
        "answer_id": answer_id,
    }


# --- Run initial logit lens on the raw prompt ---
if run_initial:
    if ll_answer_id is None:
        st.warning("Enter a valid answer token (ID or string).")
        st.stop()

    with st.spinner("Running logit lens on prompt…"):
        tokens = model.to_tokens(ll_prompt, prepend_bos=cfg.model.prepend_bos)
        step = _run_logit_lens(tokens, ll_answer_id, "Original prompt")

    st.session_state["ll_token_ids"] = tokens
    st.session_state["ll_steps"] = [step]
    st.rerun()

# --- Generate one token, append, re-run logit lens ---
if gen_step:
    tokens = st.session_state.get("ll_token_ids")
    if tokens is None:
        st.error("Run the initial logit lens first.")
        st.stop()

    prev_step = st.session_state["ll_steps"][-1]
    answer_id = prev_step["answer_id"]

    with st.spinner("Generating next token + re-running logit lens…"):
        # Greedy decode one token
        next_id, next_str, _ = generate_next_token_greedy(model, tokens)
        # Append to sequence
        new_tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], device=tokens.device, dtype=tokens.dtype)],
            dim=1,
        )
        step_num = len(st.session_state["ll_steps"])
        step = _run_logit_lens(
            new_tokens, answer_id, f"Step {step_num} (appended {repr(next_str)})"
        )

    st.session_state["ll_token_ids"] = new_tokens
    st.session_state["ll_steps"].append(step)
    st.rerun()

# --- Display all steps ---
steps = st.session_state.get("ll_steps", [])
if not steps:
    st.info("Enter a prompt and answer token, then click **Run Logit Lens**.")
    st.stop()

# Summary table across steps
st.subheader("Answer Token Tracking")
summary_rows = []
for s in steps:
    found = s["final_rank"] == 0
    summary_rows.append({
        "Step": s["label"],
        "Final-layer rank": s["final_rank"],
        "Final-layer prob": f"{s['final_prob']:.5f}",
        "Greedy prediction": repr(s["greedy_str"]),
        "Answer is top-1": "Yes" if found else "No",
    })
st.dataframe(summary_rows, use_container_width=True, hide_index=True)

# Current prompt tokens
latest = steps[-1]
st.text(f"Current sequence: {latest['prompt']}")

# Detailed logit lens for each step (expandable)
for s in steps:
    with st.expander(f"{s['label']} — rank={s['final_rank']}, prob={s['final_prob']:.5f}"):
        first_top5 = next((r["layer"] for r in s["lens_rows"] if r["rank"] < 5), None)
        if first_top5 is not None:
            st.info(f"Answer token first enters **top-5** at layer **{first_top5}**")
        else:
            st.warning("Answer token never reaches top-5 in any layer.")

        st.dataframe(
            [
                {
                    "Layer": r["layer"],
                    "Rank": r["rank"],
                    "Probability": f"{r['prob']:.5f}",
                    "Logit": f"{r['logit']:.3f}",
                }
                for r in s["lens_rows"]
            ],
            use_container_width=False,
            hide_index=True,
        )
