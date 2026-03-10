"""Shared state and utilities for the Streamlit app.

Provides:
  - @st.cache_resource model loading (pinned in GPU memory, keyed by config)
  - Config management via session state with live YAML editing
  - Sidebar GPU memory widget with clear-cache button
"""

import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch

# Ensure src/ is importable when imported from pages/
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_config, Config, DEFAULT_CONFIG


@st.cache_resource
def _cached_model(name: str, dtype: str, device: str, trust_remote_code: bool):
    """Load model once and pin in GPU memory. Re-loads only if these args change."""
    from config import ModelConfig
    from model import load_model
    cfg = ModelConfig(
        name=name, dtype=dtype, device=device or None, trust_remote_code=trust_remote_code
    )
    return load_model(cfg)


def get_model():
    """Return the cached model for the current session config."""
    cfg = get_config()
    return _cached_model(
        cfg.model.name,
        cfg.model.dtype,
        cfg.model.device or "",
        cfg.model.trust_remote_code,
    )


def get_config() -> Config:
    """Return the current Config, loading from disk on first call."""
    if "config" not in st.session_state:
        st.session_state["config"] = load_config(DEFAULT_CONFIG)
        with open(DEFAULT_CONFIG, encoding="utf-8") as f:
            st.session_state["config_yaml"] = f.read()
    return st.session_state["config"]


def apply_config_yaml(yaml_str: str) -> tuple[Config | None, str | None]:
    """Parse a YAML string into a Config. Returns (Config, None) or (None, error_msg)."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_str)
            tmp = f.name
        cfg = load_config(tmp)
        return cfg, None
    except SystemExit:
        return None, "Invalid config — check required fields (model.name)."
    except Exception as e:
        return None, str(e)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def memory_stats() -> dict:
    """Current GPU memory stats in GB, or empty dict if no CUDA.

    Uses mem_get_info() for free/total — this reflects actual driver-level
    availability across ALL processes, not just this one.
    """
    if not torch.cuda.is_available():
        return {}
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "total": total_bytes / 1e9,
        "free": free_bytes / 1e9,
    }


def hard_reset():
    """Evict the cached model and all experiment results, then free GPU memory.

    - st.cache_resource.clear() drops the model reference so PyTorch can free weights.
    - Session-state experiment tensors are deleted before gc/empty_cache.
    """
    for key in ["baseline_results", "ood_results", "patching_results", "sweep_results", "targeted_results"]:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_resource.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Prompt library — shared across all pages, persisted to disk as JSON
#
# Each entry is: {label: {"text": str, "token_ids": list[int] | None}}
#   - text:      human-readable concatenation of decoded tokens
#   - token_ids:  authoritative token sequence (if built token-by-token)
#                 None for prompts entered as plain text
#
# Old format ({label: str}) is auto-migrated on load.
# ---------------------------------------------------------------------------
_PROMPT_LIBRARY_KEY = "prompt_library"
_PROMPTS_FILE = Path(__file__).resolve().parent / "data" / "prompts.json"


def _migrate_entry(value) -> dict:
    """Normalise a library entry to {"text": str, "token_ids": ...}."""
    if isinstance(value, str):
        return {"text": value, "token_ids": None}
    if isinstance(value, dict) and "text" in value:
        return value
    return {"text": str(value), "token_ids": None}


def _load_prompts_from_disk() -> dict[str, dict]:
    """Load prompts from JSON file, returning empty dict if missing."""
    if _PROMPTS_FILE.exists():
        try:
            data = json.loads(
                _PROMPTS_FILE.read_text(encoding="utf-8"),
            )
            if isinstance(data, dict):
                return {k: _migrate_entry(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_prompts_to_disk(prompts: dict[str, dict]) -> None:
    """Persist current prompt library to JSON."""
    _PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PROMPTS_FILE.write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_saved_prompts() -> dict[str, dict]:
    """Return the prompt library {label: {"text", "token_ids"}}."""
    if _PROMPT_LIBRARY_KEY not in st.session_state:
        st.session_state[_PROMPT_LIBRARY_KEY] = _load_prompts_from_disk()
    lib = st.session_state[_PROMPT_LIBRARY_KEY]
    # In-place migration for any old-format entries still in session
    for k, v in list(lib.items()):
        if isinstance(v, str):
            lib[k] = _migrate_entry(v)
    return lib


def get_prompt_text(label: str) -> str:
    """Return the text for a prompt label, or empty string."""
    entry = get_saved_prompts().get(label)
    if entry is None:
        return ""
    return entry["text"]


def get_prompt_token_ids(label: str) -> list[int] | None:
    """Return stored token IDs for a prompt, or None if text-only."""
    entry = get_saved_prompts().get(label)
    if entry is None:
        return None
    return entry.get("token_ids")


def save_prompt(
    label: str,
    text: str,
    token_ids: list[int] | None = None,
) -> None:
    """Add or overwrite a prompt in the library."""
    get_saved_prompts()[label] = {
        "text": text,
        "token_ids": token_ids,
    }
    _save_prompts_to_disk(get_saved_prompts())


def delete_prompt(label: str) -> None:
    """Remove a single prompt from the library."""
    get_saved_prompts().pop(label, None)
    _save_prompts_to_disk(get_saved_prompts())


def clear_prompts() -> None:
    """Remove all saved prompts."""
    st.session_state[_PROMPT_LIBRARY_KEY] = {}
    _save_prompts_to_disk({})


def normalize_token_str(s: str) -> str:
    """Strip surrounding quotes from a token string copied from repr() output.

    Users often copy tokens from the prompt explorer which displays repr(),
    so '2' or "2" end up as literal quote-wrapped strings. This strips those.

    Does NOT strip whitespace — leading/trailing spaces are meaningful
    for tokens like ' France' vs 'France'.
    """
    if len(s) >= 2:
        if (s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"'):
            s = s[1:-1]
    return s


def resolve_single_token(model, s: str) -> int:
    """Resolve a string to a single token ID.

    Uses tokenizer.encode (without special tokens) and checks the result
    is exactly one token. No silent fallbacks — ' France' and 'France'
    must remain distinct.

    Raises ValueError with a diagnostic message if the string isn't a
    single token, showing what it actually encodes to.
    """
    ids = model.tokenizer.encode(s, add_special_tokens=False)
    if len(ids) == 1:
        return ids[0]

    decoded = [model.tokenizer.decode(tid) for tid in ids]
    raise ValueError(
        f"{s!r} is not a single token — it encodes to {len(ids)} tokens: "
        f"{list(zip(ids, decoded))}. "
        f"Use the Token Lookup on the Model page to find the exact token string."
    )


def token_id_input(
    model,
    label: str = "Answer token",
    key: str = "answer_token",
    help: str | None = None,
) -> int | None:
    """Streamlit widget for entering a token. Returns a resolved token ID or None.

    Accepts either:
      - A numeric token ID with # prefix (e.g. '#220')
      - A token string (e.g. '2', 'France', ' the')
      - A quoted token string from repr() output (e.g. \"' 2'\")

    Whitespace is NEVER stripped — leading/trailing spaces are meaningful
    for token identity (e.g. ' 2' vs '2' are different tokens).
    """
    raw = st.text_input(
        label,
        key=key,
        help=help or "Enter a token string, or #N for a numeric token ID (e.g. #220).",
    )
    if not raw:
        return None

    # Explicit numeric token ID: #123
    if raw.startswith("#"):
        id_part = raw[1:].strip()
        if id_part.isdigit():
            tid = int(id_part)
            if tid < model.cfg.d_vocab:
                decoded = model.tokenizer.decode(tid)
                st.caption(f"Token ID **{tid}** → `{repr(decoded)}`")
                return tid
            else:
                st.warning(f"ID {tid} out of range (vocab size {model.cfg.d_vocab})")
                return None
        st.warning(f"Invalid token ID: `{raw}` — use `#N` format (e.g. `#220`).")
        return None

    # Everything else is a token string — strip quotes from repr() output
    s = normalize_token_str(raw)
    ids = model.tokenizer.encode(s, add_special_tokens=False)

    if len(ids) == 1:
        decoded = model.tokenizer.decode(ids[0])
        st.caption(f"`{repr(s)}` → token ID **{ids[0]}** → `{repr(decoded)}`")
        return ids[0]

    # Not a single token — show what it encodes to so the user can pick
    parts = ", ".join(f"**{tid}** `{repr(model.tokenizer.decode(tid))}`" for tid in ids)
    hint = ""
    if s.isdigit():
        hint = f"  \nDid you mean token **ID** {s}? Use `#{s}` instead."
    st.warning(
        f"`{repr(s)}` encodes to {len(ids)} tokens: {parts}.{hint}  \n"
        f"Pick one of the IDs above with `#` prefix (e.g. `#{ids[0]}`), or use the exact "
        f"token string from the Token Lookup on the Model page."
    )
    return None


# ---------------------------------------------------------------------------
# Shared "active study" prompt selection — synced across analysis pages
# ---------------------------------------------------------------------------
_ACTIVE_A_KEY = "active_prompt_a"
_ACTIVE_B_KEY = "active_prompt_b"
_ACTIVE_ANSWER_KEY = "active_answer_token"


def get_active_prompts() -> dict[str, str]:
    """Return {a: label, b: label, answer: token_str} for active study."""
    return {
        "a": st.session_state.get(_ACTIVE_A_KEY, ""),
        "b": st.session_state.get(_ACTIVE_B_KEY, ""),
        "answer": st.session_state.get(_ACTIVE_ANSWER_KEY, ""),
    }


def set_active_prompt(slot: str, label: str) -> None:
    """Set the active prompt for slot 'a' or 'b'."""
    key = _ACTIVE_A_KEY if slot == "a" else _ACTIVE_B_KEY
    st.session_state[key] = label


def set_active_answer(token_str: str) -> None:
    """Set the active answer token string."""
    st.session_state[_ACTIVE_ANSWER_KEY] = token_str


def prompt_selector(
    key: str,
    label: str = "Prompt",
    allow_empty: bool = True,
    sync_slot: str | None = None,
) -> str:
    """Dropdown to pick a saved prompt. Returns the prompt text (or empty).

    If sync_slot is "a" or "b", the selection is synchronised with the
    active-study state so all analysis pages share the same selection.
    """
    prompts = get_saved_prompts()
    if not prompts:
        st.caption(
            "No saved prompts. Add prompts on the **Prompts** page."
        )
        return ""
    options = ([""] if allow_empty else []) + list(prompts.keys())

    # Pre-populate from active study when this widget key is fresh.
    # Streamlit ignores `index` if the key already lives in session_state,
    # so we seed session_state directly before the widget renders.
    if sync_slot and key not in st.session_state:
        active_label = get_active_prompts().get(sync_slot, "")
        if active_label in options:
            st.session_state[key] = active_label

    choice = st.selectbox(label, options, key=key)

    # Write back to shared active state
    if sync_slot and choice:
        set_active_prompt(sync_slot, choice)

    entry = prompts.get(choice)
    if entry is None:
        return ""
    return entry["text"] if isinstance(entry, dict) else entry


def selected_prompt_label(key: str) -> str:
    """Return the label currently chosen in a prompt_selector widget."""
    return st.session_state.get(key, "")


def prompt_tokenize(
    model,
    selector_key: str,
    prepend_bos: bool,
) -> tuple[torch.Tensor, list[str]]:
    """Tokenize the prompt selected in a prompt_selector widget.

    If the library entry has stored token_ids (built token-by-token),
    those are used directly — preserving the exact token boundaries the
    user chose (e.g. three separate tokens '"', ':', '"' instead of
    the merged '":"' the tokenizer would normally produce).

    Falls back to standard text tokenization otherwise.
    """
    from model import tokenize

    label = selected_prompt_label(selector_key)
    token_ids = get_prompt_token_ids(label) if label else None

    if token_ids is not None:
        ids = list(token_ids)
        if prepend_bos:
            bos = model.tokenizer.bos_token_id
            if bos is not None:
                ids = [bos] + ids
        tokens = torch.tensor([ids], device=model.cfg.device)
        str_tokens = [model.tokenizer.decode(t) for t in ids]
        return tokens, str_tokens

    text = get_prompt_text(label) if label else ""
    return tokenize(model, text, prepend_bos=prepend_bos)


def export_prompts_json() -> str:
    """Return the prompt library as a JSON string for download."""
    return json.dumps(get_saved_prompts(), indent=2, ensure_ascii=False)


def import_prompts_json(raw: str, merge: bool = True) -> int:
    """Import prompts from a JSON string. Returns count of prompts added.

    Accepts both old format ({label: text}) and new format
    ({label: {text, token_ids}}).

    If merge=True, existing prompts are kept and new ones are added/updated.
    If merge=False, the library is replaced entirely.
    """
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object {label: ...}")
    migrated = {k: _migrate_entry(v) for k, v in data.items()}
    if merge:
        prompts = get_saved_prompts()
        prompts.update(migrated)
    else:
        st.session_state[_PROMPT_LIBRARY_KEY] = migrated
    _save_prompts_to_disk(get_saved_prompts())
    return len(data)


def render_sidebar_memory():
    """Render GPU memory metrics and cache-management buttons in the sidebar."""
    with st.sidebar:
        st.subheader("GPU Memory")
        stats = memory_stats()
        if stats:
            c1, c2 = st.columns(2)
            c1.metric("Allocated", f"{stats['allocated']:.2f} GB")
            c2.metric("Free", f"{stats['free']:.2f} GB")
            st.caption(f"Total: {stats['total']:.1f} GB  |  Reserved: {stats['reserved']:.2f} GB")
            b1, b2, b3 = st.columns(3)
            if b1.button("Refresh", key="mem_refresh"):
                st.rerun()
            if b2.button("Clear Cache", key="mem_clear",
                         help="Release PyTorch reserved-but-unused memory."):
                gc.collect()
                torch.cuda.empty_cache()
                st.rerun()
            if b3.button("Hard Reset", key="mem_hard_reset", type="primary",
                         help="Evict the model from GPU + clear all results. "
                              "Model must be reloaded before running experiments."):
                hard_reset()
                st.rerun()
        else:
            st.info("No CUDA device")
