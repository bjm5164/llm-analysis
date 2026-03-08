"""Shared state and utilities for the Streamlit app.

Provides:
  - @st.cache_resource model loading (pinned in GPU memory, keyed by config)
  - Config management via session state with live YAML editing
  - Sidebar GPU memory widget with clear-cache button
"""

import gc
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
        return None, (
            "Invalid config — check required fields (model.name, prompts.clean, tokens.answer)."
        )
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
    for key in ["baseline_results", "ood_results", "patching_results", "sweep_results"]:
        if key in st.session_state:
            del st.session_state[key]
    st.cache_resource.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
