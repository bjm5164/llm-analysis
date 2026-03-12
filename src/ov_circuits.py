"""OV circuit eigenvalue analysis and activation-based composition scores.

Analyzes the OV matrix (W_V @ W_O) of each attention head to characterize
head behavior (copying, negation, suppression) and computes activation-based
composition scores to identify which heads form circuits on a specific input.

Neuroscience-style workflow:
  1. Stimulus: run the model on input tokens (produces a cache).
  2. Activity map: ``head_activity()`` — which heads "fired" at each position?
  3. Signal trace: from a seed head, trace its output through downstream heads
     via ``composition_from_act`` / ``composition_to_act``.
  4. Circuit graph: ``trace_circuit_act`` — BFS from the seed using composition.

Reference: "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021)
"""

from __future__ import annotations

import numpy as np
import torch
from transformer_lens import ActivationCache, HookedTransformer


# ---------------------------------------------------------------------------
# Activity map — which heads fired at a given position?
# ---------------------------------------------------------------------------

def head_activity(
    model: HookedTransformer,
    cache: ActivationCache,
    pos: int = -1,
) -> np.ndarray:
    """Per-head output norm at a single position: how strongly each head "fired".

    Computes ``‖z[l, h, pos] @ W_O[l, h]‖`` for every (layer, head).

    Args:
        pos: Token position to measure. Negative indices work (e.g. -1 = last).

    Returns:
        Float array (n_layers, n_heads) of L2 norms.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    activity = np.zeros((n_layers, n_heads), dtype=np.float32)

    with torch.no_grad():
        for layer in range(n_layers):
            z = cache["z", layer][0, pos, :, :]  # (n_heads, d_head)
            W_O = model.W_O[layer]                # (n_heads, d_head, d_model)
            # Batched: result[h] = z[h] @ W_O[h] → (n_heads, d_model)
            result = torch.einsum("hd,hdm->hm", z.float(), W_O.float())
            activity[layer] = result.norm(dim=-1).cpu().numpy()

    return activity


# ---------------------------------------------------------------------------
# Weight-based: eigenvalue analysis (input-independent)
# ---------------------------------------------------------------------------

def ov_eigenvalues_single(
    model: HookedTransformer, layer: int, head: int,
) -> np.ndarray:
    """Eigenvalues of W_OV for a single head, sorted by descending magnitude.

    Returns:
        Complex array of shape (d_model,).
    """
    with torch.no_grad():
        W_V = model.W_V[layer, head]  # (d_model, d_head)
        W_O = model.W_O[layer, head]  # (d_head, d_model)
        W_OV = (W_V @ W_O).float()
        eigs = torch.linalg.eigvals(W_OV).cpu().numpy()
    order = np.argsort(-np.abs(eigs))
    return eigs[order]


def copying_score(eigenvalues: np.ndarray) -> float:
    """Copying score for one head from its eigenvalue spectrum.

    Fraction of eigenvalue magnitude from positive-real-dominant eigenvalues.
    Score near 1.0 → identity/copy on a subspace.
    """
    magnitudes = np.abs(eigenvalues)
    total = magnitudes.sum()
    if total == 0:
        return 0.0
    real_dominant = (eigenvalues.real > 0) & (
        np.abs(eigenvalues.real) > np.abs(eigenvalues.imag)
    )
    return float(np.where(real_dominant, magnitudes, 0).sum() / total)


# ---------------------------------------------------------------------------
# Activation-based: composition on a specific input
# ---------------------------------------------------------------------------

def _head_result(
    model: HookedTransformer, cache: ActivationCache, layer: int, head: int,
) -> torch.Tensor:
    """Per-head output to residual stream: (seq, d_model).

    Computes z @ W_O from the cached hook_z (per-head pre-projection output).
    """
    z = cache["z", layer][0, :, head, :]       # (seq, d_head)
    W_O = model.W_O[layer, head]               # (d_head, d_model)
    return z @ W_O                             # (seq, d_model)


def composition_from_act(
    model: HookedTransformer,
    cache: ActivationCache,
    seed_layer: int,
    seed_head: int,
    pos: int = -1,
) -> np.ndarray:
    """Activation-based composition: which later heads read the seed's output?

    For each head B in layers after seed_layer, computes how much of the
    seed head's output reaches B's value space at query position `pos`,
    weighted by B's attention pattern.

        score = ‖Σ_p attn_B[pos,p] · (result_seed[p] @ W_V_B)‖

    Scores are raw norms (not fractions) — higher means more influence.
    Residual stream components can cancel, so a ratio-based fraction
    would be misleading.

    Returns:
        Float array (n_layers, n_heads). Layers ≤ seed_layer are 0.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = np.zeros((n_layers, n_heads), dtype=np.float32)

    z_seed = _head_result(model, cache, seed_layer, seed_head).float()  # (seq, d_model)

    with torch.no_grad():
        for lb in range(seed_layer + 1, n_layers):
            attn_B = cache["pattern", lb][0, :, pos, :]  # (n_heads, seq_k)

            for hb in range(n_heads):
                W_V_B = model.W_V[lb, hb].float()         # (d_model, d_head)
                seed_proj = z_seed @ W_V_B                 # (seq, d_head)

                w = attn_B[hb].unsqueeze(-1)               # (seq, 1)
                seed_contrib = (w * seed_proj).sum(0)       # (d_head,)
                scores[lb, hb] = seed_contrib.norm().item()

    return scores


def composition_to_act(
    model: HookedTransformer,
    cache: ActivationCache,
    seed_layer: int,
    seed_head: int,
    pos: int = -1,
) -> np.ndarray:
    """Activation-based composition: which earlier heads write to the seed?

    For each head A in layers before seed_layer, computes how much of
    head A's output reaches the seed's value space at query position `pos`,
    weighted by the seed's attention pattern.

        score = ‖Σ_p attn_seed[pos,p] · (result_A[p] @ W_V_seed)‖

    Scores are raw norms — higher means more influence.

    Returns:
        Float array (n_layers, n_heads). Layers ≥ seed_layer are 0.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = np.zeros((n_layers, n_heads), dtype=np.float32)

    W_V_seed = model.W_V[seed_layer, seed_head].float()  # (d_model, d_head)
    attn_seed = cache["pattern", seed_layer][0, seed_head, pos, :]  # (seq_k,)

    with torch.no_grad():
        w = attn_seed.unsqueeze(-1)  # (seq_k, 1)
        for la in range(seed_layer):
            for ha in range(n_heads):
                z_A = _head_result(model, cache, la, ha).float()  # (seq, d_model)
                a_proj = z_A @ W_V_seed                    # (seq, d_head)
                a_contrib = (w * a_proj).sum(0)             # (d_head,)
                scores[la, ha] = a_contrib.norm().item()

    return scores


def trace_circuit_act(
    model: HookedTransformer,
    cache: ActivationCache,
    seed_layer: int,
    seed_head: int,
    pos: int = -1,
    depth: int = 2,
    top_k: int = 5,
) -> list[tuple[int, int, int, int, float]]:
    """Trace a circuit graph from a seed head using activation-based composition.

    BFS outward from seed: forward (readers) and backward (writers),
    up to `depth` hops, keeping the top-k heads at each hop.

    Returns:
        List of (la, ha, lb, hb, score) edges, sorted by descending score.
    """
    n_heads = model.cfg.n_heads
    edges: dict[tuple[int, int, int, int], float] = {}
    visited_fwd: set[tuple[int, int]] = set()
    visited_bwd: set[tuple[int, int]] = set()

    # Forward: seed → readers → readers → ...
    frontier = [(seed_layer, seed_head)]
    for _ in range(depth):
        next_frontier = []
        for l, h in frontier:
            if (l, h) in visited_fwd:
                continue
            visited_fwd.add((l, h))
            scores = composition_from_act(model, cache, l, h, pos=pos)
            flat = scores.flatten()
            top_idx = np.argsort(-flat)[:top_k]
            for idx in top_idx:
                lb, hb = int(idx) // n_heads, int(idx) % n_heads
                s = float(flat[idx])
                if s > 0:
                    edges[(l, h, lb, hb)] = max(edges.get((l, h, lb, hb), 0), s)
                    next_frontier.append((lb, hb))
        frontier = next_frontier

    # Backward: writers → seed
    frontier = [(seed_layer, seed_head)]
    for _ in range(depth):
        next_frontier = []
        for l, h in frontier:
            if (l, h) in visited_bwd:
                continue
            visited_bwd.add((l, h))
            scores = composition_to_act(model, cache, l, h, pos=pos)
            flat = scores.flatten()
            top_idx = np.argsort(-flat)[:top_k]
            for idx in top_idx:
                la, ha = int(idx) // n_heads, int(idx) % n_heads
                s = float(flat[idx])
                if s > 0:
                    edges[(la, ha, l, h)] = max(edges.get((la, ha, l, h), 0), s)
                    next_frontier.append((la, ha))
        frontier = next_frontier

    result = [(la, ha, lb, hb, s) for (la, ha, lb, hb), s in edges.items()]
    result.sort(key=lambda x: -x[4])
    return result
