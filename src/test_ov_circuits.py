"""Tests for ov_circuits activation-based composition."""

import torch
import numpy as np
from transformer_lens import HookedTransformer

from ov_circuits import (
    head_activity,
    ov_eigenvalues_single,
    copying_score,
    _head_result,
    composition_from_act,
    composition_to_act,
    trace_circuit_act,
)


def test_shapes_and_indexing():
    """Verify all tensor shapes and indexing against a real model cache."""
    model = HookedTransformer.from_pretrained("gpt2", device="cpu")
    tokens = model.to_tokens("The cat sat on the mat")
    _, cache = model.run_with_cache(tokens)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    seq_len = tokens.shape[1]

    print(f"Model: n_layers={n_layers}, n_heads={n_heads}, "
          f"d_model={d_model}, d_head={d_head}, seq_len={seq_len}")

    # --- Check cache tensor shapes ---
    for layer in [0, n_layers // 2, n_layers - 1]:
        z = cache["z", layer]
        v = cache["v", layer]
        pattern = cache["pattern", layer]

        print(f"\nLayer {layer}:")
        print(f"  z.shape      = {z.shape}  (expect: 1, {seq_len}, {n_heads}, {d_head})")
        print(f"  v.shape      = {v.shape}  (expect: 1, {seq_len}, {n_heads}, {d_head})")
        print(f"  pattern.shape= {pattern.shape}  (expect: 1, {n_heads}, {seq_len}, {seq_len})")

        assert z.shape == (1, seq_len, n_heads, d_head), f"z shape mismatch: {z.shape}"
        assert v.shape == (1, seq_len, n_heads, d_head), f"v shape mismatch: {v.shape}"
        assert pattern.shape == (1, n_heads, seq_len, seq_len), f"pattern shape mismatch: {pattern.shape}"

    # --- Check W_V and W_O shapes ---
    print(f"\nW_V.shape = {model.W_V.shape}  (expect: {n_layers}, {n_heads}, {d_model}, {d_head})")
    print(f"W_O.shape = {model.W_O.shape}  (expect: {n_layers}, {n_heads}, {d_head}, {d_model})")
    assert model.W_V.shape == (n_layers, n_heads, d_model, d_head)
    assert model.W_O.shape == (n_layers, n_heads, d_head, d_model)

    # --- Check head_activity ---
    act = head_activity(model, cache, pos=-1)
    print(f"\nhead_activity.shape = {act.shape}  (expect: {n_layers}, {n_heads})")
    assert act.shape == (n_layers, n_heads)
    assert np.all(act >= 0), "activity norms should be non-negative"
    assert act.sum() > 0, "at least some heads should fire"

    # Verify against manual computation for one head
    layer_check, head_check = 2, 3
    z_manual = cache["z", layer_check][0, -1, head_check, :]  # (d_head,)
    w_o = model.W_O[layer_check, head_check]  # (d_head, d_model)
    manual_norm = (z_manual.float() @ w_o.float()).norm().item()
    assert abs(act[layer_check, head_check] - manual_norm) < 1e-3, (
        f"activity mismatch: {act[layer_check, head_check]} vs {manual_norm}"
    )
    print(f"head_activity manual check passed: L{layer_check}H{head_check} = {manual_norm:.4f}")

    # --- Check _head_result ---
    layer, head = 0, 0
    result = _head_result(model, cache, layer, head)
    print(f"\n_head_result(layer={layer}, head={head}).shape = {result.shape}  (expect: {seq_len}, {d_model})")
    assert result.shape == (seq_len, d_model)

    # --- Check eigenvalues ---
    eigs = ov_eigenvalues_single(model, 0, 0)
    print(f"\nov_eigenvalues_single.shape = {eigs.shape}  (expect: {d_model},)")
    assert eigs.shape == (d_model,)
    assert np.all(np.diff(np.abs(eigs)) <= 1e-6), "eigenvalues not sorted by descending magnitude"

    cscore = copying_score(eigs)
    print(f"copying_score = {cscore:.4f}  (expect: 0 <= score <= 1)")
    assert 0.0 <= cscore <= 1.0

    # --- Check composition_from_act indexing step by step ---
    seed_layer, seed_head = 2, 3
    pos = -1
    print(f"\n--- composition_from_act(seed=L{seed_layer}H{seed_head}, pos={pos}) ---")

    z_seed = _head_result(model, cache, seed_layer, seed_head).float()
    print(f"z_seed.shape = {z_seed.shape}  (expect: {seq_len}, {d_model})")
    assert z_seed.shape == (seq_len, d_model)

    lb = seed_layer + 1
    # Pattern indexing
    pattern_lb = cache["pattern", lb]
    print(f"pattern[{lb}].shape = {pattern_lb.shape}")
    attn_B = pattern_lb[0, :, pos, :]  # (n_heads, seq_k)
    print(f"attn_B.shape = {attn_B.shape}  (expect: {n_heads}, {seq_len})")
    assert attn_B.shape == (n_heads, seq_len)

    # V indexing
    v_lb = cache["v", lb]
    print(f"v[{lb}].shape = {v_lb.shape}")
    v_B = v_lb[0].float()  # (seq, n_heads, d_head)
    print(f"v_B.shape = {v_B.shape}  (expect: {seq_len}, {n_heads}, {d_head})")
    assert v_B.shape == (seq_len, n_heads, d_head)

    hb = 0
    W_V_B = model.W_V[lb, hb].float()
    print(f"W_V_B.shape = {W_V_B.shape}  (expect: {d_model}, {d_head})")
    assert W_V_B.shape == (d_model, d_head)

    seed_proj = z_seed @ W_V_B
    print(f"seed_proj.shape = {seed_proj.shape}  (expect: {seq_len}, {d_head})")
    assert seed_proj.shape == (seq_len, d_head)

    w = attn_B[hb].unsqueeze(-1)
    print(f"w.shape = {w.shape}  (expect: {seq_len}, 1)")
    assert w.shape == (seq_len, 1)

    seed_contrib = (w * seed_proj).sum(0)
    print(f"seed_contrib.shape = {seed_contrib.shape}  (expect: {d_head},)")
    assert seed_contrib.shape == (d_head,)

    actual_mixed = (w * v_B[:, hb, :]).sum(0)
    print(f"actual_mixed.shape = {actual_mixed.shape}  (expect: {d_head},)")
    assert actual_mixed.shape == (d_head,)

    # --- Run full composition functions ---
    print("\n--- Full function calls ---")
    readers = composition_from_act(model, cache, seed_layer, seed_head, pos=pos)
    print(f"readers.shape = {readers.shape}  (expect: {n_layers}, {n_heads})")
    assert readers.shape == (n_layers, n_heads)
    assert np.all(readers[:seed_layer + 1] == 0), "layers <= seed should be 0"
    assert np.all(readers >= 0), "scores should be non-negative norms"
    print(f"readers non-zero layers: {np.nonzero(readers.sum(axis=1))[0].tolist()}")
    print(f"readers max = {readers.max():.4f}")

    writers = composition_to_act(model, cache, seed_layer, seed_head, pos=pos)
    print(f"writers.shape = {writers.shape}  (expect: {n_layers}, {n_heads})")
    assert writers.shape == (n_layers, n_heads)
    assert np.all(writers[seed_layer:] == 0), "layers >= seed should be 0"
    assert np.all(writers >= 0), "scores should be non-negative norms"
    print(f"writers non-zero layers: {np.nonzero(writers.sum(axis=1))[0].tolist()}")
    print(f"writers max = {writers.max():.4f}")

    # --- Run trace_circuit_act ---
    edges = trace_circuit_act(model, cache, seed_layer, seed_head, pos=pos, depth=1, top_k=3)
    print(f"\ntrace_circuit_act returned {len(edges)} edges")
    for la, ha, lb, hb, s in edges[:5]:
        print(f"  L{la}H{ha} -> L{lb}H{hb}: {s:.4f}")
        assert s > 0, f"score should be positive: {s}"

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_shapes_and_indexing()
