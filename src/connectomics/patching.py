"""Activation patching: causally localize where OOD tokens disrupt computation.

This is the core of the project. The workflow:
1. Run clean prompt -> get clean_cache
2. Run corrupted prompt (OOD suffix injected) -> get corrupted baseline
3. Patch clean activations into the corrupted run at each (layer, position)
4. Measure which patches rescue the correct answer

Uses TransformerLens patching module exclusively — no hand-rolled hooks
for the sweep phase. Custom hooks are for targeted follow-up only.

Key APIs:
    transformer_lens.patching.get_act_patch_resid_pre
    transformer_lens.patching.get_act_patch_resid_mid
    transformer_lens.patching.get_act_patch_attn_out
    transformer_lens.patching.get_act_patch_mlp_out
    transformer_lens.patching.get_act_patch_attn_head_out_by_pos
    model.run_with_hooks (for targeted follow-up interventions)
"""

import torch
from transformer_lens import HookedTransformer, ActivationCache
from transformer_lens import patching as tl_patching


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def make_logit_diff_metric(
    model: HookedTransformer,
    answer: str | int,
    distractor: str | int | None = None,
    pos: int = -1,
):
    """Create a patching metric based on answer logit or logit difference.

    For arithmetic, logit-difference is usually the most interpretable metric.
    If no distractor is given, uses raw answer logit.

    Returns a callable: metric_fn(logits) -> scalar
    """
    answer_id = model.to_single_token(answer) if isinstance(answer, str) else answer

    if distractor is not None:
        distractor_id = (model.to_single_token(distractor)
                         if isinstance(distractor, str) else distractor)

        def metric_fn(logits, pos=pos):
            return logits[0, pos, answer_id] - logits[0, pos, distractor_id]
    else:
        def metric_fn(logits, pos=pos):
            return logits[0, pos, answer_id]

    return metric_fn


def make_prob_metric(
    model: HookedTransformer,
    answer: str | int,
    pos: int = -1,
):
    """Metric based on P(answer) — useful for reporting but less stable for patching."""
    answer_id = model.to_single_token(answer) if isinstance(answer, str) else answer

    def metric_fn(logits, pos=pos):
        return torch.softmax(logits[0, pos, :], dim=-1)[answer_id]

    return metric_fn


# ---------------------------------------------------------------------------
# Sweep patching — use TL's built-in helpers
# ---------------------------------------------------------------------------

def _get_shared_positions(clean_cache: ActivationCache, corrupted_tokens: torch.Tensor) -> int:
    """Number of positions shared between clean and corrupted prompts."""
    import transformer_lens.utils as tl_utils
    clean_shape = clean_cache[tl_utils.get_act_name("resid_pre", 0)].shape
    return min(clean_shape[1], corrupted_tokens.shape[1])


def patch_resid_pre(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    metric_fn,
) -> torch.Tensor:
    """Sweep activation patching over resid_pre at every (layer, position).

    Only patches positions that exist in both clean and corrupted prompts.
    This handles the common case where OOD suffixes make the corrupted
    prompt longer than the clean prompt.

    Returns: (n_layers, shared_positions) tensor of patched metric values.
    """
    n_shared = _get_shared_positions(clean_cache, corrupted_tokens)
    return tl_patching.get_act_patch_resid_pre(
        model, corrupted_tokens, clean_cache, metric_fn
    ) if corrupted_tokens.shape[1] == clean_cache[
        "blocks.0.hook_resid_pre"
    ].shape[1] else _manual_sweep(
        model, corrupted_tokens, clean_cache, metric_fn,
        "resid_pre", n_shared
    )


def patch_attn_out(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    metric_fn,
) -> torch.Tensor:
    """Sweep patching over attention output at every (layer, position).

    Returns: (n_layers, shared_positions) tensor.
    """
    n_shared = _get_shared_positions(clean_cache, corrupted_tokens)
    return tl_patching.get_act_patch_attn_out(
        model, corrupted_tokens, clean_cache, metric_fn
    ) if corrupted_tokens.shape[1] == clean_cache[
        "blocks.0.hook_resid_pre"
    ].shape[1] else _manual_sweep(
        model, corrupted_tokens, clean_cache, metric_fn,
        "attn_out", n_shared
    )


def patch_mlp_out(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    metric_fn,
) -> torch.Tensor:
    """Sweep patching over MLP output at every (layer, position).

    Returns: (n_layers, shared_positions) tensor.
    """
    n_shared = _get_shared_positions(clean_cache, corrupted_tokens)
    return tl_patching.get_act_patch_mlp_out(
        model, corrupted_tokens, clean_cache, metric_fn
    ) if corrupted_tokens.shape[1] == clean_cache[
        "blocks.0.hook_resid_pre"
    ].shape[1] else _manual_sweep(
        model, corrupted_tokens, clean_cache, metric_fn,
        "mlp_out", n_shared
    )


def patch_head_out(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    metric_fn,
) -> torch.Tensor:
    """Sweep patching over per-head outputs at every (layer, head, position).

    Returns: (n_layers, n_heads, shared_positions) tensor.
    """
    n_shared = _get_shared_positions(clean_cache, corrupted_tokens)
    if corrupted_tokens.shape[1] == clean_cache["blocks.0.hook_resid_pre"].shape[1]:
        return tl_patching.get_act_patch_attn_head_out_by_pos(
            model, corrupted_tokens, clean_cache, metric_fn
        )
    return _manual_head_sweep(
        model, corrupted_tokens, clean_cache, metric_fn, n_shared
    )


def _manual_sweep(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    metric_fn,
    hook_type: str,
    n_shared: int,
) -> torch.Tensor:
    """Manual patching sweep for unequal-length prompts.

    Patches one (layer, position) at a time over the shared prefix.
    """
    import transformer_lens.utils as tl_utils

    n_layers = model.cfg.n_layers
    results = torch.zeros(n_layers, n_shared)

    for layer in range(n_layers):
        hook_name = tl_utils.get_act_name(hook_type, layer)
        clean_act = clean_cache[hook_name]

        for pos in range(n_shared):
            def patch_hook(activation, hook, _pos=pos):
                activation[0, _pos, :] = clean_act[0, _pos, :]
                return activation

            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    corrupted_tokens,
                    fwd_hooks=[(hook_name, patch_hook)],
                )
            results[layer, pos] = metric_fn(patched_logits).item()

    return results


def _manual_head_sweep(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    metric_fn,
    n_shared: int,
) -> torch.Tensor:
    """Manual per-head patching sweep for unequal-length prompts."""
    import transformer_lens.utils as tl_utils

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    results = torch.zeros(n_layers, n_heads, n_shared)

    for layer in range(n_layers):
        hook_name = tl_utils.get_act_name("z", layer)
        clean_act = clean_cache[hook_name]

        for head in range(n_heads):
            for pos in range(n_shared):
                def patch_hook(activation, hook, _head=head, _pos=pos):
                    activation[0, _pos, _head, :] = clean_act[0, _pos, _head, :]
                    return activation

                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        corrupted_tokens,
                        fwd_hooks=[(hook_name, patch_hook)],
                    )
                results[layer, head, pos] = metric_fn(patched_logits).item()

    return results


# ---------------------------------------------------------------------------
# Targeted interventions — for follow-up after sweep localization
# ---------------------------------------------------------------------------

def patch_position_at_layer(
    model: HookedTransformer,
    corrupted_tokens: torch.Tensor,
    clean_cache: ActivationCache,
    layer: int,
    pos: int,
    hook_point: str = "resid_pre",
) -> torch.Tensor:
    """Patch a single (layer, position) and return full output logits.

    Use this for targeted follow-up after the sweep identifies key sites.
    """
    import transformer_lens.utils as tl_utils
    hook_name = tl_utils.get_act_name(hook_point, layer)
    clean_act = clean_cache[hook_name]

    def patch_hook(activation, hook):
        activation[0, pos, :] = clean_act[0, pos, :]
        return activation

    with torch.no_grad():
        return model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(hook_name, patch_hook)],
        )


def zero_suffix_contribution(
    model: HookedTransformer,
    tokens: torch.Tensor,
    suffix_start: int,
    layer: int,
    hook_point: str = "resid_pre",
) -> torch.Tensor:
    """Zero out residual stream at suffix positions for a given layer.

    Tests whether the suffix positions are actively injecting harmful
    information vs the damage happening through attention re-routing.
    """
    import transformer_lens.utils as tl_utils
    hook_name = tl_utils.get_act_name(hook_point, layer)

    def zero_hook(activation, hook):
        activation[0, suffix_start:, :] = 0.0
        return activation

    with torch.no_grad():
        return model.run_with_hooks(
            tokens,
            fwd_hooks=[(hook_name, zero_hook)],
        )


# ---------------------------------------------------------------------------
# Baseline comparison helper
# ---------------------------------------------------------------------------

def compare_clean_corrupted(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    corrupted_tokens: torch.Tensor,
    metric_fn,
) -> dict:
    """Run both prompts and report baseline metrics before patching.

    Always do this first — before patching, understand how much
    the corruption actually hurts.
    """
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits = model(corrupted_tokens)

    clean_score = metric_fn(clean_logits).item()
    corrupt_score = metric_fn(corrupted_logits).item()

    return {
        "clean_score": clean_score,
        "corrupted_score": corrupt_score,
        "delta": clean_score - corrupt_score,
        "clean_logits": clean_logits,
        "clean_cache": clean_cache,
        "corrupted_logits": corrupted_logits,
    }
