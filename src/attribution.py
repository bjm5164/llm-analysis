"""Direct logit attribution: which components write the answer direction?

Uses the TransformerLens ActivationCache decomposition APIs:
- tokens_to_residual_directions: get the unembedding direction for an answer token
- decompose_resid: coarse attribution (embed, pos_embed, per-block attn/MLP)
- get_full_resid_decomposition: fine-grained (individual heads, MLP)
- logit_attrs: project component contributions onto a token direction
- stack_head_results + apply_ln_to_stack: per-head analysis

Workflow per the answer-tracing skill:
1. Pin down the answer token direction
2. Measure final logit margin
3. accumulated_resid for readability (see logit_lens.py)
4. decompose_resid for coarse component attribution
5. Drill down to per-head attribution
"""

import torch
from transformer_lens import HookedTransformer, ActivationCache


def answer_residual_direction(
    model: HookedTransformer,
    answer: str | int,
) -> torch.Tensor:
    """Get the residual-stream direction for an answer token.

    This is the unembedding vector — the direction in residual space
    that the model's final linear layer uses to produce the answer logit.

    The answer must resolve to exactly one token. Raises ValueError if
    a string encodes to multiple tokens.
    """
    if isinstance(answer, str):
        answer_id = _answer_token_id(model, answer)
        answer_tokens = torch.tensor([[answer_id]], device=model.cfg.device)
    else:
        answer_tokens = torch.tensor([[answer]], device=model.cfg.device)
    return model.tokens_to_residual_directions(answer_tokens).squeeze(0)


def final_logit_margin(
    model: HookedTransformer,
    logits: torch.Tensor,
    answer: str | int,
    distractors: list[str | int] | None = None,
    pos: int = -1,
) -> dict:
    """Measure the final logit margin before any decomposition.

    Returns dict with answer logit, probability, rank, and optionally
    logit differences vs distractors.
    """
    from app_state import resolve_single_token
    answer_id = resolve_single_token(model, answer) if isinstance(answer, str) else answer
    final = logits[0, pos, :]
    probs = torch.softmax(final, dim=-1)

    result = {
        "answer_token": model.tokenizer.decode(answer_id),
        "answer_id": answer_id,
        "logit": final[answer_id].item(),
        "prob": probs[answer_id].item(),
        "rank": (probs > probs[answer_id]).sum().item() + 1,
    }

    if distractors:
        for d in distractors:
            d_id = resolve_single_token(model, d) if isinstance(d, str) else d
            d_str = model.tokenizer.decode(d_id)
            result[f"logit_diff_vs_{repr(d_str)}"] = (
                final[answer_id] - final[d_id]
            ).item()

    return result


def _answer_token_id(model: HookedTransformer, answer: str | int) -> int:
    """Resolve answer to a single token ID."""
    if isinstance(answer, str):
        from app_state import resolve_single_token
        return resolve_single_token(model, answer)
    return answer


def component_attribution(
    model: HookedTransformer,
    cache: ActivationCache,
    answer: str | int,
    pos: int = -1,
) -> tuple[torch.Tensor, list[str]]:
    """Coarse per-component direct logit attribution.

    Uses decompose_resid to break the residual stream into:
    embed, pos_embed, and per-block attention/MLP outputs.
    Then uses logit_attrs with the answer token ID to project each
    component onto the answer's residual direction.

    Returns:
        (attribution_scores, component_labels)
    """
    answer_id = _answer_token_id(model, answer)
    resid_stack, labels = cache.decompose_resid(
        apply_ln=True, pos_slice=pos, return_labels=True
    )
    # logit_attrs takes token IDs (str, int, or tensor), not direction vectors
    attrs = cache.logit_attrs(resid_stack, answer_id)
    if attrs.ndim == 2:
        attrs = attrs[:, 0]  # remove batch dim
    return attrs, labels


def full_decomposition_attribution(
    model: HookedTransformer,
    cache: ActivationCache,
    answer: str | int,
    pos: int = -1,
) -> tuple[torch.Tensor, list[str]]:
    """Fine-grained attribution: individual heads + MLP per layer.

    Uses get_full_resid_decomposition for the most granular breakdown.

    Returns:
        (attribution_scores, component_labels)
    """
    answer_id = _answer_token_id(model, answer)
    resid_stack, labels = cache.get_full_resid_decomposition(
        apply_ln=True, pos_slice=pos, return_labels=True
    )
    attrs = cache.logit_attrs(resid_stack, answer_id)
    if attrs.ndim == 2:
        attrs = attrs[:, 0]
    return attrs, labels


def head_attribution(
    model: HookedTransformer,
    cache: ActivationCache,
    answer: str | int,
    pos: int = -1,
) -> torch.Tensor:
    """Per-head direct logit attribution toward the answer token.

    Uses stack_head_results + apply_ln_to_stack — the canonical TL path
    for per-head analysis. Respects RMSNorm for Qwen models.

    Returns:
        (n_layers, n_heads) tensor of attribution scores.
    """
    answer_id = _answer_token_id(model, answer)
    # Stack all head outputs across layers
    per_head_resid, _ = cache.stack_head_results(
        layer=-1, pos_slice=pos, return_labels=True
    )
    # Apply final LN to the stack (critical for correct attribution)
    per_head_resid = cache.apply_ln_to_stack(
        per_head_resid, layer=-1, pos_slice=pos
    )
    # logit_attrs with token ID
    attrs = cache.logit_attrs(per_head_resid, answer_id)
    if attrs.ndim == 2:
        attrs = attrs[:, 0]
    return attrs.reshape(model.cfg.n_layers, model.cfg.n_heads)


def print_top_components(
    attrs: torch.Tensor,
    labels: list[str],
    answer_label: str,
    k: int = 10,
):
    """Print the top-k components by absolute attribution."""
    sorted_idx = attrs.abs().argsort(descending=True)[:k]
    print(f"\nTop {k} components by attribution toward {answer_label}:")
    for i in sorted_idx:
        print(f"  {labels[i]:30s}  {attrs[i].item():+.4f}")


def print_top_heads(
    head_attrs: torch.Tensor,
    answer_label: str,
    k: int = 10,
):
    """Print the top-k heads by absolute attribution."""
    n_heads = head_attrs.shape[1]
    flat = head_attrs.flatten()
    sorted_idx = flat.abs().argsort(descending=True)[:k]
    print(f"\nTop {k} heads by attribution toward {answer_label}:")
    for idx in sorted_idx:
        li = idx.item() // n_heads
        hi = idx.item() % n_heads
        print(f"  L{li:2d} H{hi:2d}  {flat[idx].item():+.4f}")
