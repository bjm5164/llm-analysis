"""Model loading, tokenization, and cache helpers via TransformerLens.

Follows the TransformerLens Qwen setup skill:
- explicit BOS handling everywhere
- tokenization verification before any interpretability claims
- selective caching for memory-efficient sweeps
- uses TL APIs (to_tokens, to_str_tokens, to_single_token, run_with_cache)
"""

import gc
import random as _random

import torch
from transformer_lens import HookedTransformer
import transformer_lens.utils as tl_utils

from config import ModelConfig, CorruptionConfig


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_model(model_cfg: ModelConfig) -> HookedTransformer:
    """Load a model into TransformerLens HookedTransformer from config."""
    device = model_cfg.device
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = DTYPE_MAP.get(model_cfg.dtype, torch.float32)

    model = HookedTransformer.from_pretrained_no_processing(
        model_cfg.name,
        device=device,
        dtype=dtype,
    )
    cfg = model.cfg
    print(f"Loaded {model_cfg.name} on {device} ({model_cfg.dtype})")
    print(f"  layers={cfg.n_layers}  d_model={cfg.d_model}  "
          f"heads={cfg.n_heads}  d_head={cfg.d_head}  vocab={cfg.d_vocab}")
    if hasattr(cfg, "n_key_value_heads") and cfg.n_key_value_heads != cfg.n_heads:
        print(f"  GQA: n_kv_heads={cfg.n_key_value_heads}")
    return model


# ---------------------------------------------------------------------------
# Tokenization — always explicit about BOS
# ---------------------------------------------------------------------------

def tokenize(
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """Tokenize a prompt. Returns (token_ids [1, S], str_tokens [S]).

    BOS handling is explicit to avoid silent mismatches across prompt variants.
    Decodes each token individually for reliability across tokenizer backends.
    """
    tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    str_tokens = [model.tokenizer.decode(t.item()) for t in tokens[0]]
    return tokens, str_tokens


def verify_tokenization(
    model: HookedTransformer,
    prompt: str,
    prepend_bos: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """Tokenize and print each token for inspection."""
    tokens, str_tokens = tokenize(model, prompt, prepend_bos=prepend_bos)
    print(f"Prompt: {repr(prompt)}  (prepend_bos={prepend_bos})")
    print(f"  {'idx':>4}  {'id':>6}  token")
    for i, (tid, stok) in enumerate(zip(tokens[0], str_tokens)):
        print(f"  [{i:2d}]  {tid.item():6d}  {repr(stok)}")
    return tokens, str_tokens


def verify_answer_token(model: HookedTransformer, answer: str) -> int:
    """Verify that the answer is a single token and return its ID."""
    from app_state import resolve_single_token
    answer_id = resolve_single_token(model, answer)
    answer_str = model.to_single_str_token(answer_id)
    print(f"Answer token: {repr(answer)} -> id={answer_id}, decoded={repr(answer_str)}")
    return answer_id


# ---------------------------------------------------------------------------
# Cache helpers — selective caching for memory efficiency
# ---------------------------------------------------------------------------

CACHE_STRATEGIES = {
    "full": None,  # cache everything
    "resid": lambda model: [
        tl_utils.get_act_name(hook, layer)
        for layer in range(model.cfg.n_layers)
        for hook in ("resid_pre", "resid_mid", "resid_post")
    ],
    "attn_pattern": lambda model: [
        tl_utils.get_act_name("pattern", layer)
        for layer in range(model.cfg.n_layers)
    ],
    "minimal": lambda model: [
        tl_utils.get_act_name(hook, layer)
        for layer in range(model.cfg.n_layers)
        for hook in ("resid_pre", "pattern")
    ],
}


def run_with_cache(
    model: HookedTransformer,
    tokens: torch.Tensor,
    strategy: str = "full",
):
    """Run model with cache using a named strategy.

    Strategies:
        full         - cache everything (short prompts / exploration)
        resid        - residual stream only (logit lens / DLA)
        attn_pattern - attention patterns only
        minimal      - resid_pre + patterns
    """
    builder = CACHE_STRATEGIES.get(strategy)
    kwargs = {}
    if builder is not None:
        filter_set = set(builder(model))
        kwargs["names_filter"] = lambda name: name in filter_set
    with torch.no_grad():
        return model.run_with_cache(tokens, **kwargs)


def sanity_check(model: HookedTransformer, prompt: str = "what is 1 + 1", strategy: str = "full"):
    """Minimal cache sanity check per the setup skill."""
    tokens, str_tokens = verify_tokenization(model, prompt)
    logits, cache = run_with_cache(model, tokens, strategy=strategy)

    print(f"\nCache sanity check:")
    print(f"  logits:       {logits.shape}")
    print(f"  resid_pre[0]: {cache[tl_utils.get_act_name('resid_pre', 0)].shape}")
    print(f"  pattern[0]:   {cache[tl_utils.get_act_name('pattern', 0)].shape}")

    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_vals, top_ids = probs.topk(5)
    print(f"\nTop-5 next-token predictions:")
    for tid, p in zip(top_ids, top_vals):
        print(f"  {repr(model.tokenizer.decode(tid.item())):15s}  {p.item():.4f}")

    return logits, cache


# ---------------------------------------------------------------------------
# Memory utilities — for interactive / notebook use
# ---------------------------------------------------------------------------

def gpu_memory() -> dict:
    """Return current GPU memory stats in GB. Prints if CUDA is available."""
    if not torch.cuda.is_available():
        print("No CUDA device.")
        return {}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved  = torch.cuda.memory_reserved()  / 1e9
    total     = torch.cuda.get_device_properties(0).total_memory / 1e9
    free      = total - reserved
    stats = {"allocated": allocated, "reserved": reserved, "free": free, "total": total}
    print(f"GPU memory — allocated: {allocated:.2f} GB  reserved: {reserved:.2f} GB  "
          f"free: {free:.2f} GB  total: {total:.2f} GB")
    return stats


def corrupt_tokens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    cfg: CorruptionConfig,
    prepend_bos: bool = True,
) -> torch.Tensor:
    """Inject whitespace/newline tokens at random positions in a token sequence.

    Only strings that map to a single token ID are used; multi-token strings
    are silently skipped. BOS at position 0 is never displaced.

    Args:
        tokens:      (1, seq_len) token tensor — typically the clean prompt.
        cfg:         CorruptionConfig with inject list, count, and seed.
        prepend_bos: if True, position 0 (BOS) is excluded from injection sites.

    Returns:
        (1, seq_len + cfg.count) token tensor on the same device.
    """
    rng = _random.Random(cfg.seed)

    inject_ids = [
        ids[0]
        for s in cfg.inject
        for ids in [model.tokenizer.encode(s, add_special_tokens=False)]
        if len(ids) == 1
    ]
    if not inject_ids:
        raise ValueError(
            f"None of {cfg.inject!r} map to single tokens in this tokenizer. "
            "Try '\\n', ' ', or '\\t'."
        )

    seq = tokens[0].tolist()
    start = 1 if prepend_bos else 0
    for _ in range(cfg.count):
        pos = rng.randint(start, len(seq))
        seq.insert(pos, rng.choice(inject_ids))

    return torch.tensor([seq], device=tokens.device, dtype=tokens.dtype)


def answer_token_logit_lens(
    model: HookedTransformer,
    tokens: torch.Tensor,
    answer_id: int,
) -> list[dict]:
    """Run logit-lens over residual stream to find where an answer token emerges.

    For every layer, projects resid_post at the final token position through the
    model's final LayerNorm + unembed, then records the answer token's logit rank
    and softmax probability.

    Args:
        tokens:    (1, seq_len) token tensor.
        answer_id: target token ID to track.

    Returns:
        List of dicts with keys: layer, rank (0-based), prob, logit.
    """
    n_layers = model.cfg.n_layers
    names = [tl_utils.get_act_name("resid_post", layer) for layer in range(n_layers)]
    filter_set = set(names)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens, names_filter=lambda name: name in filter_set
        )

    results = []
    for layer in range(n_layers):
        resid = cache[tl_utils.get_act_name("resid_post", layer)]  # (1, seq, d)
        final_resid = resid[0, -1, :]  # (d,)

        # Apply final layer norm + unembed (same as model's last step)
        normed = model.ln_final(final_resid.unsqueeze(0).unsqueeze(0))  # (1,1,d)
        logits_vec = model.unembed(normed)[0, 0, :]  # (vocab,)

        probs = torch.softmax(logits_vec, dim=-1)
        # rank: how many tokens have higher prob (0-based; 0 = top prediction)
        rank = int((probs > probs[answer_id]).sum().item())

        results.append({
            "layer": layer,
            "rank": rank,
            "prob": float(probs[answer_id].item()),
            "logit": float(logits_vec[answer_id].item()),
        })
    return results


def generate_next_token_greedy(
    model: HookedTransformer,
    tokens: torch.Tensor,
) -> tuple[int, str, float]:
    """Generate one token via greedy decoding (argmax of final-position logits).

    Args:
        tokens: (1, seq_len) token tensor.

    Returns:
        (token_id, decoded_string, probability)
    """
    with torch.no_grad():
        logits = model(tokens)  # (1, seq, vocab)
    logits_final = logits[0, -1, :]
    probs = torch.softmax(logits_final, dim=-1)
    next_id = int(logits_final.argmax().item())
    next_str = model.tokenizer.decode([next_id])
    return next_id, next_str, float(probs[next_id].item())


def free_memory(*args):
    """Delete passed objects, run gc, and empty the CUDA cache.

    Usage in a notebook::

        logits, cache = run_with_cache(model, tokens)
        # ... analysis ...
        free_memory(logits, cache)   # or just free_memory() to only clear cache
    """
    for obj in args:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gpu_memory()
