"""
Constrained decoding via llguidance (CFG-based, same approach as OpenAI Structured Outputs).

Provides:
  - LLGuidanceConstraint: wraps llguidance's LLMatcher for use in a manual
    token-by-token generation loop (compatible with TransformerLens).
  - constrained_generate(): run constrained generation and return full token-level
    metadata (which tokens were forced, how many alternatives existed, etc.)
  - unconstrained_generate(): baseline free-form generation for comparison.

Designed so every generated token can be traced through the model with
run_with_cache for mechanistic interpretability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import llguidance as llg
from huggingface_hub import hf_hub_download


@dataclass
class TokenConstraintInfo:
    """Per-token metadata from constrained generation."""
    position: int
    token_id: int
    token_str: str
    allowed_count: int          # how many tokens the grammar allowed at this step
    was_forced: bool            # True if only 1 token was allowed (fast-forward)
    is_schema_token: bool       # True if this token is part of the JSON structure (not user content)
    logit_mask: Optional[torch.Tensor] = None  # the actual mask applied (vocab-sized bool tensor)


@dataclass
class ConstrainedGenerationResult:
    """Full result of a constrained generation run."""
    prompt_tokens: list[int]
    generated_tokens: list[int]
    all_tokens: list[int]       # prompt + generated
    token_info: list[TokenConstraintInfo]
    decoded_output: str
    parsed_output: Optional[dict] = None  # parsed JSON if applicable


def _find_tokenizer_json(model_name: str) -> str:
    """Locate or download the tokenizer.json for a HF model."""
    path = hf_hub_download(model_name, "tokenizer.json")
    return path


def _build_llg_tokenizer(model_name: str) -> llg.LLTokenizer:
    """Create an llguidance tokenizer from a HF model name."""
    tok_path = _find_tokenizer_json(model_name)
    return llg.LLTokenizer(tok_path)


def make_entity_extraction_grammar(
    additional_properties: bool = False,
) -> str:
    """Create a JSON-schema grammar for entity extraction: {"entity": "<string>"}"""
    schema = {
        "type": "object",
        "properties": {
            "entity": {"type": "string"}
        },
        "required": ["entity"],
        "additionalProperties": additional_properties,
    }
    return llg.grammar_from("json_schema", json.dumps(schema))


def make_json_schema_grammar(schema: dict) -> str:
    """Create an llguidance grammar from an arbitrary JSON schema."""
    return llg.grammar_from("json_schema", json.dumps(schema))


class LLGuidanceConstraint:
    """
    Wraps llguidance LLMatcher to produce logit masks for constrained decoding.

    Usage in a manual generation loop:
        constraint = LLGuidanceConstraint(model_name, grammar)
        for step in range(max_tokens):
            logits = model(input_ids)[:, -1, :]
            mask, info = constraint.get_mask_and_info(logits.device)
            logits[~mask] = float('-inf')
            token = sample(logits)
            done = constraint.consume(token)
            if done:
                break
    """

    def __init__(self, model_name: str, grammar: str, model_vocab_size: int | None = None):
        self.llg_tok = _build_llg_tokenizer(model_name)
        self.grammar = grammar
        self.matcher = llg.LLMatcher(self.llg_tok, grammar)
        self._llg_vocab_size = self.llg_tok.vocab_size
        # Model vocab may be larger (extra special tokens); pad mask to match
        self.vocab_size = max(model_vocab_size or 0, self._llg_vocab_size)
        self._step = 0

    def reset(self):
        """Reset for a new generation."""
        self.matcher = llg.LLMatcher(self.llg_tok, self.grammar)
        self._step = 0

    def get_mask_and_info(self, device: torch.device) -> tuple[torch.Tensor, TokenConstraintInfo]:
        """
        Compute the token mask for the current step.

        Returns:
            mask: bool tensor of shape (vocab_size,) — True = allowed
            info: constraint metadata for this step (token_id/token_str filled after consume)
        """
        # Check for fast-forward (deterministic) tokens first
        ff_tokens = self.matcher.compute_ff_tokens()

        bitmask = self.matcher.compute_bitmask()
        mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)

        # Convert bitmask bytes to bool tensor efficiently
        bitmask_bytes = bytes(bitmask)
        bitmask_tensor = torch.frombuffer(
            bytearray(bitmask_bytes), dtype=torch.uint8
        )
        # Unpack bits: each byte -> 8 bools (LSB first, matching llguidance convention)
        bits = bitmask_tensor.unsqueeze(1).bitwise_and(
            torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8)
        ).bool().flatten()
        mask[:min(len(bits), self.vocab_size)] = bits[:self.vocab_size]
        mask = mask.to(device)

        allowed_count = int(mask.sum().item())
        was_forced = len(ff_tokens) > 0 or allowed_count == 1
        # Schema tokens are heavily constrained (< 1% of vocab allowed)
        # This catches {, ", :, }, key names — not just single-option forced tokens
        is_schema = allowed_count < (self.vocab_size * 0.01)

        info = TokenConstraintInfo(
            position=self._step,
            token_id=-1,  # filled after consume
            token_str="",
            allowed_count=allowed_count,
            was_forced=was_forced,
            is_schema_token=is_schema,
            logit_mask=mask.clone(),
        )

        return mask, info

    def consume(self, token_id: int) -> bool:
        """
        Feed a token to the grammar matcher.

        Returns True if generation should stop.
        """
        ff_tokens = self.matcher.compute_ff_tokens()
        if ff_tokens:
            self.matcher.consume_tokens(ff_tokens)
        else:
            self.matcher.consume_token(token_id)
        self._step += 1
        return self.matcher.is_stopped()

    @property
    def is_accepting(self) -> bool:
        return self.matcher.is_accepting()

    @property
    def is_stopped(self) -> bool:
        return self.matcher.is_stopped()


def constrained_generate(
    model,  # TransformerLens HookedTransformer
    prompt_tokens: list[int],
    grammar: str,
    model_name: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
) -> ConstrainedGenerationResult:
    """
    Run constrained generation token-by-token.

    This does NOT use model.generate() — it runs a manual loop so each
    forward pass can be independently traced with run_with_cache.

    Args:
        model: TransformerLens HookedTransformer
        prompt_tokens: tokenized prompt
        grammar: llguidance grammar string (from make_*_grammar)
        model_name: HF model name (for llguidance tokenizer)
        max_new_tokens: max tokens to generate
        temperature: sampling temperature (0 = greedy)

    Returns:
        ConstrainedGenerationResult with full token-level metadata
    """
    device = model.cfg.device
    constraint = LLGuidanceConstraint(model_name, grammar, model_vocab_size=model.cfg.d_vocab)

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated_tokens: list[int] = []
    token_infos: list[TokenConstraintInfo] = []

    for step in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :]  # (1, vocab)

        # Apply grammar constraint
        mask, info = constraint.get_mask_and_info(device)
        logits[:, ~mask] = float("-inf")

        # Sample
        if temperature == 0:
            token_id = logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, 1).item()

        # Update info
        info.token_id = token_id
        info.token_str = constraint.llg_tok.decode_str([token_id])
        token_infos.append(info)
        generated_tokens.append(token_id)

        # Feed to grammar
        done = constraint.consume(token_id)
        if done:
            break

        # Extend input
        input_ids = torch.cat([
            input_ids,
            torch.tensor([[token_id]], dtype=torch.long, device=device)
        ], dim=1)

    all_tokens = prompt_tokens + generated_tokens
    decoded = constraint.llg_tok.decode_str(generated_tokens)

    # Try to parse as JSON
    parsed = None
    try:
        parsed = json.loads(decoded)
    except (json.JSONDecodeError, ValueError):
        pass

    return ConstrainedGenerationResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        all_tokens=all_tokens,
        token_info=token_infos,
        decoded_output=decoded,
        parsed_output=parsed,
    )


def unconstrained_generate(
    model,  # TransformerLens HookedTransformer
    prompt_tokens: list[int],
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    eos_token_id: Optional[int] = None,
) -> ConstrainedGenerationResult:
    """
    Run unconstrained (free-form) generation for baseline comparison.

    Returns the same result type for easy comparison with constrained_generate.
    """
    device = model.cfg.device
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    generated_tokens: list[int] = []
    token_infos: list[TokenConstraintInfo] = []

    for step in range(max_new_tokens):
        logits = model(input_ids)[:, -1, :]

        if temperature == 0:
            token_id = logits.argmax(dim=-1).item()
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, 1).item()

        info = TokenConstraintInfo(
            position=step,
            token_id=token_id,
            token_str="",  # filled below if we have a tokenizer
            allowed_count=logits.shape[-1],
            was_forced=False,
            is_schema_token=False,
        )
        token_infos.append(info)
        generated_tokens.append(token_id)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        input_ids = torch.cat([
            input_ids,
            torch.tensor([[token_id]], dtype=torch.long, device=device)
        ], dim=1)

    all_tokens = prompt_tokens + generated_tokens

    return ConstrainedGenerationResult(
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        all_tokens=all_tokens,
        token_info=token_infos,
        decoded_output="",  # caller can decode with their tokenizer
        parsed_output=None,
    )


def analyze_constraint_pattern(result: ConstrainedGenerationResult) -> dict:
    """
    Summarize the constraint pattern from a generation result.

    Returns a dict with:
        - total_tokens: total generated tokens
        - forced_tokens: number of tokens that were grammar-forced
        - free_tokens: number of tokens where model had real choice
        - schema_token_positions: list of positions that are structural
        - content_token_positions: list of positions that are model-chosen content
        - avg_allowed_at_content: average number of allowed tokens at content positions
    """
    forced = [t for t in result.token_info if t.was_forced]
    free = [t for t in result.token_info if not t.was_forced]
    schema_pos = [t.position for t in result.token_info if t.is_schema_token]
    content_pos = [t.position for t in result.token_info if not t.is_schema_token]

    avg_allowed = 0.0
    if free:
        avg_allowed = sum(t.allowed_count for t in free) / len(free)

    return {
        "total_tokens": len(result.token_info),
        "forced_tokens": len(forced),
        "free_tokens": len(free),
        "schema_token_positions": schema_pos,
        "content_token_positions": content_pos,
        "avg_allowed_at_content": avg_allowed,
        "per_token": [
            {
                "pos": t.position,
                "token": t.token_str,
                "allowed": t.allowed_count,
                "forced": t.was_forced,
                "schema": t.is_schema_token,
            }
            for t in result.token_info
        ],
    }
