"""Experiment config dataclasses and YAML loader.

The YAML config is the single source of truth for all experiment parameters.
No defaults exist in code — if a required field is missing from config, it errors.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_CONFIG = "config.yaml"


@dataclass
class ModelConfig:
    name: str
    dtype: str
    device: str | None = None
    trust_remote_code: bool = True
    prepend_bos: bool = True
    cache_strategy: str = "full"


@dataclass
class TransformerLensConfig:
    """Subset of ModelConfig fields relevant to TransformerLens model loading and tokenization.

    This is separate from ModelConfig to avoid coupling the rest of the codebase to TL-specific config fields.
    """
    fold_ln: bool = False
    center_writing_weights: bool = False
    center_unembed: bool = False
    refactor_factored_attn_matrices: bool = False
    fold_value_biases: bool = False
    default_prepend_bos: bool = True
    default_padding_side: str = "right"


@dataclass
class PromptsConfig:
    clean: str
    corrupted: str
    control: str | None = None


@dataclass
class TokensConfig:
    answer: str
    distractor: str | None = None


@dataclass
class PatchingConfig:
    metric: str = "logit"
    sweeps: list[str] = field(default_factory=lambda: [
        "resid_pre", "attn_out", "mlp_out", "head_out"
    ])


@dataclass
class OutputConfig:
    plot_root: str = "plots"
    save_cache: bool = False
    cache_dir: str = "cache"


@dataclass
class CorruptionConfig:
    type: str = "none"                          # "none" or "whitespace_injection"
    inject: list[str] = field(default_factory=lambda: ["\n", " "])  # tokens to inject
    count: int = 3                              # number of tokens to insert
    seed: int | None = 42                       # None for non-reproducible


@dataclass
class CorruptionSweepConfig:
    """A list of named corruption variants for experiment 04.

    Each variant is a dict with keys: label, type, inject, count, seed.
    Missing keys fall back to CorruptionConfig defaults.
    """
    variants: list = field(default_factory=list)

    def named_configs(self) -> list[tuple[str, "CorruptionConfig"]]:
        """Return (label, CorruptionConfig) pairs for each variant."""
        result = []
        for i, v in enumerate(self.variants):
            result.append((
                v.get("label", f"variant_{i}"),
                CorruptionConfig(
                    type=v.get("type", "whitespace_injection"),
                    inject=v.get("inject", ["\n", " "]),
                    count=v.get("count", 3),
                    seed=v.get("seed", None),
                ),
            ))
        return result


@dataclass
class Config:
    model: ModelConfig
    transformer_lens: TransformerLensConfig
    prompts: PromptsConfig
    tokens: TokensConfig
    patching: PatchingConfig
    output: OutputConfig
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    corruption_sweep: CorruptionSweepConfig = field(default_factory=CorruptionSweepConfig)


def _build(dataclass_type, yaml_dict: dict | None, section_name: str):
    """Create a dataclass from a YAML dict. Errors on missing required fields."""
    if not yaml_dict:
        yaml_dict = {}

    filtered = {k: v for k, v in yaml_dict.items() if v is not None and v != ""}
    known = {k: v for k, v in filtered.items() if k in dataclass_type.__dataclass_fields__}

    try:
        return dataclass_type(**known)
    except TypeError as e:
        print(f"Error in config section '{section_name}': {e}", file=sys.stderr)
        sys.exit(1)


def load_config(path: Path | str = DEFAULT_CONFIG) -> Config:
    """Load config from a YAML file and return a Config object."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}

    return Config(
        model=_build(ModelConfig, raw.get("model"), "model"),
        transformer_lens=_build(TransformerLensConfig, raw.get("transformer_lens"), "transformer_lens"),
        prompts=_build(PromptsConfig, raw.get("prompts"), "prompts"),
        tokens=_build(TokensConfig, raw.get("tokens"), "tokens"),
        patching=_build(PatchingConfig, raw.get("patching"), "patching"),
        output=_build(OutputConfig, raw.get("output"), "output"),
        corruption=_build(CorruptionConfig, raw.get("corruption"), "corruption"),
        corruption_sweep=_build(CorruptionSweepConfig, raw.get("corruption_sweep"), "corruption_sweep"),
    )
