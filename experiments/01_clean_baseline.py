#!/usr/bin/env python3
"""Experiment 01: Clean prompt baseline — where does the model write the answer?"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config, DEFAULT_CONFIG
from model import load_model, verify_tokenization, verify_answer_token, run_with_cache
from attribution import (
    final_logit_margin, component_attribution, head_attribution,
    print_top_components, print_top_heads,
)
from visualization import (
    plot_tokenization_panel, plot_component_attribution,
    plot_head_attribution, plot_top_head_attention,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    args = p.parse_args()
    cfg = load_config(args.config)
    outdir = f"{cfg.output.plot_root}/01_baseline"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    model = load_model(cfg.model)
    bos = cfg.model.prepend_bos

    print("\n--- Tokenization ---")
    tokens, str_tokens = verify_tokenization(model, cfg.prompts.clean, prepend_bos=bos)
    answer_id = verify_answer_token(model, cfg.tokens.answer)
    plot_tokenization_panel(str_tokens, tokens,
                            title=f"Prompt: {repr(cfg.prompts.clean)}",
                            outdir=outdir, filename="tokenization.png")

    print("\n--- Forward pass ---")
    logits, cache = run_with_cache(model, tokens, strategy=cfg.model.cache_strategy)
    distractors = [cfg.tokens.distractor] if cfg.tokens.distractor else None
    margin = final_logit_margin(model, logits, cfg.tokens.answer, distractors)
    print("Final logit margin:")
    for k, v in margin.items():
        print(f"  {k}: {v}")

    print("\n--- Component attribution ---")
    attrs, labels = component_attribution(model, cache, cfg.tokens.answer, pos=-1)
    print_top_components(attrs, labels, repr(cfg.tokens.answer))
    plot_component_attribution(attrs, labels, repr(cfg.tokens.answer),
                               outdir=outdir, filename="component_attribution.png")

    print("\n--- Per-head attribution ---")
    head_attrs = head_attribution(model, cache, cfg.tokens.answer, pos=-1)
    print_top_heads(head_attrs, repr(cfg.tokens.answer))
    plot_head_attribution(head_attrs, repr(cfg.tokens.answer),
                          outdir=outdir, filename="head_attribution.png")
    plot_top_head_attention(model, cache, head_attrs, str_tokens, k=4,
                            outdir=outdir, filename_prefix="top_head_attn")

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
