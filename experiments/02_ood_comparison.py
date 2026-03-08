#!/usr/bin/env python3
"""Experiment 02: Compare clean vs OOD-corrupted prompts."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from connectomics.config import load_config, DEFAULT_CONFIG
from connectomics.model import load_model, verify_tokenization, verify_answer_token, run_with_cache, corrupt_tokens
from connectomics.attribution import (
    final_logit_margin, head_attribution, print_top_heads,
)
from connectomics.visualization import (
    plot_tokenization_panel, plot_head_attribution, plot_attention_pattern,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    args = p.parse_args()
    cfg = load_config(args.config)
    outdir = f"{cfg.output.plot_root}/02_ood_comparison"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    model = load_model(cfg.model)
    bos = cfg.model.prepend_bos
    verify_answer_token(model, cfg.tokens.answer)

    # Build variants: (label, tokens, str_tokens, suffix_start)
    raw = {"clean": cfg.prompts.clean, "corrupted": cfg.prompts.corrupted}
    if cfg.prompts.control:
        raw["control"] = cfg.prompts.control

    clean_len = model.to_tokens(cfg.prompts.clean, prepend_bos=bos).shape[1]
    variants = []
    for label, prompt in raw.items():
        toks, str_toks = verify_tokenization(model, prompt, prepend_bos=bos)
        suffix_start = None if label == "clean" else clean_len
        variants.append((label, toks, str_toks, suffix_start))

    if cfg.corruption.type != "none":
        clean_toks = variants[0][1]
        inj_toks = corrupt_tokens(model, clean_toks, cfg.corruption, prepend_bos=bos)
        inj_str = [model.tokenizer.decode(t.item()) for t in inj_toks[0]]
        print(f"\nWhitespace injection: {cfg.corruption.count} token(s) from "
              f"{cfg.corruption.inject!r}, seed={cfg.corruption.seed}")
        variants.append(("whitespace_injected", inj_toks, inj_str, None))

    results = {}
    for label, tokens, str_tokens, suffix_start in variants:
        print(f"\n{'='*40}\n  {label.upper()}\n{'='*40}")
        logits, cache = run_with_cache(model, tokens, strategy=cfg.model.cache_strategy)

        plot_tokenization_panel(str_tokens, tokens, suffix_start=suffix_start,
                                title=label,
                                outdir=outdir, filename=f"tokenization_{label}.png")

        distractors = [cfg.tokens.distractor] if cfg.tokens.distractor else None
        margin = final_logit_margin(model, logits, cfg.tokens.answer, distractors)
        print(f"  P({repr(cfg.tokens.answer)})={margin['prob']:.4f}  rank={margin['rank']}")

        results[label] = {"tokens": tokens, "str_tokens": str_tokens,
                          "logits": logits, "cache": cache}

    # Head attribution
    print("\n--- Head attribution comparison ---")
    head_results = {}
    for label, r in results.items():
        ha = head_attribution(model, r["cache"], cfg.tokens.answer, pos=-1)
        head_results[label] = ha
        plot_head_attribution(ha, f"{repr(cfg.tokens.answer)} ({label})",
                              outdir=outdir, filename=f"head_attribution_{label}.png")

    diff = head_results["corrupted"] - head_results["clean"]
    plot_head_attribution(diff, f"{repr(cfg.tokens.answer)} (corrupted - clean)",
                          outdir=outdir, filename="head_attribution_diff.png")
    print_top_heads(diff, f"{repr(cfg.tokens.answer)} change (corrupted - clean)")

    # Top changed heads attention patterns
    n_heads = diff.shape[1]
    flat = diff.flatten()
    top_idx = flat.abs().argsort(descending=True)[:4]
    for rank, idx in enumerate(top_idx):
        li, hi = idx.item() // n_heads, idx.item() % n_heads
        for label, r in results.items():
            plot_attention_pattern(r["cache"], li, hi, r["str_tokens"],
                                  title=f"L{li} H{hi} ({label})",
                                  outdir=outdir,
                                  filename=f"changed_head_{rank}_L{li}_H{hi}_{label}.png")

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
