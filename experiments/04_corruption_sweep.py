#!/usr/bin/env python3
"""Experiment 04: Sweep over whitespace injection variants.

Runs the clean prompt once, then N corruption variants, and compares
head attribution diffs across all of them. The summary heatmap shows
which heads are consistently disrupted vs. placement/type-dependent.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config, DEFAULT_CONFIG
from model import load_model, verify_tokenization, verify_answer_token, run_with_cache, corrupt_tokens
from attribution import head_attribution, final_logit_margin
from visualization import plot_sweep_summary, plot_head_attribution


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    p.add_argument("-k", "--top-heads", type=int, default=12,
                   help="Number of top heads to show in summary (default: 12)")
    args = p.parse_args()
    cfg = load_config(args.config)
    outdir = f"{cfg.output.plot_root}/04_corruption_sweep"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    variants = cfg.corruption_sweep.named_configs()
    if not variants:
        print("No variants defined in corruption_sweep.variants — edit config.yaml.")
        return

    model = load_model(cfg.model)
    bos = cfg.model.prepend_bos
    verify_answer_token(model, cfg.tokens.answer)
    distractors = [cfg.tokens.distractor] if cfg.tokens.distractor else None

    # --- Clean baseline ---
    print("\n--- Clean baseline ---")
    clean_tokens, _ = verify_tokenization(model, cfg.prompts.clean, prepend_bos=bos)
    clean_logits, clean_cache = run_with_cache(model, clean_tokens, strategy=cfg.model.cache_strategy)
    clean_margin = final_logit_margin(model, clean_logits, cfg.tokens.answer, distractors)
    print(f"  P({repr(cfg.tokens.answer)})={clean_margin['prob']:.4f}  "
          f"logit={clean_margin['logit']:.3f}  rank={clean_margin['rank']}")
    clean_attrs = head_attribution(model, clean_cache, cfg.tokens.answer, pos=-1)

    # --- Sweep variants ---
    variant_labels = []
    diffs = []
    probs = []

    for label, corruption_cfg in variants:
        print(f"\n  [{label}]  inject={corruption_cfg.inject!r}  "
              f"count={corruption_cfg.count}  seed={corruption_cfg.seed}")
        inj_tokens = corrupt_tokens(model, clean_tokens, corruption_cfg, prepend_bos=bos)
        logits, cache = run_with_cache(model, inj_tokens, strategy=cfg.model.cache_strategy)
        margin = final_logit_margin(model, logits, cfg.tokens.answer, distractors)
        print(f"    P({repr(cfg.tokens.answer)})={margin['prob']:.4f}  "
              f"logit={margin['logit']:.3f}  rank={margin['rank']}")

        attrs = head_attribution(model, cache, cfg.tokens.answer, pos=-1)
        diff = attrs - clean_attrs

        variant_labels.append(label)
        diffs.append(diff)
        probs.append(margin['prob'])

        plot_head_attribution(diff, f"{repr(cfg.tokens.answer)} ({label} - clean)",
                              outdir=outdir, filename=f"head_attr_diff_{label}.png")

    # --- Summary ---
    print("\n--- Summary: P(answer) across variants ---")
    print(f"  {'label':30s}  P(answer)")
    print(f"  {'clean':30s}  {clean_margin['prob']:.4f}")
    for label, prob in zip(variant_labels, probs):
        print(f"  {label:30s}  {prob:.4f}")

    plot_sweep_summary(
        variant_labels, diffs, clean_attrs,
        answer_label=repr(cfg.tokens.answer),
        k=args.top_heads,
        outdir=outdir,
        filename="sweep_summary.png",
    )

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
