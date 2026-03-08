#!/usr/bin/env python3
"""Experiment 03: Activation patching to localize OOD disruption."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from connectomics.config import load_config, DEFAULT_CONFIG
from connectomics.model import load_model, verify_tokenization, verify_answer_token
from connectomics.patching import (
    make_logit_diff_metric, make_prob_metric,
    compare_clean_corrupted,
    patch_resid_pre, patch_attn_out, patch_mlp_out, patch_head_out,
)
from connectomics.visualization import (
    plot_tokenization_panel, plot_patch_heatmap, plot_head_patch_heatmap,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    args = p.parse_args()
    cfg = load_config(args.config)
    outdir = f"{cfg.output.plot_root}/03_patching"
    Path(outdir).mkdir(parents=True, exist_ok=True)

    model = load_model(cfg.model)
    bos = cfg.model.prepend_bos

    print("\n--- Clean prompt ---")
    clean_tokens, clean_str = verify_tokenization(model, cfg.prompts.clean, prepend_bos=bos)
    print("\n--- Corrupted prompt ---")
    corrupt_tokens, corrupt_str = verify_tokenization(model, cfg.prompts.corrupted, prepend_bos=bos)

    suffix_start = clean_tokens.shape[1]
    plot_tokenization_panel(corrupt_str, corrupt_tokens, suffix_start=suffix_start,
                            title=f"Corrupted: {repr(cfg.prompts.corrupted)}",
                            outdir=outdir, filename="tokenization_corrupted.png")
    verify_answer_token(model, cfg.tokens.answer)

    # Metric
    if cfg.patching.metric == "logit_diff" and cfg.tokens.distractor:
        metric_fn = make_logit_diff_metric(model, cfg.tokens.answer, cfg.tokens.distractor)
    else:
        metric_fn = make_logit_diff_metric(model, cfg.tokens.answer)

    # Baseline
    print("\n--- Baseline comparison ---")
    baseline = compare_clean_corrupted(model, clean_tokens, corrupt_tokens, metric_fn)
    print(f"Clean metric:     {baseline['clean_score']:.4f}")
    print(f"Corrupted metric: {baseline['corrupted_score']:.4f}")
    print(f"Delta:            {baseline['delta']:.4f}")
    clean_cache = baseline["clean_cache"]

    # Shared token labels for heatmaps
    n_shared = min(clean_tokens.shape[1], corrupt_tokens.shape[1])
    shared_str = corrupt_str[:n_shared]

    # Run configured sweeps
    sweeps = cfg.patching.sweeps
    sweep_runners = {
        "resid_pre": (patch_resid_pre, "Resid-pre patching"),
        "attn_out": (patch_attn_out, "Attn-out patching"),
        "mlp_out": (patch_mlp_out, "MLP-out patching"),
    }

    sweep_results = {}
    for sweep_name in sweeps:
        if sweep_name in sweep_runners:
            fn, title = sweep_runners[sweep_name]
            print(f"\n--- {title} ---")
            scores = fn(model, corrupt_tokens, clean_cache, metric_fn)
            sweep_results[sweep_name] = scores
            plot_patch_heatmap(scores, shared_str, title=title,
                               outdir=outdir, filename=f"patch_{sweep_name}.png")

    if "resid_pre" in sweep_results:
        scores = sweep_results["resid_pre"]
        flat = scores.flatten()
        top_idx = flat.argsort(descending=True)[:10]
        n_pos = scores.shape[1]
        print(f"\nTop 10 rescuing (layer, position) for resid_pre:")
        for idx in top_idx:
            li, pi = idx.item() // n_pos, idx.item() % n_pos
            tok = shared_str[pi] if pi < len(shared_str) else "?"
            print(f"  Layer {li:2d}  Pos {pi:2d} ({repr(tok):10s})  score={flat[idx].item():.4f}")

    if "head_out" in sweeps:
        print("\n--- Per-head patching ---")
        head_scores = patch_head_out(model, corrupt_tokens, clean_cache, metric_fn)
        plot_head_patch_heatmap(head_scores, title="Per-head patching rescue",
                                outdir=outdir, filename="patch_head_out.png")

        head_summed = head_scores.sum(dim=-1)
        n_heads = head_summed.shape[1]
        flat_heads = head_summed.flatten()
        top_heads = flat_heads.argsort(descending=True)[:10]
        print(f"\nTop 10 rescuing heads:")
        for idx in top_heads:
            li, hi = idx.item() // n_heads, idx.item() % n_heads
            print(f"  L{li:2d} H{hi:2d}  rescue={flat_heads[idx].item():.4f}")

    # Summary
    print(f"\n--- Summary ---")
    print(f"Clean: {baseline['clean_score']:.4f}  Corrupted: {baseline['corrupted_score']:.4f}")
    if "attn_out" in sweep_results and "mlp_out" in sweep_results:
        attn_max = sweep_results["attn_out"].max().item()
        mlp_max = sweep_results["mlp_out"].max().item()
        print(f"Max attn-out rescue: {attn_max:.4f}  Max mlp-out rescue: {mlp_max:.4f}")
        print(f"-> Damage primarily through {'attention routing' if attn_max > mlp_max else 'MLP writing'}")

    print(f"\nAll plots saved to {outdir}/")


if __name__ == "__main__":
    main()
