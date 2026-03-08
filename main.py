#!/usr/bin/env python3
"""Quick sanity check: load model, verify tokenization, inspect cache shapes."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from connectomics.config import load_config, DEFAULT_CONFIG
from connectomics.model import load_model, sanity_check, verify_answer_token


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", default=DEFAULT_CONFIG)
    args = p.parse_args()
    cfg = load_config(args.config)
    model = load_model(cfg.model)
    verify_answer_token(model, cfg.tokens.answer)
    sanity_check(model, cfg.prompts.clean, strategy=cfg.model.cache_strategy)


if __name__ == "__main__":
    main()
