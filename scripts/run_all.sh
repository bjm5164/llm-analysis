#!/bin/bash
# Run the full experiment pipeline.
# All params come from config.yaml. Override with CLI flags or -c:
#   ./scripts/run_all.sh
#   ./scripts/run_all.sh --model gpt2
#   ./scripts/run_all.sh -c experiments/my_config.yaml
set -eu

echo "=== Step 1: Clean Baseline ==="
python experiments/01_clean_baseline.py "$@"

echo ""
echo "=== Step 2: OOD Comparison ==="
python experiments/02_ood_comparison.py "$@"

echo ""
echo "=== Step 3: Activation Patching ==="
python experiments/03_activation_patching.py "$@"

echo ""
echo "=== Pipeline Complete ==="
