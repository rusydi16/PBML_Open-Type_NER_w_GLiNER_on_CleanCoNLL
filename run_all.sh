#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
ABLATION_CONFIG="${ABLATION_CONFIG:-configs/ablation.yaml}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/finetune.yaml}"

# Parse flags
RUN_BOOTSTRAP=false
RUN_ABLATION=false
RUN_FINETUNE=false
RUN_FULL=false

for arg in "$@"; do
    case $arg in
        --full) RUN_FULL=true ;;
        --bootstrap) RUN_BOOTSTRAP=true ;;
        --ablation) RUN_ABLATION=true ;;
        --finetune) RUN_FINETUNE=true ;;
        *) CONFIG="$arg" ;;
    esac
done

if $RUN_FULL; then
    RUN_BOOTSTRAP=true
    RUN_ABLATION=true
    RUN_FINETUNE=true
fi

echo "=========================================="
echo "GLiNER CleanCoNLL Evaluation Pipeline"
echo "=========================================="

echo ""
echo "[1/4] Preparing data..."
python scripts/prepare_data.py --config "$CONFIG"

echo ""
echo "[2/4] Running GLiNER inference..."
python scripts/run_inference.py --config "$CONFIG"

echo ""
echo "[3/4] Evaluating predictions..."
if $RUN_BOOTSTRAP; then
    python scripts/evaluate.py --config "$CONFIG" --bootstrap
else
    python scripts/evaluate.py --config "$CONFIG"
fi

echo ""
echo "[4/4] Generating report..."
python scripts/generate_report.py --config "$CONFIG"

if $RUN_ABLATION; then
    echo ""
    echo "[Extra] Running model size ablation..."
    python scripts/run_ablation.py --config "$ABLATION_CONFIG"
fi

if $RUN_FINETUNE; then
    echo ""
    echo "[Extra] Running fine-tuning comparison..."
    python scripts/run_finetune.py --config "$FINETUNE_CONFIG"
fi

echo ""
echo "=========================================="
echo "Pipeline complete! Results in results/"
echo "=========================================="
