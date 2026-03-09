#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

echo "=========================================="
echo "GLiNER CleanCoNLL Evaluation Pipeline"
echo "Config: $CONFIG"
echo "=========================================="

echo ""
echo "[1/4] Preparing data..."
python scripts/prepare_data.py --config "$CONFIG"

echo ""
echo "[2/4] Running GLiNER inference..."
python scripts/run_inference.py --config "$CONFIG"

echo ""
echo "[3/4] Evaluating predictions..."
python scripts/evaluate.py --config "$CONFIG"

echo ""
echo "[4/4] Generating report..."
python scripts/generate_report.py --config "$CONFIG"

echo ""
echo "=========================================="
echo "Pipeline complete! Results in results/"
echo "=========================================="
