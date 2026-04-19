#!/usr/bin/env python3
"""Pipeline orchestrator for the GLiNER x CleanCoNLL evaluation study.

Cross-platform replacement for ``run_all.sh``. Runs the stages:

    prepare_data -> run_inference -> evaluate -> generate_report

with optional ``--bootstrap`` / ``--ablation`` / ``--finetune`` / ``--full``
add-ons. Uses ``sys.executable`` so the current venv's Python is always used
without any activation step.

Examples::

    python run_all.py                  # core pipeline
    python run_all.py --bootstrap      # + significance testing
    python run_all.py --ablation       # + model size ablation
    python run_all.py --finetune       # + fine-tuning comparison
    python run_all.py --full           # all of the above

Override configs via CLI argument or environment variables::

    python run_all.py configs/alt.yaml
    CONFIG=configs/alt.yaml python run_all.py
    ABLATION_CONFIG=... FINETUNE_CONFIG=... python run_all.py --full
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run_stage(label: str, cmd: list[str]) -> None:
    print()
    print(label)
    print("  $ " + " ".join(repr(c) if " " in c else c for c in cmd))
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        print(
            f"\nStage failed (exit {result.returncode}): {label}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Override default config path (default: configs/default.yaml or $CONFIG)",
    )
    parser.add_argument("--full", action="store_true",
                        help="Shorthand for --bootstrap --ablation --finetune --baseline --eda")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Add paired-bootstrap significance testing")
    parser.add_argument("--ablation", action="store_true",
                        help="Add model-size ablation (small/medium/large)")
    parser.add_argument("--finetune", action="store_true",
                        help="Add fine-tuning comparison (CoNLL-03 vs CleanCoNLL)")
    parser.add_argument("--baseline", action="store_true",
                        help="Add non-GLiNER baseline comparison (BERT-NER)")
    parser.add_argument("--eda", action="store_true",
                        help="Regenerate docs/eda_summary.md")
    args = parser.parse_args()

    config = args.config or os.environ.get("CONFIG", "configs/default.yaml")
    ablation_config = os.environ.get("ABLATION_CONFIG", "configs/ablation.yaml")
    finetune_config = os.environ.get("FINETUNE_CONFIG", "configs/finetune.yaml")
    baseline_config = os.environ.get("BASELINE_CONFIG", "configs/baseline.yaml")

    run_bootstrap = args.bootstrap or args.full
    run_ablation = args.ablation or args.full
    run_finetune = args.finetune or args.full
    run_baseline = args.baseline or args.full
    run_eda = args.eda or args.full

    py = sys.executable  # current interpreter -> automatic venv awareness

    banner = "=" * 42
    print(banner)
    print("GLiNER CleanCoNLL Evaluation Pipeline")
    print(banner)
    print(f"  python : {py}")
    print(f"  config : {config}")

    run_stage(
        "[1/4] Preparing data...",
        [py, "scripts/prepare_data.py", "--config", config],
    )
    run_stage(
        "[2/4] Running GLiNER inference...",
        [py, "scripts/run_inference.py", "--config", config],
    )

    evaluate_cmd = [py, "scripts/evaluate.py", "--config", config]
    if run_bootstrap:
        evaluate_cmd.append("--bootstrap")
    run_stage("[3/4] Evaluating predictions...", evaluate_cmd)

    run_stage(
        "[4/4] Generating report...",
        [py, "scripts/generate_report.py", "--config", config],
    )

    if run_ablation:
        run_stage(
            "[Extra] Running model size ablation...",
            [py, "scripts/run_ablation.py", "--config", ablation_config],
        )

    if run_finetune:
        run_stage(
            "[Extra] Running fine-tuning comparison...",
            [py, "scripts/run_finetune.py", "--config", finetune_config],
        )

    if run_baseline:
        run_stage(
            "[Extra] Running non-GLiNER baseline comparison...",
            [py, "scripts/run_baseline.py", "--config", baseline_config],
        )

    if run_eda:
        run_stage(
            "[Extra] Generating EDA report...",
            [py, "scripts/run_eda.py", "--config", config],
        )

    print()
    print(banner)
    print("Pipeline complete! Results in results/")
    print(banner)


if __name__ == "__main__":
    main()
