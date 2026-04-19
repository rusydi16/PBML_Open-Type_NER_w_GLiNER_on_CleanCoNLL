#!/usr/bin/env python3
"""CLI script that evaluates GLiNER predictions against gold annotations."""

import argparse
import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.data_utils import align_sentences_by_tokens, load_sentences_json
from src.metrics import (
    classify_errors,
    compute_entity_metrics_aggregated,
    compute_per_type_metrics_aggregated,
)
from src.noise_analysis import aggregate_noise_analysis, classify_noise_attribution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate GLiNER predictions against gold annotations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Run bootstrap significance testing (slower)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    entity_types = [entry["conll_label"] for entry in config["labels"]]
    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]

    split = args.split
    datasets = ["conll03", "cleanconll"]

    # Load gold and prediction files
    gold = {}
    pred = {}
    for dataset in datasets:
        gold_path = os.path.join(processed_dir, f"{dataset}_{split}.json")
        pred_path = os.path.join(results_dir, f"predictions_{dataset}_{split}.json")
        print(f"Loading {gold_path}")
        gold[dataset] = load_sentences_json(gold_path)
        print(f"Loading {pred_path}")
        pred[dataset] = load_sentences_json(pred_path)

    # Evaluate each dataset
    dataset_f1 = {}
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset} ({split})")
        print(f"{'='*60}")

        # Build pred_by_id dict
        pred_by_id = {s["id"]: s["predictions"] for s in pred[dataset]}

        # Collect per-sentence gold/pred pairs (preserves sentence boundaries
        # so entity tuples don't collide across sentences during aggregation).
        per_sentence_pairs = []
        error_counts = {
            "type_error": 0,
            "boundary_error": 0,
            "type_boundary_error": 0,
            "missing": 0,
            "spurious": 0,
        }

        for sentence in gold[dataset]:
            sid = sentence["id"]
            gold_entities = sentence["entities"]
            pred_entities = pred_by_id.get(sid, [])

            per_sentence_pairs.append((gold_entities, pred_entities))

            # Accumulate error classification
            sent_errors = classify_errors(gold_entities, pred_entities)
            for k in error_counts:
                error_counts[k] += sent_errors[k]

        # Compute overall + per-type metrics with per-sentence aggregation.
        overall = compute_entity_metrics_aggregated(per_sentence_pairs)
        per_type = compute_per_type_metrics_aggregated(per_sentence_pairs, entity_types)

        dataset_f1[dataset] = overall["f1"]

        # Save results
        result = {
            "overall": overall,
            "per_type": per_type,
            "errors": error_counts,
        }
        os.makedirs(results_dir, exist_ok=True)
        metrics_path = os.path.join(results_dir, f"metrics_{dataset}_{split}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Saved metrics to: {metrics_path}")

        # Print summary
        print(f"  Overall - P: {overall['precision']:.4f}  R: {overall['recall']:.4f}  F1: {overall['f1']:.4f}")
        for etype in entity_types:
            m = per_type[etype]
            print(f"  {etype:>6s} - P: {m['precision']:.4f}  R: {m['recall']:.4f}  F1: {m['f1']:.4f}")
        print(f"  Errors: {error_counts}")

    # Noise attribution
    print(f"\n{'='*60}")
    print(f"Noise Attribution Analysis ({split})")
    print(f"{'='*60}")

    # Sentences must be aligned by token content (not by sequential ID),
    # because CleanCoNLL drops/splits some sentences so position-based IDs
    # reference different content across the two datasets.
    idx_conll, idx_clean = align_sentences_by_tokens(gold["conll03"], gold["cleanconll"])
    pred_conll_by_id = {s["id"]: s["predictions"] for s in pred["conll03"]}

    per_sentence_results = []
    for ci, kk in zip(idx_conll, idx_clean):
        conll_sent = gold["conll03"][ci]
        clean_sent = gold["cleanconll"][kk]
        pred_entities = pred_conll_by_id.get(conll_sent["id"], [])
        result = classify_noise_attribution(
            pred_entities,
            conll_sent["entities"],
            clean_sent["entities"],
        )
        per_sentence_results.append(result)

    print(
        f"  Aligned {len(per_sentence_results)} sentences by token content "
        f"(of {len(gold['conll03'])} conll / {len(gold['cleanconll'])} clean)."
    )

    noise_agg = aggregate_noise_analysis(per_sentence_results, max_examples=10)

    noise_path = os.path.join(results_dir, f"noise_analysis_{split}.json")
    with open(noise_path, "w", encoding="utf-8") as f:
        json.dump(noise_agg, f, indent=2, ensure_ascii=False)
    print(f"  Saved noise analysis to: {noise_path}")

    # Print summary
    print(f"\n  F1 Scores:")
    for dataset in datasets:
        print(f"    {dataset}: {dataset_f1[dataset]:.4f}")

    print(f"\n  Noise Attribution Counts:")
    count_keys = [
        "correct_both", "noise_penalized_correct", "model_learned_noise",
        "genuine_error", "missed_both", "missed_conll_only", "missed_clean_only",
    ]
    for k in count_keys:
        print(f"    {k}: {noise_agg[k]}")

    # Bootstrap significance testing (optional)
    if args.bootstrap:
        from src.statistical_tests import bootstrap_entity_f1, paired_bootstrap_test

        print(f"\n{'='*60}")
        print(f"Bootstrap Significance Testing ({split})")
        print(f"{'='*60}")

        # Adapt pred entries: rename "predictions" -> "entities" for bootstrap functions
        pred_adapted = {}
        for dataset in datasets:
            pred_adapted[dataset] = [
                {"id": s["id"], "entities": s["predictions"]} for s in pred[dataset]
            ]

        bootstrap_results = {}
        for dataset in datasets:
            print(f"  Running bootstrap for {dataset} (n={args.n_bootstrap})...")
            bs = bootstrap_entity_f1(
                gold[dataset],
                pred_adapted[dataset],
                n_iterations=args.n_bootstrap,
                seed=config["seed"],
            )
            bootstrap_results[dataset] = bs

            bs_path = os.path.join(results_dir, f"bootstrap_{dataset}_{split}.json")
            with open(bs_path, "w", encoding="utf-8") as f:
                json.dump(bs, f, indent=2, ensure_ascii=False)
            print(f"    95% CI: [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")

        # Paired bootstrap test: conll03 vs cleanconll
        print(f"  Running paired bootstrap test...")
        sig = paired_bootstrap_test(
            gold["conll03"],
            pred_adapted["conll03"],
            gold["cleanconll"],
            pred_adapted["cleanconll"],
            n_iterations=args.n_bootstrap,
            seed=config["seed"],
        )
        sig_path = os.path.join(results_dir, f"significance_test_{split}.json")
        with open(sig_path, "w", encoding="utf-8") as f:
            json.dump(sig, f, indent=2, ensure_ascii=False)
        print(f"    Delta (cleanconll - conll03): {sig['delta_mean']:.4f} (p={sig['p_value']:.4f})")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
