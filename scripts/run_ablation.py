#!/usr/bin/env python3
"""CLI script for model size ablation study across GLiNER variants."""

import argparse
import gc
import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tqdm import tqdm

from src.data_utils import align_sentences_by_tokens, load_sentences_json
from src.inference import load_gliner_model, predict_sentence, set_seed
from src.metrics import (
    classify_errors,
    compute_entity_metrics_aggregated,
    compute_per_type_metrics_aggregated,
)
from src.noise_analysis import aggregate_noise_analysis, classify_noise_attribution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model size ablation study with GLiNER variants."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ablation.yaml",
        help="Path to ablation YAML config file (default: configs/ablation.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun each model even if its output files already exist",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    threshold = config["threshold"]
    gliner_labels = [entry["gliner_label"] for entry in config["labels"]]
    label_map = {entry["gliner_label"]: entry["conll_label"] for entry in config["labels"]}
    entity_types = [entry["conll_label"] for entry in config["labels"]]
    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]
    split = args.split
    datasets = ["conll03", "cleanconll"]

    # Load gold data for both datasets
    gold = {}
    for dataset in datasets:
        gold_path = os.path.join(processed_dir, f"{dataset}_{split}.json")
        print(f"Loading gold data: {gold_path}")
        gold[dataset] = load_sentences_json(gold_path)

    # Summary table rows
    summary_rows = []

    # Run ablation for each model
    for model_info in config["models"]:
        model_name = model_info["name"]
        short_name = model_info["short_name"]
        params = model_info["params"]

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({short_name}, {params})")
        print(f"{'='*60}")

        # Create output directory
        model_results_dir = os.path.join(results_dir, "ablation", short_name)
        os.makedirs(model_results_dir, exist_ok=True)

        # Resume: skip this model if all outputs exist
        metrics_paths = {
            d: os.path.join(model_results_dir, f"metrics_{d}_{split}.json")
            for d in datasets
        }
        noise_path = os.path.join(model_results_dir, f"noise_analysis_{split}.json")
        all_done = (
            not args.force
            and all(os.path.exists(p) for p in metrics_paths.values())
            and os.path.exists(noise_path)
        )
        if all_done:
            print(f"  All outputs exist for {short_name}, loading cached metrics.")
            dataset_f1 = {}
            for d, mp in metrics_paths.items():
                with open(mp, "r", encoding="utf-8") as f:
                    dataset_f1[d] = json.load(f)["overall"]["f1"]
            with open(noise_path, "r", encoding="utf-8") as f:
                noise_agg = json.load(f)
            f1_delta = dataset_f1.get("cleanconll", 0.0) - dataset_f1.get("conll03", 0.0)
            summary_rows.append({
                "model": short_name,
                "params": params,
                "conll03_f1": dataset_f1.get("conll03", 0.0),
                "cleanconll_f1": dataset_f1.get("cleanconll", 0.0),
                "f1_delta": round(f1_delta, 4),
                "noise_penalized": noise_agg.get("noise_penalized_correct", 0),
            })
            continue

        # Load model
        print(f"  Loading model: {model_name}")
        model = load_gliner_model(model_name)

        # Run inference and evaluation on both datasets
        dataset_f1 = {}
        predictions = {}

        for dataset in datasets:
            print(f"\n  Running inference on {dataset} ({split})...")
            sentences = gold[dataset]
            pred_sentences = []

            for sentence in tqdm(sentences, desc=f"    {dataset}"):
                preds = predict_sentence(
                    model,
                    sentence["tokens"],
                    gliner_labels,
                    label_map,
                    threshold=threshold,
                )
                pred_sentences.append({
                    "id": sentence["id"],
                    "tokens": sentence["tokens"],
                    "predictions": preds,
                })

            predictions[dataset] = pred_sentences

            # Save predictions
            pred_path = os.path.join(
                model_results_dir, f"predictions_{dataset}_{split}.json"
            )
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(pred_sentences, f, indent=2, ensure_ascii=False)
            print(f"  Saved predictions to: {pred_path}")

            # Evaluate — keep sentence boundaries so aggregation is correct.
            pred_by_id = {s["id"]: s["predictions"] for s in pred_sentences}
            per_sentence_pairs = []
            error_counts = {
                "type_error": 0,
                "boundary_error": 0,
                "type_boundary_error": 0,
                "missing": 0,
                "spurious": 0,
            }

            for sentence in sentences:
                sid = sentence["id"]
                gold_entities = sentence["entities"]
                pred_entities = pred_by_id.get(sid, [])
                per_sentence_pairs.append((gold_entities, pred_entities))

                sent_errors = classify_errors(gold_entities, pred_entities)
                for k in error_counts:
                    error_counts[k] += sent_errors[k]

            overall = compute_entity_metrics_aggregated(per_sentence_pairs)
            per_type = compute_per_type_metrics_aggregated(
                per_sentence_pairs, entity_types
            )
            dataset_f1[dataset] = overall["f1"]

            # Save metrics
            result = {
                "overall": overall,
                "per_type": per_type,
                "errors": error_counts,
            }
            metrics_path = os.path.join(
                model_results_dir, f"metrics_{dataset}_{split}.json"
            )
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"  Saved metrics to: {metrics_path}")
            print(
                f"  {dataset} - P: {overall['precision']:.4f}  "
                f"R: {overall['recall']:.4f}  F1: {overall['f1']:.4f}"
            )

        # Noise attribution analysis — align sentences by token content
        # (sequential IDs are not stable across CoNLL and CleanCoNLL).
        print(f"\n  Running noise attribution analysis...")
        pred_conll_by_id = {s["id"]: s["predictions"] for s in predictions["conll03"]}
        idx_conll, idx_clean = align_sentences_by_tokens(
            gold["conll03"], gold["cleanconll"]
        )

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

        noise_agg = aggregate_noise_analysis(per_sentence_results, max_examples=10)

        noise_path = os.path.join(
            model_results_dir, f"noise_analysis_{split}.json"
        )
        with open(noise_path, "w", encoding="utf-8") as f:
            json.dump(noise_agg, f, indent=2, ensure_ascii=False)
        print(f"  Saved noise analysis to: {noise_path}")

        # Record summary row
        f1_delta = dataset_f1.get("cleanconll", 0.0) - dataset_f1.get("conll03", 0.0)
        noise_penalized = noise_agg.get("noise_penalized_correct", 0)

        summary_rows.append({
            "model": short_name,
            "params": params,
            "conll03_f1": dataset_f1.get("conll03", 0.0),
            "cleanconll_f1": dataset_f1.get("cleanconll", 0.0),
            "f1_delta": round(f1_delta, 4),
            "noise_penalized": noise_penalized,
        })

        # Delete model to free memory
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        print(f"  Model {short_name} unloaded.")

    # Build summary table
    import pandas as pd

    df = pd.DataFrame(summary_rows)
    print(f"\n{'='*60}")
    print("Ablation Study Summary")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    # Save summary table
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "ablation_table.csv")
    md_path = os.path.join(results_dir, "ablation_table.md")

    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV table to: {csv_path}")

    df.to_markdown(md_path, index=False)
    print(f"Saved Markdown table to: {md_path}")

    print("\nAblation study complete.")


if __name__ == "__main__":
    main()
