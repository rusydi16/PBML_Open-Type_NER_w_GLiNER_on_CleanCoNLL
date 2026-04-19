#!/usr/bin/env python3
"""CLI script that runs GLiNER inference on processed datasets."""

import argparse
import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tqdm import tqdm

from src.data_utils import load_sentences_json
from src.inference import load_gliner_model, predict_sentence, set_seed


def _save_atomic(path: str, data) -> None:
    """Write JSON atomically via temp + rename to prevent partial-write corruption."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_partial_predictions(path: str):
    """Load existing predictions file. Returns (results_list, set_of_processed_ids)."""
    if not os.path.exists(path):
        return [], set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        return results, {r["id"] for r in results}
    except (json.JSONDecodeError, KeyError, TypeError):
        print(f"  WARNING: {path} is unreadable, starting fresh.")
        return [], set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GLiNER inference on processed datasets."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["conll03", "cleanconll"],
        help="Dataset names to run inference on (default: conll03 cleanconll)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Data split to use (default: test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing predictions instead of resuming",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Write partial predictions every N sentences (default: 500)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    set_seed(config["seed"])

    # Build label mapping from config
    gliner_labels = [entry["gliner_label"] for entry in config["labels"]]
    label_map = {
        entry["gliner_label"]: entry["conll_label"] for entry in config["labels"]
    }

    threshold = config["model"]["threshold"]

    # GLiNER model is loaded lazily on first use so that fully-resumed runs
    # avoid paying the model-load cost.
    model = None

    def _get_model():
        nonlocal model
        if model is None:
            print(f"Loading model: {config['model']['name']}")
            model = load_gliner_model(config["model"]["name"])
            print("Model loaded successfully.")
        return model

    # Process each dataset
    for dataset in args.datasets:
        input_path = os.path.join(
            config["paths"]["processed_data"], f"{dataset}_{args.split}.json"
        )
        output_dir = config["paths"]["results"]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"predictions_{dataset}_{args.split}.json"
        )

        print(f"\nDataset: {dataset} | Split: {args.split}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")

        sentences = load_sentences_json(input_path)

        if args.force:
            results, processed_ids = [], set()
        else:
            results, processed_ids = _load_partial_predictions(output_path)
            if processed_ids:
                print(f"  Resuming: {len(processed_ids)} sentences already predicted.")

        to_process = [s for s in sentences if s["id"] not in processed_ids]
        if not to_process:
            print(f"  All {len(sentences)} sentences already processed, skipping inference.")
            total_predictions = sum(len(r["predictions"]) for r in results)
            print(f"  Sentences: {len(results)}  Predictions: {total_predictions}")
            continue

        total_predictions = sum(len(r["predictions"]) for r in results)

        active_model = _get_model()
        for i, sentence in enumerate(tqdm(to_process, desc=f"Predicting ({dataset})")):
            tokens = sentence["tokens"]
            predictions = predict_sentence(
                active_model, tokens, gliner_labels, label_map, threshold
            )
            total_predictions += len(predictions)
            results.append(
                {
                    "id": sentence["id"],
                    "tokens": tokens,
                    "predictions": predictions,
                }
            )

            if (i + 1) % args.checkpoint_every == 0:
                _save_atomic(output_path, results)

        _save_atomic(output_path, results)

        print(f"  Sentences: {len(results)}")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Saved to: {output_path}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
