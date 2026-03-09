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

    # Load GLiNER model
    print(f"Loading model: {config['model']['name']}")
    model = load_gliner_model(config["model"]["name"])
    print("Model loaded successfully.")

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
        results = []
        total_predictions = 0

        for sentence in tqdm(sentences, desc=f"Predicting ({dataset})"):
            tokens = sentence["tokens"]
            predictions = predict_sentence(
                model, tokens, gliner_labels, label_map, threshold
            )
            total_predictions += len(predictions)
            results.append(
                {
                    "id": sentence["id"],
                    "tokens": tokens,
                    "predictions": predictions,
                }
            )

        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"  Sentences: {len(results)}")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Saved to: {output_path}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
