#!/usr/bin/env python3
"""CLI script that fine-tunes GLiNER on CoNLL-03 vs CleanCoNLL and evaluates both on CleanCoNLL test set."""

import argparse
import gc
import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml
from tqdm import tqdm

from src.data_utils import load_sentences_json
from src.finetune import convert_dataset_to_gliner_format, finetune_gliner
from src.inference import load_gliner_model, predict_sentence, set_seed
from src.metrics import (
    classify_errors,
    compute_entity_metrics_aggregated,
    compute_per_type_metrics_aggregated,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune GLiNER on CoNLL-03 vs CleanCoNLL and evaluate on CleanCoNLL test set."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/finetune.yaml",
        help="Path to YAML config file (default: configs/finetune.yaml)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split to use (default: test)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Globally skip training and load existing fine-tuned models",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even when a fine-tuned model already exists on disk",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    set_seed(config["seed"])

    # Build label mappings from config
    gliner_labels = [entry["gliner_label"] for entry in config["labels"]]
    label_map = {
        entry["gliner_label"]: entry["conll_label"] for entry in config["labels"]
    }
    conll_to_gliner = {
        entry["conll_label"]: entry["gliner_label"] for entry in config["labels"]
    }
    entity_types = [entry["conll_label"] for entry in config["labels"]]

    threshold = config["threshold"]
    processed_data = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]
    models_dir = config["paths"]["models"]

    # Load CleanCoNLL test set as the evaluation gold standard
    eval_path = os.path.join(processed_data, f"cleanconll_{args.split}.json")
    print(f"Loading evaluation data: {eval_path}")
    eval_sentences = load_sentences_json(eval_path)
    print(f"  Loaded {len(eval_sentences)} evaluation sentences.")

    # Output directory for finetune results
    finetune_results_dir = os.path.join(results_dir, "finetune")
    os.makedirs(finetune_results_dir, exist_ok=True)

    # Training configurations
    training_configs = [
        {"name": "conll03", "display": "CoNLL-03"},
        {"name": "cleanconll", "display": "CleanCoNLL"},
    ]

    comparison_rows = []

    def _model_dir_has_weights(dirpath: str) -> bool:
        """A model dir counts as 'already trained' if it holds model weights."""
        if not os.path.isdir(dirpath):
            return False
        for fname in os.listdir(dirpath):
            if fname.endswith((".safetensors", ".bin", ".pt")):
                return True
        return False

    for tc in training_configs:
        name = tc["name"]
        display = tc["display"]
        model_dir = os.path.join(models_dir, f"finetuned_{name}")

        print(f"\n{'=' * 60}")
        print(f"Training config: {display} ({name})")
        print(f"{'=' * 60}")

        # Per-config resume decision
        model_exists = _model_dir_has_weights(model_dir)
        if args.force_retrain:
            should_train = True
        elif args.skip_training:
            should_train = False
        else:
            should_train = not model_exists
            if model_exists:
                print(f"  Detected existing weights in {model_dir}, skipping training.")

        # --- Training or loading ---
        if should_train:
            # Load training sentences.
            train_path = os.path.join(processed_data, f"{name}_train.json")
            print(f"Loading training data: {train_path}")
            train_sentences = load_sentences_json(train_path)
            print(f"  Loaded {len(train_sentences)} training sentences.")

            # Load matching dev split as eval data — required by GLiNER 0.2+.
            # Using the same distribution as training is the honest setup for
            # in-training validation / save_steps checkpoint selection.
            dev_path = os.path.join(processed_data, f"{name}_dev.json")
            print(f"Loading eval data: {dev_path}")
            dev_sentences = load_sentences_json(dev_path)
            print(f"  Loaded {len(dev_sentences)} eval sentences.")

            # Convert both to GLiNER format.
            print("Converting to GLiNER format...")
            train_data = convert_dataset_to_gliner_format(train_sentences, conll_to_gliner)
            dev_data = convert_dataset_to_gliner_format(dev_sentences, conll_to_gliner)
            print(f"  Converted {len(train_data)} train / {len(dev_data)} eval samples.")

            # Fine-tune.
            print(f"Fine-tuning {config['base_model']} ...")
            os.makedirs(model_dir, exist_ok=True)
            training_cfg = config["training"]
            model = finetune_gliner(
                model_name=config["base_model"],
                train_data=train_data,
                eval_data=dev_data,
                output_dir=model_dir,
                max_steps=training_cfg["max_steps"],
                learning_rate=float(training_cfg["learning_rate"]),
                batch_size=training_cfg["batch_size"],
                warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
                seed=config["seed"],
            )
            print(f"  Model saved to: {model_dir}")
        else:
            # Load existing fine-tuned model
            print(f"Loading existing model from: {model_dir}")
            model = load_gliner_model(model_dir)
            print("  Model loaded successfully.")

        # --- Evaluation on CleanCoNLL test ---
        print(f"\nEvaluating on CleanCoNLL {args.split} set...")
        per_sentence_pairs = []
        prediction_results = []
        errors = {
            "type_error": 0,
            "boundary_error": 0,
            "type_boundary_error": 0,
            "missing": 0,
            "spurious": 0,
        }

        for sentence in tqdm(eval_sentences, desc=f"Evaluating ({name})"):
            tokens = sentence["tokens"]
            gold_entities = sentence["entities"]
            predictions = predict_sentence(
                model, tokens, gliner_labels, label_map, threshold
            )
            per_sentence_pairs.append((gold_entities, predictions))
            for k, v in classify_errors(gold_entities, predictions).items():
                errors[k] += v
            prediction_results.append({
                "id": sentence["id"],
                "tokens": tokens,
                "gold_entities": gold_entities,
                "predictions": predictions,
            })

        # Compute metrics with proper per-sentence aggregation.
        metrics = compute_entity_metrics_aggregated(per_sentence_pairs)
        per_type = compute_per_type_metrics_aggregated(per_sentence_pairs, entity_types)

        print(f"\n  Results for finetuned_{name} on CleanCoNLL {args.split}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1:        {metrics['f1']:.4f}")
        print(f"    TP: {metrics['tp']}  FP: {metrics['fp']}  FN: {metrics['fn']}")

        # Save metrics
        metrics_output = {
            "model": f"finetuned_{name}",
            "eval_split": f"cleanconll_{args.split}",
            "overall": metrics,
            "per_type": per_type,
            "errors": errors,
        }
        metrics_path = os.path.join(finetune_results_dir, f"metrics_finetuned_{name}.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_output, f, indent=2, ensure_ascii=False)
        print(f"    Metrics saved to: {metrics_path}")

        # Save predictions
        preds_path = os.path.join(finetune_results_dir, f"predictions_finetuned_{name}.json")
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(prediction_results, f, indent=2, ensure_ascii=False)
        print(f"    Predictions saved to: {preds_path}")

        # Record comparison row
        row = {
            "model": f"finetuned_{name}",
            "train_data": display,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
        }
        for etype in entity_types:
            row[f"{etype}_f1"] = per_type[etype]["f1"]
        for err_type, count in errors.items():
            row[err_type] = count
        comparison_rows.append(row)

        # Delete model to free memory
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # Build comparison table
    print(f"\n{'=' * 60}")
    print("Comparison Table")
    print(f"{'=' * 60}")

    df = pd.DataFrame(comparison_rows)

    # Save as CSV
    csv_path = os.path.join(finetune_results_dir, "finetune_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Table saved to: {csv_path}")

    # Save as Markdown
    md_path = os.path.join(finetune_results_dir, "finetune_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print(f"Table saved to: {md_path}")

    # Print table
    print(f"\n{df.to_string(index=False)}")
    print("\nFine-tuning experiment complete.")


if __name__ == "__main__":
    main()
