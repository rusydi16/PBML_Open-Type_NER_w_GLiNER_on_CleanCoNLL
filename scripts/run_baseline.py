#!/usr/bin/env python3
"""Run non-GLiNER baseline NER models on CoNLL-03 and CleanCoNLL.

Addresses the "baseline selain GLiNER" feedback from the proposal defense:
a supervised BERT model fine-tuned on CoNLL is the cleanest comparison point
because it shows the opposite bias from an open-type model — it should appear
stronger on the noisy CoNLL test set (which matches its training distribution)
and relatively weaker on CleanCoNLL, where the corrected labels penalize any
memorized noise.

Output mirrors the ablation layout:

    results/baseline/<short_name>/predictions_<dataset>_test.json
    results/baseline/<short_name>/metrics_<dataset>_test.json
    results/baseline/<short_name>/noise_analysis_test.json
    results/baseline_table.csv
    results/baseline_table.md
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.baseline import load_hf_ner_pipeline, predict_sentence_hf  # noqa: E402
from src.data_utils import align_sentences_by_tokens, load_sentences_json  # noqa: E402
from src.inference import set_seed  # noqa: E402
from src.metrics import (  # noqa: E402
    classify_errors,
    compute_entity_metrics_aggregated,
    compute_per_type_metrics_aggregated,
)
from src.noise_analysis import (  # noqa: E402
    aggregate_noise_analysis,
    classify_noise_attribution,
)


def _load_gold(processed_dir: str, split: str) -> dict[str, list]:
    gold = {}
    for ds in ("conll03", "cleanconll"):
        path = os.path.join(processed_dir, f"{ds}_{split}.json")
        print(f"Loading gold data: {path}")
        gold[ds] = load_sentences_json(path)
    return gold


def _run_inference(
    nlp,
    sentences: list[dict],
    allowed_labels: set[str],
    desc: str,
) -> list[dict]:
    out = []
    for sent in tqdm(sentences, desc=desc):
        preds = predict_sentence_hf(nlp, sent["tokens"], allowed_labels)
        out.append(
            {
                "id": sent["id"],
                "tokens": sent["tokens"],
                "predictions": preds,
            }
        )
    return out


def _evaluate(predictions, gold_sentences, entity_types):
    pred_by_id = {s["id"]: s["predictions"] for s in predictions}
    per_sentence_pairs = []
    errors = {
        "type_error": 0,
        "boundary_error": 0,
        "type_boundary_error": 0,
        "missing": 0,
        "spurious": 0,
    }
    for sent in gold_sentences:
        sid = sent["id"]
        gold_ents = sent["entities"]
        pred_ents = pred_by_id.get(sid, [])
        per_sentence_pairs.append((gold_ents, pred_ents))
        for k, v in classify_errors(gold_ents, pred_ents).items():
            errors[k] += v
    overall = compute_entity_metrics_aggregated(per_sentence_pairs)
    per_type = compute_per_type_metrics_aggregated(per_sentence_pairs, entity_types)
    return {
        "overall": overall,
        "per_type": per_type,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Baseline config YAML (default: configs/baseline.yaml)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun each baseline even when its outputs already exist",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]
    entity_types = config["entity_types"]
    allowed_labels = set(entity_types)
    split = args.split
    datasets = ("conll03", "cleanconll")

    gold = _load_gold(processed_dir, split)

    summary_rows = []
    for model_info in config["models"]:
        model_name = model_info["name"]
        short = model_info["short_name"]
        display = model_info.get("display", short)
        family = model_info.get("family", "hf")

        print(f"\n{'=' * 60}")
        print(f"Baseline: {display} ({short}, family={family})")
        print(f"{'=' * 60}")

        out_dir = os.path.join(results_dir, "baseline", short)
        os.makedirs(out_dir, exist_ok=True)

        # Resume: skip if all outputs exist.
        required = {
            d: os.path.join(out_dir, f"metrics_{d}_{split}.json") for d in datasets
        }
        noise_path = os.path.join(out_dir, f"noise_analysis_{split}.json")
        all_exist = (
            not args.force
            and all(os.path.isfile(p) for p in required.values())
            and os.path.isfile(noise_path)
        )
        if all_exist:
            print(f"  All outputs exist for {short}, loading cached metrics.")
            dataset_f1 = {}
            for d, p in required.items():
                with open(p, "r", encoding="utf-8") as f:
                    dataset_f1[d] = json.load(f)["overall"]["f1"]
            with open(noise_path, "r", encoding="utf-8") as f:
                noise_agg = json.load(f)
            summary_rows.append(
                {
                    "baseline": short,
                    "display": display,
                    "conll03_f1": dataset_f1.get("conll03", 0.0),
                    "cleanconll_f1": dataset_f1.get("cleanconll", 0.0),
                    "f1_delta": round(
                        dataset_f1.get("cleanconll", 0.0)
                        - dataset_f1.get("conll03", 0.0),
                        4,
                    ),
                    "noise_penalized": noise_agg.get("noise_penalized_correct", 0),
                }
            )
            continue

        if family != "hf":
            print(
                f"  Skipping {short}: family '{family}' is not supported by "
                f"this script. Only 'hf' (HuggingFace token-classification) "
                f"is implemented."
            )
            continue

        print(f"  Loading HF NER pipeline: {model_name}")
        nlp = load_hf_ner_pipeline(model_name)

        predictions = {}
        dataset_f1 = {}
        for d in datasets:
            print(f"\n  Running inference on {d} ({split})...")
            preds = _run_inference(
                nlp, gold[d], allowed_labels, desc=f"    {d}"
            )
            predictions[d] = preds
            pred_path = os.path.join(out_dir, f"predictions_{d}_{split}.json")
            with open(pred_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)
            print(f"  Saved predictions to: {pred_path}")

            metrics_result = _evaluate(preds, gold[d], entity_types)
            metrics_path = os.path.join(out_dir, f"metrics_{d}_{split}.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_result, f, indent=2, ensure_ascii=False)
            print(f"  Saved metrics to: {metrics_path}")
            overall = metrics_result["overall"]
            print(
                f"  {d} - P: {overall['precision']:.4f}  "
                f"R: {overall['recall']:.4f}  F1: {overall['f1']:.4f}"
            )
            dataset_f1[d] = overall["f1"]

        # Noise attribution, token-aligned.
        print(f"\n  Running noise attribution analysis...")
        pred_conll_by_id = {s["id"]: s["predictions"] for s in predictions["conll03"]}
        idx_conll, idx_clean = align_sentences_by_tokens(
            gold["conll03"], gold["cleanconll"]
        )
        per_sentence = []
        for ci, kk in zip(idx_conll, idx_clean):
            conll_sent = gold["conll03"][ci]
            clean_sent = gold["cleanconll"][kk]
            pred_ents = pred_conll_by_id.get(conll_sent["id"], [])
            per_sentence.append(
                classify_noise_attribution(
                    pred_ents, conll_sent["entities"], clean_sent["entities"]
                )
            )
        noise_agg = aggregate_noise_analysis(per_sentence, max_examples=10)
        with open(noise_path, "w", encoding="utf-8") as f:
            json.dump(noise_agg, f, indent=2, ensure_ascii=False)
        print(f"  Saved noise analysis to: {noise_path}")

        summary_rows.append(
            {
                "baseline": short,
                "display": display,
                "conll03_f1": dataset_f1["conll03"],
                "cleanconll_f1": dataset_f1["cleanconll"],
                "f1_delta": round(
                    dataset_f1["cleanconll"] - dataset_f1["conll03"], 4
                ),
                "noise_penalized": noise_agg.get("noise_penalized_correct", 0),
            }
        )

        # Free GPU memory before the next model.
        del nlp
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if not summary_rows:
        print("\nNo baselines ran; nothing to summarise.")
        return

    import pandas as pd

    df = pd.DataFrame(summary_rows)
    print(f"\n{'=' * 60}")
    print("Baseline Summary")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))

    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "baseline_table.csv")
    md_path = os.path.join(results_dir, "baseline_table.md")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV table to: {csv_path}")
    df.to_markdown(md_path, index=False)
    print(f"Saved Markdown table to: {md_path}")


if __name__ == "__main__":
    main()
