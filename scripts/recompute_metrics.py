#!/usr/bin/env python3
"""Recompute metrics for every existing predictions file.

Traverses ``results/`` looking for ``predictions_*_test.json`` files produced
by the core pipeline, ablation, baseline, and finetune stages, and rewrites
the matching ``metrics_*_test.json`` using the per-sentence aggregated
entity metrics. Useful after fixing the metric aggregation bug — no need to
re-run inference.

Also refreshes ``results/ablation_table.*``, ``results/baseline_table.*``,
and ``results/finetune/finetune_table.*`` from the new numbers.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml  # noqa: E402

from src.data_utils import (  # noqa: E402
    align_sentences_by_tokens,
    load_sentences_json,
)
from src.metrics import (  # noqa: E402
    classify_errors,
    compute_entity_metrics_aggregated,
    compute_per_type_metrics_aggregated,
)
from src.noise_analysis import (  # noqa: E402
    aggregate_noise_analysis,
    classify_noise_attribution,
)


def load_config() -> dict:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _eval_block(
    gold_sentences: list[dict],
    pred_sentences: list[dict],
    entity_types: list[str],
) -> dict:
    """Given gold + per-sentence predictions, return a metrics dict."""
    pred_by_id = {s["id"]: s.get("predictions", s.get("entities", [])) for s in pred_sentences}
    per_sentence_pairs = []
    errors = {
        "type_error": 0,
        "boundary_error": 0,
        "type_boundary_error": 0,
        "missing": 0,
        "spurious": 0,
    }
    for sent in gold_sentences:
        gold = sent["entities"]
        pred = pred_by_id.get(sent["id"], [])
        per_sentence_pairs.append((gold, pred))
        for k, v in classify_errors(gold, pred).items():
            errors[k] += v
    overall = compute_entity_metrics_aggregated(per_sentence_pairs)
    per_type = compute_per_type_metrics_aggregated(per_sentence_pairs, entity_types)
    return {"overall": overall, "per_type": per_type, "errors": errors}


def recompute_core(config: dict, processed: str, results: str) -> None:
    entity_types = [e["conll_label"] for e in config["labels"]]
    split = "test"

    for ds in ("conll03", "cleanconll"):
        gold_path = os.path.join(processed, f"{ds}_{split}.json")
        pred_path = os.path.join(results, f"predictions_{ds}_{split}.json")
        if not os.path.isfile(pred_path):
            print(f"  skip core {ds}: no predictions at {pred_path}")
            continue
        gold = load_sentences_json(gold_path)
        pred = load_sentences_json(pred_path)
        result = _eval_block(gold, pred, entity_types)
        out = os.path.join(results, f"metrics_{ds}_{split}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        o = result["overall"]
        print(
            f"  core {ds:10s}  TP={o['tp']:5d} FP={o['fp']:5d} FN={o['fn']:5d}  "
            f"P={o['precision']:.4f} R={o['recall']:.4f} F1={o['f1']:.4f}"
        )


def recompute_noise_analysis(
    config: dict, processed: str, results: str, pred_dir: str, out_dir: str
) -> dict:
    """Recompute noise_analysis_test.json from a given predictions directory."""
    split = "test"
    gold_conll = load_sentences_json(os.path.join(processed, f"conll03_{split}.json"))
    gold_clean = load_sentences_json(os.path.join(processed, f"cleanconll_{split}.json"))

    conll_pred_path = os.path.join(pred_dir, f"predictions_conll03_{split}.json")
    if not os.path.isfile(conll_pred_path):
        return {}
    pred_conll = load_sentences_json(conll_pred_path)
    pred_conll_by_id = {s["id"]: s["predictions"] for s in pred_conll}

    idx_conll, idx_clean = align_sentences_by_tokens(gold_conll, gold_clean)
    per_sentence_results = []
    for ci, kk in zip(idx_conll, idx_clean):
        conll_sent = gold_conll[ci]
        clean_sent = gold_clean[kk]
        pred_ents = pred_conll_by_id.get(conll_sent["id"], [])
        per_sentence_results.append(
            classify_noise_attribution(
                pred_ents, conll_sent["entities"], clean_sent["entities"]
            )
        )
    agg = aggregate_noise_analysis(per_sentence_results, max_examples=10)
    out_path = os.path.join(out_dir, f"noise_analysis_{split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    return agg


def recompute_ablation(config: dict, processed: str, results: str) -> None:
    ablation_root = os.path.join(results, "ablation")
    if not os.path.isdir(ablation_root):
        print("  skip ablation: directory not found")
        return
    entity_types = [e["conll_label"] for e in config["labels"]]
    split = "test"
    summary_rows = []
    for short in sorted(os.listdir(ablation_root)):
        sub = os.path.join(ablation_root, short)
        if not os.path.isdir(sub):
            continue
        dataset_f1 = {}
        for ds in ("conll03", "cleanconll"):
            gold = load_sentences_json(os.path.join(processed, f"{ds}_{split}.json"))
            pred_path = os.path.join(sub, f"predictions_{ds}_{split}.json")
            if not os.path.isfile(pred_path):
                continue
            pred = load_sentences_json(pred_path)
            result = _eval_block(gold, pred, entity_types)
            with open(
                os.path.join(sub, f"metrics_{ds}_{split}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            dataset_f1[ds] = result["overall"]["f1"]
            o = result["overall"]
            print(
                f"  ablation/{short}/{ds:10s}  TP={o['tp']:5d} FP={o['fp']:5d} "
                f"FN={o['fn']:5d}  F1={o['f1']:.4f}"
            )
        noise_agg = recompute_noise_analysis(config, processed, results, sub, sub)
        summary_rows.append(
            {
                "model": short,
                "conll03_f1": dataset_f1.get("conll03"),
                "cleanconll_f1": dataset_f1.get("cleanconll"),
                "f1_delta": round(
                    (dataset_f1.get("cleanconll") or 0) - (dataset_f1.get("conll03") or 0),
                    4,
                ),
                "noise_penalized": noise_agg.get("noise_penalized_correct", 0),
            }
        )
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(os.path.join(results, "ablation_table.csv"), index=False)
        df.to_markdown(os.path.join(results, "ablation_table.md"), index=False)


def recompute_baseline(config: dict, processed: str, results: str) -> None:
    baseline_root = os.path.join(results, "baseline")
    if not os.path.isdir(baseline_root):
        print("  skip baseline: directory not found")
        return
    entity_types = [e["conll_label"] for e in config["labels"]]
    split = "test"
    summary_rows = []
    for short in sorted(os.listdir(baseline_root)):
        sub = os.path.join(baseline_root, short)
        if not os.path.isdir(sub):
            continue
        dataset_f1 = {}
        for ds in ("conll03", "cleanconll"):
            gold = load_sentences_json(os.path.join(processed, f"{ds}_{split}.json"))
            pred_path = os.path.join(sub, f"predictions_{ds}_{split}.json")
            if not os.path.isfile(pred_path):
                continue
            pred = load_sentences_json(pred_path)
            result = _eval_block(gold, pred, entity_types)
            with open(
                os.path.join(sub, f"metrics_{ds}_{split}.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            dataset_f1[ds] = result["overall"]["f1"]
            o = result["overall"]
            print(
                f"  baseline/{short}/{ds:10s}  TP={o['tp']:5d} FP={o['fp']:5d} "
                f"FN={o['fn']:5d}  F1={o['f1']:.4f}"
            )
        noise_agg = recompute_noise_analysis(config, processed, results, sub, sub)
        summary_rows.append(
            {
                "baseline": short,
                "conll03_f1": dataset_f1.get("conll03"),
                "cleanconll_f1": dataset_f1.get("cleanconll"),
                "f1_delta": round(
                    (dataset_f1.get("cleanconll") or 0) - (dataset_f1.get("conll03") or 0),
                    4,
                ),
                "noise_penalized": noise_agg.get("noise_penalized_correct", 0),
            }
        )
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(os.path.join(results, "baseline_table.csv"), index=False)
        df.to_markdown(os.path.join(results, "baseline_table.md"), index=False)


def recompute_finetune(config: dict, processed: str, results: str) -> None:
    ft_root = os.path.join(results, "finetune")
    if not os.path.isdir(ft_root):
        print("  skip finetune: directory not found")
        return
    entity_types = [e["conll_label"] for e in config["labels"]]
    split = "test"
    eval_sentences = load_sentences_json(
        os.path.join(processed, f"cleanconll_{split}.json")
    )
    summary_rows = []
    for fname in sorted(os.listdir(ft_root)):
        if not fname.startswith("predictions_finetuned_") or not fname.endswith(".json"):
            continue
        train_name = fname[len("predictions_finetuned_"):-len(".json")]
        pred_path = os.path.join(ft_root, fname)
        pred = load_sentences_json(pred_path)
        result = _eval_block(eval_sentences, pred, entity_types)
        metrics_path = os.path.join(ft_root, f"metrics_finetuned_{train_name}.json")
        result_out = {
            "model": f"finetuned_{train_name}",
            "eval_split": f"cleanconll_{split}",
            **result,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result_out, f, indent=2, ensure_ascii=False)
        o = result["overall"]
        print(
            f"  finetune/finetuned_{train_name:10s}  TP={o['tp']:5d} FP={o['fp']:5d} "
            f"FN={o['fn']:5d}  F1={o['f1']:.4f}"
        )

        row = {
            "model": f"finetuned_{train_name}",
            "train_data": train_name,
            "precision": result["overall"]["precision"],
            "recall": result["overall"]["recall"],
            "f1": result["overall"]["f1"],
            "tp": result["overall"]["tp"],
            "fp": result["overall"]["fp"],
            "fn": result["overall"]["fn"],
        }
        for et in entity_types:
            row[f"{et}_f1"] = result["per_type"][et]["f1"]
        for err_type, count in result["errors"].items():
            row[err_type] = count
        summary_rows.append(row)

    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(os.path.join(ft_root, "finetune_table.csv"), index=False)
        df.to_markdown(os.path.join(ft_root, "finetune_table.md"), index=False)


def main() -> None:
    config = load_config()
    processed = config["paths"]["processed_data"]
    results = config["paths"]["results"]

    print("[core]")
    recompute_core(config, processed, results)
    # Also refresh core noise_analysis_test.json
    recompute_noise_analysis(config, processed, results, results, results)
    print()
    print("[ablation]")
    recompute_ablation(config, processed, results)
    print()
    print("[baseline]")
    recompute_baseline(config, processed, results)
    print()
    print("[finetune]")
    recompute_finetune(config, processed, results)
    print()
    print(
        "All metrics recomputed. Run `python scripts/evaluate.py --config configs/default.yaml "
        "--bootstrap` if you also need bootstrap CIs refreshed, and "
        "`python scripts/generate_report.py --config configs/default.yaml` for the summary report."
    )


if __name__ == "__main__":
    main()
