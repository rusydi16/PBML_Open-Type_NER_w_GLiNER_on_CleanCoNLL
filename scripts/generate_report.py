#!/usr/bin/env python3
"""CLI script that generates comparison tables and a findings report."""

import argparse
import json
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml


def _load_json(path: str) -> dict:
    """Load a JSON file and return the parsed dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(val: float) -> str:
    """Format a float as a percentage string with two decimals."""
    return f"{val * 100:.2f}"


def build_comparison_df(
    conll_metrics: dict,
    clean_metrics: dict,
    entity_types: list[str],
) -> pd.DataFrame:
    """Build a comparison DataFrame across the two datasets."""
    rows = []

    # Overall row
    co = conll_metrics["overall"]
    cl = clean_metrics["overall"]
    rows.append(
        {
            "Entity Type": "**Overall**",
            "CoNLL-03 P": _fmt(co["precision"]),
            "CoNLL-03 R": _fmt(co["recall"]),
            "CoNLL-03 F1": _fmt(co["f1"]),
            "CleanCoNLL P": _fmt(cl["precision"]),
            "CleanCoNLL R": _fmt(cl["recall"]),
            "CleanCoNLL F1": _fmt(cl["f1"]),
            "F1 Delta": _fmt(cl["f1"] - co["f1"]),
        }
    )

    # Per-type rows
    for etype in entity_types:
        co_t = conll_metrics["per_type"].get(etype, {})
        cl_t = clean_metrics["per_type"].get(etype, {})
        co_f1 = co_t.get("f1", 0.0)
        cl_f1 = cl_t.get("f1", 0.0)
        rows.append(
            {
                "Entity Type": etype,
                "CoNLL-03 P": _fmt(co_t.get("precision", 0.0)),
                "CoNLL-03 R": _fmt(co_t.get("recall", 0.0)),
                "CoNLL-03 F1": _fmt(co_f1),
                "CleanCoNLL P": _fmt(cl_t.get("precision", 0.0)),
                "CleanCoNLL R": _fmt(cl_t.get("recall", 0.0)),
                "CleanCoNLL F1": _fmt(cl_f1),
                "F1 Delta": _fmt(cl_f1 - co_f1),
            }
        )

    return pd.DataFrame(rows)


def generate_findings(
    conll_metrics: dict,
    clean_metrics: dict,
    noise_analysis: dict,
    entity_types: list[str],
    comparison_df: pd.DataFrame,
) -> str:
    """Generate the full findings.md content."""
    lines: list[str] = []

    # Title
    lines.append("# Findings: Fair Evaluation of GLiNER on CleanCoNLL vs CoNLL-03")
    lines.append("")

    # --- Section 1: Performance Comparison ---
    co_f1 = conll_metrics["overall"]["f1"]
    cl_f1 = clean_metrics["overall"]["f1"]
    delta = cl_f1 - co_f1
    direction = "higher" if delta > 0 else "lower" if delta < 0 else "identical"

    lines.append("## 1. Performance Comparison")
    lines.append("")
    lines.append(
        f"Overall F1 on CoNLL-03: **{_fmt(co_f1)}%** | "
        f"Overall F1 on CleanCoNLL: **{_fmt(cl_f1)}%** | "
        f"Delta: **{_fmt(delta)}** pp ({direction} on CleanCoNLL)."
    )
    lines.append("")

    lines.append("### Per-Entity-Type Breakdown")
    lines.append("")
    for etype in entity_types:
        co_t = conll_metrics["per_type"].get(etype, {})
        cl_t = clean_metrics["per_type"].get(etype, {})
        co_f1_t = co_t.get("f1", 0.0)
        cl_f1_t = cl_t.get("f1", 0.0)
        d = cl_f1_t - co_f1_t
        dir_t = "higher" if d > 0 else "lower" if d < 0 else "unchanged"
        lines.append(
            f"- **{etype}**: CoNLL-03 F1 = {_fmt(co_f1_t)}%, "
            f"CleanCoNLL F1 = {_fmt(cl_f1_t)}%, "
            f"Delta = {_fmt(d)} pp ({dir_t})"
        )
    lines.append("")

    # --- Section 2: Error Category Changes ---
    lines.append("## 2. Error Category Changes")
    lines.append("")

    error_categories = [
        "type_error",
        "boundary_error",
        "type_boundary_error",
        "missing",
        "spurious",
    ]
    lines.append("| Error Category | CoNLL-03 | CleanCoNLL | Delta |")
    lines.append("|---|---|---|---|")
    co_errors = conll_metrics.get("errors", {})
    cl_errors = clean_metrics.get("errors", {})
    for cat in error_categories:
        co_val = co_errors.get(cat, 0)
        cl_val = cl_errors.get(cat, 0)
        d = cl_val - co_val
        sign = "+" if d > 0 else ""
        lines.append(f"| {cat} | {co_val} | {cl_val} | {sign}{d} |")
    lines.append("")

    # --- Section 3: Noise Attribution ---
    lines.append("## 3. Noise Attribution")
    lines.append("")

    noise_keys = [
        ("correct_both", "Predictions correct under both annotation sets"),
        ("noise_penalized_correct", "Correct predictions penalised due to noisy CoNLL-03 labels"),
        ("model_learned_noise", "Model learned noisy patterns from training data"),
        ("genuine_error", "Genuine model errors (wrong under both annotations)"),
        ("missed_both", "Entities missed under both annotation sets"),
        ("missed_conll_only", "Entities missed only when evaluating against CoNLL-03"),
        ("missed_clean_only", "Entities missed only when evaluating against CleanCoNLL"),
    ]
    for key, explanation in noise_keys:
        count = noise_analysis.get(key, 0)
        lines.append(f"- **{key}**: {count} — {explanation}")
    lines.append("")

    # Example entities from noise_penalized_correct
    examples = noise_analysis.get("examples_noise_penalized", [])
    if examples:
        lines.append("**Example entities penalised by noisy labels** (up to 5):")
        lines.append("")
        for ex in examples[:5]:
            lines.append(f"- {ex}")
        lines.append("")

    # --- Section 4: Conclusion ---
    lines.append("## 4. Conclusion")
    lines.append("")

    abs_delta = abs(delta)
    if delta > 0:
        summary = (
            f"When evaluated against the cleaned annotations (CleanCoNLL), "
            f"GLiNER achieves an overall F1 that is {_fmt(abs_delta)} percentage points "
            f"higher than on the original CoNLL-03 test set. "
            f"This suggests that a portion of the apparent errors on CoNLL-03 "
            f"are attributable to noisy gold labels rather than genuine model mistakes. "
            f"Noise attribution analysis identified {noise_analysis.get('noise_penalized_correct', 0)} "
            f"predictions that were correct but penalised by the original annotations. "
            f"These findings support the value of using CleanCoNLL for a fairer evaluation "
            f"of NER models."
        )
    elif delta < 0:
        summary = (
            f"When evaluated against CleanCoNLL, GLiNER's overall F1 is {_fmt(abs_delta)} "
            f"percentage points lower than on the original CoNLL-03 test set. "
            f"This indicates that some predictions the model gets 'right' on CoNLL-03 "
            f"may actually be matching noisy labels. "
            f"Noise attribution analysis found {noise_analysis.get('model_learned_noise', 0)} "
            f"cases where the model appears to have learned noisy patterns."
        )
    else:
        summary = (
            "The overall F1 is identical on both annotation sets, suggesting "
            "annotation noise in CoNLL-03 has minimal net effect on this model's scores."
        )

    lines.append(summary)
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate comparison tables and findings report."
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
        help="Data split to report on (default: test)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    results_dir = config["paths"]["results"]
    entity_types = [entry["conll_label"] for entry in config["labels"]]

    # Load the three JSON files
    conll_path = os.path.join(results_dir, f"metrics_conll03_{args.split}.json")
    clean_path = os.path.join(results_dir, f"metrics_cleanconll_{args.split}.json")
    noise_path = os.path.join(results_dir, f"noise_analysis_{args.split}.json")

    print(f"Loading {conll_path}")
    conll_metrics = _load_json(conll_path)
    print(f"Loading {clean_path}")
    clean_metrics = _load_json(clean_path)
    print(f"Loading {noise_path}")
    noise_analysis = _load_json(noise_path)

    # Build comparison table
    df = build_comparison_df(conll_metrics, clean_metrics, entity_types)

    # Save CSV
    csv_path = os.path.join(results_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    # Save markdown table
    md_table_path = os.path.join(results_dir, "comparison_table.md")
    md_table = df.to_markdown(index=False)
    with open(md_table_path, "w", encoding="utf-8") as f:
        f.write(md_table)
        f.write("\n")
    print(f"Saved {md_table_path}")

    # Generate findings report
    findings = generate_findings(
        conll_metrics, clean_metrics, noise_analysis, entity_types, df
    )
    findings_path = os.path.join(results_dir, "findings.md")
    with open(findings_path, "w", encoding="utf-8") as f:
        f.write(findings)
    print(f"Saved {findings_path}")

    # Print comparison table to stdout
    print(f"\n{'='*60}")
    print("Comparison Table")
    print(f"{'='*60}")
    print(md_table)
    print(f"\nReport generation complete.")


if __name__ == "__main__":
    main()
