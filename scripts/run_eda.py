#!/usr/bin/env python3
"""Generate an EDA report comparing CoNLL-03 and CleanCoNLL.

Outputs ``docs/eda_summary.md`` containing:

  * Per-split basic statistics (sentences, tokens, entities, averages)
  * Entity-type distribution side-by-side
  * Entity-length histograms
  * Delta analysis on aligned test sentences (what changed and by how much)
  * Example sentences for each change category

No GPU required; pure Python + pandas (already in requirements.txt).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from src.data_utils import (  # noqa: E402
    align_sentences_by_tokens,
    load_sentences_json,
)
from src.eda import (  # noqa: E402
    aggregate_deltas,
    basic_stats,
    entity_length_histogram,
    entity_type_counts,
)


SPLITS = ("train", "dev", "test")
DATASETS = ("conll03", "cleanconll")
DATASET_DISPLAY = {"conll03": "CoNLL-03", "cleanconll": "CleanCoNLL"}


def _md_table(rows: list[dict], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows
    )
    return "\n".join([head, sep, body])


def build_basic_stats_table(data: dict) -> str:
    rows = []
    for ds in DATASETS:
        for split in SPLITS:
            if split not in data[ds]:
                continue
            stats = basic_stats(data[ds][split])
            rows.append({"dataset": DATASET_DISPLAY[ds], "split": split, **stats})
    return _md_table(
        rows,
        [
            "dataset",
            "split",
            "sentences",
            "tokens",
            "entities",
            "avg_sent_len",
            "avg_ent_per_sent",
            "avg_ent_len",
        ],
    )


def build_entity_distribution_table(data: dict, entity_types: list[str]) -> str:
    rows = []
    for split in SPLITS:
        for ds in DATASETS:
            if split not in data[ds]:
                continue
            counts = entity_type_counts(data[ds][split])
            total = sum(counts.values())
            row = {"split": split, "dataset": DATASET_DISPLAY[ds], "total": total}
            for et in entity_types:
                row[et] = counts.get(et, 0)
            rows.append(row)
    return _md_table(rows, ["split", "dataset", "total", *entity_types])


def build_entity_length_table(data: dict) -> str:
    hists = {}
    keys: set[str] = set()
    for ds in DATASETS:
        if "test" not in data[ds]:
            continue
        h = entity_length_histogram(data[ds]["test"])
        hists[ds] = h
        keys.update(h.keys())

    def sort_key(k: str) -> tuple[int, int]:
        if k.startswith(">="):
            return (1, int(k[2:]))
        return (0, int(k))

    sorted_keys = sorted(keys, key=sort_key)
    rows = []
    for k in sorted_keys:
        row = {"entity_length (tokens)": k}
        for ds in DATASETS:
            row[DATASET_DISPLAY[ds]] = hists.get(ds, {}).get(k, 0)
        rows.append(row)
    return _md_table(
        rows, ["entity_length (tokens)", *[DATASET_DISPLAY[d] for d in DATASETS]]
    )


def build_delta_section(data: dict, entity_types: list[str]) -> str:
    out_lines: list[str] = []
    for split in SPLITS:
        if split not in data["conll03"] or split not in data["cleanconll"]:
            continue
        conll = data["conll03"][split]
        clean = data["cleanconll"][split]
        idx_a, idx_b = align_sentences_by_tokens(conll, clean)
        pairs = [(conll[a], clean[b]) for a, b in zip(idx_a, idx_b)]
        agg = aggregate_deltas(pairs, examples_per_category=3)
        totals = agg["totals"]

        out_lines.append(f"### Split: `{split}`")
        out_lines.append("")
        out_lines.append(
            f"- Aligned sentence pairs: **{len(pairs)}** "
            f"(of {len(conll)} CoNLL / {len(clean)} CleanCoNLL)"
        )
        out_lines.append("")
        out_lines.append("**Delta counts (across aligned entities):**")
        out_lines.append("")
        out_lines.append(
            _md_table(
                [
                    {"category": "exact_match", "count": totals.get("exact_match", 0),
                     "meaning": "identical (start, end, label) in both"},
                    {"category": "type_changed", "count": totals.get("type_changed", 0),
                     "meaning": "same span, label corrected in CleanCoNLL"},
                    {"category": "boundary_changed",
                     "count": totals.get("boundary_changed", 0),
                     "meaning": "same label, span adjusted"},
                    {"category": "removed", "count": totals.get("removed", 0),
                     "meaning": "present only in CoNLL-03"},
                    {"category": "added", "count": totals.get("added", 0),
                     "meaning": "present only in CleanCoNLL"},
                ],
                ["category", "count", "meaning"],
            )
        )
        out_lines.append("")

        # Per-type table
        by_type_rows = []
        for et in entity_types:
            bt = agg["by_type"].get(et, {})
            by_type_rows.append(
                {
                    "type": et,
                    "exact_match": bt.get("exact_match", 0),
                    "changed/removed (CoNLL side)": bt.get("changed_or_removed_a", 0),
                    "changed/added (Clean side)": bt.get("changed_or_added_b", 0),
                }
            )
        out_lines.append("**Per entity type:**")
        out_lines.append("")
        out_lines.append(
            _md_table(
                by_type_rows,
                [
                    "type",
                    "exact_match",
                    "changed/removed (CoNLL side)",
                    "changed/added (Clean side)",
                ],
            )
        )
        out_lines.append("")

        if split == "test":
            out_lines.append("**Example changes (test split):**")
            out_lines.append("")
            for cat, items in agg["examples"].items():
                if not items:
                    continue
                out_lines.append(f"*{cat}*")
                out_lines.append("")
                for ex in items:
                    if cat == "type_changed":
                        out_lines.append(
                            f"- `{ex['span_a']}` : **{ex['label_a']}** "
                            f"→ **{ex['label_b']}**  \n  "
                            f"  _\"...{ex['sentence'][:120]}...\"_"
                        )
                    elif cat == "boundary_changed":
                        out_lines.append(
                            f"- Label **{ex['label_a']}**: `{ex['span_a']}` "
                            f"→ `{ex['span_b']}`  \n  "
                            f"  _\"...{ex['sentence'][:120]}...\"_"
                        )
                    elif cat == "removed":
                        out_lines.append(
                            f"- `{ex['span']}` ({ex['label_a']}) removed in "
                            f"CleanCoNLL  \n  "
                            f"  _\"...{ex['sentence'][:120]}...\"_"
                        )
                    elif cat == "added":
                        out_lines.append(
                            f"- `{ex['span']}` ({ex['label_b']}) added in "
                            f"CleanCoNLL  \n  "
                            f"  _\"...{ex['sentence'][:120]}...\"_"
                        )
                out_lines.append("")
    return "\n".join(out_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--output",
        default="docs/eda_summary.md",
        help="Output markdown path (default: docs/eda_summary.md)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    entity_types = [entry["conll_label"] for entry in config["labels"]]
    processed_dir = config["paths"]["processed_data"]

    data: dict[str, dict[str, list]] = {ds: {} for ds in DATASETS}
    for ds in DATASETS:
        for split in SPLITS:
            path = os.path.join(processed_dir, f"{ds}_{split}.json")
            if os.path.isfile(path):
                data[ds][split] = load_sentences_json(path)

    missing = [
        (ds, s) for ds in DATASETS for s in SPLITS if s not in data[ds]
    ]
    if missing:
        print(f"  NOTE: skipping missing splits: {missing}")

    sections: list[str] = []
    sections.append("# EDA: CoNLL-03 vs CleanCoNLL\n")
    sections.append(
        "Auto-generated by `scripts/run_eda.py`. Run again after rebuilding "
        "`data/processed/` to refresh.\n"
    )

    sections.append("## 1. Basic Statistics\n")
    sections.append(build_basic_stats_table(data) + "\n")

    sections.append("## 2. Entity-Type Distribution\n")
    sections.append(
        f"Entity types configured: `{', '.join(entity_types)}`. "
        "Totals include all types even if not in this list.\n"
    )
    sections.append(build_entity_distribution_table(data, entity_types) + "\n")

    sections.append("## 3. Entity-Length Distribution (test split)\n")
    sections.append(
        "Counts of entities by span length (in tokens). Lengths >= 10 are "
        "bucketed as `>=10`.\n"
    )
    sections.append(build_entity_length_table(data) + "\n")

    sections.append(
        "## 4. Delta Analysis (CleanCoNLL vs CoNLL-03, token-aligned sentences)\n"
    )
    sections.append(
        "Sentences are aligned by exact token-sequence match. Sentences with "
        "patched tokens (e.g. `SKIING-WORLD` → `SKIING - WORLD`) are dropped "
        "from this view because token-index-based entity spans would not be "
        "directly comparable. Each entity is counted **once**, using this "
        "priority when finding a correspondence in the other dataset:\n\n"
        "1. exact match on `(start, end, label)`\n"
        "2. same `(start, end)` but different label → **type_changed**\n"
        "3. same label, overlapping span → **boundary_changed**\n"
        "4. no correspondence on one side → **removed** / **added**\n"
    )
    sections.append(build_delta_section(data, entity_types))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(sections))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
