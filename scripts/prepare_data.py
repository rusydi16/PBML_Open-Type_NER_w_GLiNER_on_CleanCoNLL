#!/usr/bin/env python3
"""Prepare CoNLL-03 and CleanCoNLL datasets for evaluation.

Parses raw CoNLL files and saves them as JSON in data/processed/.
"""

import argparse
import os
import sys

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml  # noqa: E402

from src.data_utils import (  # noqa: E402
    parse_cleanconll_file,
    parse_conll03_file,
    save_sentences_json,
    setup_cleanconll,
)


# Split mapping: split_name -> raw filename
CONLL03_SPLITS = {
    "train": "eng.train",
    "dev": "eng.testa",
    "test": "eng.testb",
}

CLEANCONLL_SPLITS = {
    "train": "cleanconll.train",
    "dev": "cleanconll.dev",
    "test": "cleanconll.test",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare CoNLL-03 and CleanCoNLL datasets."
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to config YAML (default: configs/default.yaml)",
    )
    parser.add_argument(
        "--skip-cleanconll-build",
        action="store_true",
        help="Skip cloning / setting up the CleanCoNLL repository",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_data_path = config["paths"]["raw_data"]
    processed_data_path = config["paths"]["processed_data"]
    cleanconll_repo_path = config["paths"]["cleanconll_repo"]

    os.makedirs(processed_data_path, exist_ok=True)

    # --- CoNLL-03 ---
    print("=" * 60)
    print("Processing CoNLL-03 dataset")
    print("=" * 60)

    for split_name, filename in CONLL03_SPLITS.items():
        filepath = os.path.join(raw_data_path, filename)
        if not os.path.isfile(filepath):
            print(f"  WARNING: {filepath} not found, skipping {split_name} split.")
            continue

        print(f"  Parsing {filepath} ...")
        sentences = parse_conll03_file(filepath, split_name)
        out_path = os.path.join(processed_data_path, f"conll03_{split_name}.json")
        save_sentences_json(sentences, out_path)
        print(f"  Saved {len(sentences)} sentences to {out_path}")

    # --- CleanCoNLL ---
    print()
    print("=" * 60)
    print("Processing CleanCoNLL dataset")
    print("=" * 60)

    if not args.skip_cleanconll_build:
        setup_cleanconll(cleanconll_repo_path, raw_data_path)

    cleanconll_data_dir = os.path.join(cleanconll_repo_path, "data", "cleanconll")

    for split_name, filename in CLEANCONLL_SPLITS.items():
        filepath = os.path.join(cleanconll_data_dir, filename)
        if not os.path.isfile(filepath):
            print(f"  WARNING: {filepath} not found, skipping {split_name} split.")
            continue

        print(f"  Parsing {filepath} ...")
        sentences = parse_cleanconll_file(filepath, split_name)
        out_path = os.path.join(processed_data_path, f"cleanconll_{split_name}.json")
        save_sentences_json(sentences, out_path)
        print(f"  Saved {len(sentences)} sentences to {out_path}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
