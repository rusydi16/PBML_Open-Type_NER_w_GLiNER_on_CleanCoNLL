"""Utilities for loading, parsing, and saving NER data."""

import json
import os
import subprocess
from typing import Any


def bio_tags_to_entities(tokens: list[str], tags: list[str]) -> list[dict[str, Any]]:
    """Convert BIO-tagged token sequences to entity spans.

    Returns a list of dicts with keys: text, start, end, label.
    ``start`` is inclusive and ``end`` is exclusive (token indices).
    """
    entities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            # Close any open entity
            if current is not None:
                current["text"] = " ".join(tokens[current["start"]:current["end"]])
                entities.append(current)
            label = tag[2:]
            current = {"start": i, "end": i + 1, "label": label}
        elif tag.startswith("I-"):
            label = tag[2:]
            if current is not None and current["label"] == label:
                # Continue the current entity
                current["end"] = i + 1
            else:
                # I-X follows a different type or no entity — start new
                if current is not None:
                    current["text"] = " ".join(tokens[current["start"]:current["end"]])
                    entities.append(current)
                current = {"start": i, "end": i + 1, "label": label}
        else:
            # O tag — close any open entity
            if current is not None:
                current["text"] = " ".join(tokens[current["start"]:current["end"]])
                entities.append(current)
                current = None

    # Close trailing entity
    if current is not None:
        current["text"] = " ".join(tokens[current["start"]:current["end"]])
        entities.append(current)

    return entities


def parse_conll03_file(filepath: str, split_name: str) -> list[dict[str, Any]]:
    """Parse a CoNLL-03 format file (space-separated: word POS chunk NER).

    Skips -DOCSTART- lines. Blank lines separate sentences.
    Returns a list of sentence dicts with keys: id, tokens, ner_tags, entities.
    """
    sentences: list[dict[str, Any]] = []
    tokens: list[str] = []
    ner_tags: list[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                # Blank line — end of sentence
                if tokens:
                    idx = len(sentences) + 1
                    sentences.append({
                        "id": f"{split_name}-{idx:04d}",
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                        "entities": bio_tags_to_entities(tokens, ner_tags),
                    })
                    tokens = []
                    ner_tags = []
                continue

            parts = line.split()
            if parts[0] == "-DOCSTART-":
                continue

            word = parts[0]
            ner = parts[-1]  # Last column is NER tag
            tokens.append(word)
            ner_tags.append(ner)

    # Handle file not ending with blank line
    if tokens:
        idx = len(sentences) + 1
        sentences.append({
            "id": f"{split_name}-{idx:04d}",
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": bio_tags_to_entities(tokens, ner_tags),
        })

    return sentences


def parse_cleanconll_file(filepath: str, split_name: str) -> list[dict[str, Any]]:
    """Parse a CleanCoNLL format file (tab-separated: word POS wiki NER_intermediate NER_final).

    Uses column 5 (index 4) as the NER tag.
    Returns the same format as parse_conll03_file.
    """
    sentences: list[dict[str, Any]] = []
    tokens: list[str] = []
    ner_tags: list[str] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    idx = len(sentences) + 1
                    sentences.append({
                        "id": f"{split_name}-{idx:04d}",
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                        "entities": bio_tags_to_entities(tokens, ner_tags),
                    })
                    tokens = []
                    ner_tags = []
                continue

            parts = line.split("\t")
            if parts[0] == "-DOCSTART-":
                continue

            word = parts[0]
            ner = parts[4]  # Column index 4 = NER_final
            tokens.append(word)
            ner_tags.append(ner)

    # Handle file not ending with blank line
    if tokens:
        idx = len(sentences) + 1
        sentences.append({
            "id": f"{split_name}-{idx:04d}",
            "tokens": tokens,
            "ner_tags": ner_tags,
            "entities": bio_tags_to_entities(tokens, ner_tags),
        })

    return sentences


def save_sentences_json(sentences: list[dict[str, Any]], filepath: str) -> None:
    """Save a list of sentence dicts to a JSON file with indent=2."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)


def load_sentences_json(filepath: str) -> list[dict[str, Any]]:
    """Load a list of sentence dicts from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_cleanconll(cleanconll_repo_path: str, raw_data_path: str) -> None:
    """Clone the CleanCoNLL repository if not already present and print instructions."""
    if not os.path.isdir(cleanconll_repo_path):
        print(f"Cloning CleanCoNLL repository to {cleanconll_repo_path} ...")
        subprocess.run(
            ["git", "clone", "https://github.com/flairNLP/CleanCoNLL.git", cleanconll_repo_path],
            check=True,
        )
    else:
        print(f"CleanCoNLL repository already exists at {cleanconll_repo_path}")

    print(
        f"\nTo prepare CleanCoNLL data, place the original CoNLL-03 files in:\n"
        f"  {raw_data_path}\n"
        f"Then run the CleanCoNLL patching script as described in the CleanCoNLL README."
    )
