"""Utilities for loading, parsing, and saving NER data."""

import json
import os
import shutil
import subprocess
from typing import Any


# Mapping from our raw-data filenames to the names the CleanCoNLL build
# script expects in <repo>/data/conll03/.
_CLEANCONLL_RAW_RENAMES = {
    "eng.train": "train.txt",
    "eng.testa": "valid.txt",
    "eng.testb": "test.txt",
}
_CLEANCONLL_EXPECTED_OUTPUTS = ("cleanconll.train", "cleanconll.dev", "cleanconll.test")


def _find_bash() -> str | None:
    """Locate a bash executable on PATH, with Windows-specific fallbacks."""
    bash = shutil.which("bash")
    if bash:
        return bash
    # Common Git for Windows install locations.
    for candidate in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


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


def align_sentences_by_tokens(
    sentences_a: list[dict[str, Any]],
    sentences_b: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Match sentences across two datasets by token-content equality.

    Returns parallel ``(indices_a, indices_b)`` lists such that
    ``sentences_a[indices_a[k]]`` and ``sentences_b[indices_b[k]]`` hold the
    same token sequence. Sequential sentence IDs (e.g. ``test-0100``) cannot
    be trusted across CoNLL-03 and CleanCoNLL because cleaning drops or
    splits some sentences, which shifts the numbering. Token-content match
    gives a stable 1:1 correspondence for the ~97% of sentences that kept
    identical tokens; the rest (sentences whose tokens were patched) are
    dropped from the alignment.
    """
    b_by_tokens: dict[tuple, int] = {}
    for i, sent in enumerate(sentences_b):
        key = tuple(sent["tokens"])
        # If the same token tuple appears twice, keep the first occurrence.
        b_by_tokens.setdefault(key, i)

    indices_a: list[int] = []
    indices_b: list[int] = []
    for ai, sa in enumerate(sentences_a):
        bi = b_by_tokens.get(tuple(sa["tokens"]))
        if bi is not None:
            indices_a.append(ai)
            indices_b.append(bi)
    return indices_a, indices_b


def setup_cleanconll(cleanconll_repo_path: str, raw_data_path: str) -> None:
    """Clone the CleanCoNLL repo and run its build script if outputs are missing.

    The build script (``create_cleanconll_from_conll03.sh``) is pure shell —
    ``awk`` / ``patch`` / ``cut`` / ``paste`` / ``sed`` — so on Windows it runs
    under Git Bash. Inputs are the raw CoNLL-03 files at ``raw_data_path``; they
    are staged into ``<repo>/data/conll03/`` with the names the script wants.
    """
    # 1. Clone if missing.
    if not os.path.isdir(cleanconll_repo_path):
        print(f"Cloning CleanCoNLL repository to {cleanconll_repo_path} ...")
        subprocess.run(
            ["git", "clone", "https://github.com/flairNLP/CleanCoNLL.git", cleanconll_repo_path],
            check=True,
        )
    else:
        print(f"CleanCoNLL repository already exists at {cleanconll_repo_path}")

    # 2. Already built? Bail out.
    cleanconll_out_dir = os.path.join(cleanconll_repo_path, "data", "cleanconll")
    if all(
        os.path.isfile(os.path.join(cleanconll_out_dir, fn))
        for fn in _CLEANCONLL_EXPECTED_OUTPUTS
    ):
        print(f"CleanCoNLL data already built at {cleanconll_out_dir}, skipping build.")
        return

    # 3. Stage raw inputs as the build script expects.
    missing_raw = [
        src for src in _CLEANCONLL_RAW_RENAMES
        if not os.path.isfile(os.path.join(raw_data_path, src))
    ]
    if missing_raw:
        print(
            f"\nCannot auto-build CleanCoNLL: missing raw files "
            f"{missing_raw} in {raw_data_path}.\n"
            f"Run `python scripts/download_conll03.py` first, then retry."
        )
        return

    conll03_staging = os.path.join(cleanconll_repo_path, "data", "conll03")
    os.makedirs(conll03_staging, exist_ok=True)
    for src, dst in _CLEANCONLL_RAW_RENAMES.items():
        dst_path = os.path.join(conll03_staging, dst)
        if not os.path.isfile(dst_path):
            shutil.copy2(os.path.join(raw_data_path, src), dst_path)
            print(f"  Staged {src} -> {dst_path}")

    # 4. Locate bash and run the build script.
    bash = _find_bash()
    if bash is None:
        print(
            f"\nCannot auto-build CleanCoNLL: no 'bash' executable found.\n"
            f"Install Git for Windows (which provides Git Bash), or manually run:\n"
            f"  cd {cleanconll_repo_path}\n"
            f"  bash create_cleanconll_from_conll03.sh"
        )
        return

    print(f"\nRunning CleanCoNLL build script via: {bash}")
    result = subprocess.run(
        [bash, "create_cleanconll_from_conll03.sh"],
        cwd=cleanconll_repo_path,
    )
    if result.returncode != 0:
        print(
            f"\nBuild script exited with code {result.returncode}. "
            f"Check the output above and rerun manually if needed:\n"
            f"  cd {cleanconll_repo_path} && bash create_cleanconll_from_conll03.sh"
        )
        return

    # 5. Verify.
    built = [
        fn for fn in _CLEANCONLL_EXPECTED_OUTPUTS
        if os.path.isfile(os.path.join(cleanconll_out_dir, fn))
    ]
    if len(built) == len(_CLEANCONLL_EXPECTED_OUTPUTS):
        print(f"CleanCoNLL built successfully: {cleanconll_out_dir}")
    else:
        missing = set(_CLEANCONLL_EXPECTED_OUTPUTS) - set(built)
        print(
            f"WARNING: build script finished but expected files missing: {sorted(missing)}."
        )
