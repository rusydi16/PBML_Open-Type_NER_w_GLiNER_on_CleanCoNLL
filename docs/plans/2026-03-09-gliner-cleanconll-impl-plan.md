# GLiNER CleanCoNLL Evaluation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a reproducible pipeline that evaluates GLiNER NER on CoNLL-03 vs CleanCoNLL to measure the impact of label noise.

**Architecture:** Standalone Python scripts in `scripts/` with shared utilities in `src/`. A single `run_all.sh` chains the four pipeline steps. Config via YAML.

**Tech Stack:** Python 3.10+, gliner, torch, numpy, pandas, pyyaml

---

### Task 1: Scaffold repo structure and config

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `configs/default.yaml`
- Create: `src/__init__.py`
- Create: `data/raw/.gitkeep`
- Create: `data/processed/.gitkeep`
- Create: `results/.gitkeep`
- Create: `tests/__init__.py`

**Step 1: Create `.gitignore`**

```
# Data (user must provide)
data/raw/*.txt
data/raw/eng.*
data/cleanconll_repo/
data/processed/

# Results (generated)
results/*.json
results/*.csv
results/*.md

# Python
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
*.egg

# Environment
.venv/
venv/
.env

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db
```

**Step 2: Create `requirements.txt`**

```
gliner>=0.2.0
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
```

**Step 3: Create `configs/default.yaml`**

```yaml
model:
  name: "knowledgator/gliner-multitask-large-v0.5"
  threshold: 0.5

labels:
  - gliner_label: "person"
    conll_label: "PER"
  - gliner_label: "organization"
    conll_label: "ORG"
  - gliner_label: "location"
    conll_label: "LOC"
  - gliner_label: "miscellaneous"
    conll_label: "MISC"

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  results: "results"
  cleanconll_repo: "data/cleanconll_repo"

seed: 42
```

**Step 4: Create placeholder files**

Create `src/__init__.py`, `tests/__init__.py`, `data/raw/.gitkeep`, `data/processed/.gitkeep`, `results/.gitkeep` as empty files.

**Step 5: Init git repo and commit**

```bash
git init
git add -A
git commit -m "chore: scaffold repo structure, config, and dependencies"
```

---

### Task 2: Implement `src/data_utils.py` — CoNLL-03 parsing and CleanCoNLL patching

**Files:**
- Create: `src/data_utils.py`
- Create: `tests/test_data_utils.py`

**Step 1: Write tests for CoNLL-03 parsing**

```python
# tests/test_data_utils.py
"""Tests for data_utils module."""
import json
import tempfile
import os
from src.data_utils import parse_conll03_file, bio_tags_to_entities


def test_bio_tags_to_entities_simple():
    tokens = ["John", "lives", "in", "New", "York"]
    tags = ["B-PER", "O", "O", "B-LOC", "I-LOC"]
    entities = bio_tags_to_entities(tokens, tags)
    assert len(entities) == 2
    assert entities[0] == {"text": "John", "start": 0, "end": 1, "label": "PER"}
    assert entities[1] == {"text": "New York", "start": 3, "end": 5, "label": "LOC"}


def test_bio_tags_to_entities_empty():
    tokens = ["The", "cat", "sat"]
    tags = ["O", "O", "O"]
    entities = bio_tags_to_entities(tokens, tags)
    assert entities == []


def test_bio_tags_to_entities_consecutive():
    tokens = ["John", "Mary", "ran"]
    tags = ["B-PER", "B-PER", "O"]
    entities = bio_tags_to_entities(tokens, tags)
    assert len(entities) == 2
    assert entities[0] == {"text": "John", "start": 0, "end": 1, "label": "PER"}
    assert entities[1] == {"text": "Mary", "start": 1, "end": 2, "label": "PER"}


def test_parse_conll03_file():
    content = (
        "-DOCSTART- -X- -X- O\n"
        "\n"
        "EU NNP B-NP B-ORG\n"
        "rejects VBZ B-VP O\n"
        "German JJ B-NP B-MISC\n"
        "call NN I-NP O\n"
        "\n"
        "Peter NNP B-NP B-PER\n"
        "Blackburn NNP I-NP I-PER\n"
        "\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        f.flush()
        sentences = parse_conll03_file(f.name)
    os.unlink(f.name)

    assert len(sentences) == 2
    assert sentences[0]["tokens"] == ["EU", "rejects", "German", "call"]
    assert sentences[0]["ner_tags"] == ["B-ORG", "O", "B-MISC", "O"]
    assert len(sentences[0]["entities"]) == 2
    assert sentences[1]["tokens"] == ["Peter", "Blackburn"]
    assert sentences[1]["entities"][0]["label"] == "PER"
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_data_utils.py -v
```
Expected: FAIL (ImportError)

**Step 3: Implement `src/data_utils.py`**

```python
"""Utilities for parsing CoNLL-03 and CleanCoNLL data files."""
import json
import os
import subprocess
from typing import Any


def bio_tags_to_entities(tokens: list[str], tags: list[str]) -> list[dict[str, Any]]:
    """Convert BIO-tagged token sequences to a list of entity spans.

    Args:
        tokens: List of token strings.
        tags: List of BIO tag strings (e.g., "B-PER", "I-PER", "O").

    Returns:
        List of entity dicts with keys: text, start, end, label.
        start is inclusive token index, end is exclusive.
    """
    entities = []
    current_entity = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            label = tag[2:]
            current_entity = {
                "text": token,
                "start": i,
                "end": i + 1,
                "label": label,
            }
        elif tag.startswith("I-") and current_entity is not None:
            label = tag[2:]
            if label == current_entity["label"]:
                current_entity["text"] += " " + token
                current_entity["end"] = i + 1
            else:
                entities.append(current_entity)
                current_entity = {
                    "text": token,
                    "start": i,
                    "end": i + 1,
                    "label": label,
                }
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    return entities


def parse_conll03_file(filepath: str, split_name: str = "data") -> list[dict[str, Any]]:
    """Parse a CoNLL-03 format file into a list of sentence dicts.

    CoNLL-03 format: word POS chunk NER (space-separated columns).
    Sentences separated by blank lines. -DOCSTART- lines are skipped.

    Args:
        filepath: Path to the CoNLL-03 file.
        split_name: Name prefix for sentence IDs (e.g., "test", "dev").

    Returns:
        List of sentence dicts with keys: id, tokens, ner_tags, entities.
    """
    sentences = []
    current_tokens = []
    current_tags = []
    sent_idx = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_tokens:
                    entities = bio_tags_to_entities(current_tokens, current_tags)
                    sentences.append({
                        "id": f"{split_name}-{sent_idx:04d}",
                        "tokens": current_tokens,
                        "ner_tags": current_tags,
                        "entities": entities,
                    })
                    sent_idx += 1
                    current_tokens = []
                    current_tags = []
            elif line.startswith("-DOCSTART-"):
                continue
            else:
                parts = line.split()
                if len(parts) >= 4:
                    current_tokens.append(parts[0])
                    current_tags.append(parts[3])

    if current_tokens:
        entities = bio_tags_to_entities(current_tokens, current_tags)
        sentences.append({
            "id": f"{split_name}-{sent_idx:04d}",
            "tokens": current_tokens,
            "ner_tags": current_tags,
            "entities": entities,
        })

    return sentences


def parse_cleanconll_file(filepath: str, split_name: str = "data") -> list[dict[str, Any]]:
    """Parse a CleanCoNLL format file into a list of sentence dicts.

    CleanCoNLL format: word POS wiki_link NER_intermediate NER_final (tab-separated).
    Uses column 5 (index 4) as the final corrected NER tag.

    Args:
        filepath: Path to the CleanCoNLL file.
        split_name: Name prefix for sentence IDs.

    Returns:
        List of sentence dicts with keys: id, tokens, ner_tags, entities.
    """
    sentences = []
    current_tokens = []
    current_tags = []
    sent_idx = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current_tokens:
                    entities = bio_tags_to_entities(current_tokens, current_tags)
                    sentences.append({
                        "id": f"{split_name}-{sent_idx:04d}",
                        "tokens": current_tokens,
                        "ner_tags": current_tags,
                        "entities": entities,
                    })
                    sent_idx += 1
                    current_tokens = []
                    current_tags = []
            elif line.startswith("-DOCSTART-"):
                continue
            else:
                parts = line.split("\t")
                if len(parts) >= 5:
                    current_tokens.append(parts[0])
                    current_tags.append(parts[4])  # Column 5: final NER

    if current_tokens:
        entities = bio_tags_to_entities(current_tokens, current_tags)
        sentences.append({
            "id": f"{split_name}-{sent_idx:04d}",
            "tokens": current_tokens,
            "ner_tags": current_tags,
            "entities": entities,
        })

    return sentences


def save_sentences_json(sentences: list[dict], filepath: str) -> None:
    """Save parsed sentences to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)


def load_sentences_json(filepath: str) -> list[dict]:
    """Load parsed sentences from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_cleanconll(cleanconll_repo_path: str, raw_data_path: str) -> None:
    """Clone CleanCoNLL repo and run the build script.

    Args:
        cleanconll_repo_path: Where to clone the repo.
        raw_data_path: Path to directory containing eng.train, eng.testa, eng.testb.
    """
    if not os.path.exists(cleanconll_repo_path):
        subprocess.run(
            ["git", "clone", "https://github.com/flairNLP/CleanCoNLL.git", cleanconll_repo_path],
            check=True,
        )
    print(f"CleanCoNLL repo ready at {cleanconll_repo_path}")
    print("To build CleanCoNLL, follow the instructions in the repo README.")
    print(f"Make sure your CoNLL-03 files are in: {raw_data_path}")
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_data_utils.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/data_utils.py tests/test_data_utils.py
git commit -m "feat: add CoNLL-03/CleanCoNLL parsing utilities with tests"
```

---

### Task 3: Implement `src/inference.py` — GLiNER wrapper

**Files:**
- Create: `src/inference.py`
- Create: `tests/test_inference.py`

**Step 1: Write tests**

```python
# tests/test_inference.py
"""Tests for inference module (unit tests with mocked model)."""
from unittest.mock import MagicMock
from src.inference import (
    tokens_to_text_with_offsets,
    map_char_spans_to_token_spans,
    map_gliner_label_to_conll,
)


def test_tokens_to_text_with_offsets():
    tokens = ["EU", "rejects", "German", "call"]
    text, offsets = tokens_to_text_with_offsets(tokens)
    assert text == "EU rejects German call"
    assert offsets == [(0, 2), (3, 10), (11, 17), (18, 22)]


def test_map_char_spans_to_token_spans():
    offsets = [(0, 2), (3, 10), (11, 17), (18, 22)]
    # "EU" spans chars 0-2
    start_tok, end_tok = map_char_spans_to_token_spans(0, 2, offsets)
    assert start_tok == 0
    assert end_tok == 1
    # "German call" spans chars 11-22
    start_tok, end_tok = map_char_spans_to_token_spans(11, 22, offsets)
    assert start_tok == 2
    assert end_tok == 4


def test_map_gliner_label_to_conll():
    label_map = {
        "person": "PER",
        "organization": "ORG",
        "location": "LOC",
        "miscellaneous": "MISC",
    }
    assert map_gliner_label_to_conll("person", label_map) == "PER"
    assert map_gliner_label_to_conll("organization", label_map) == "ORG"
    assert map_gliner_label_to_conll("unknown", label_map) == "unknown"
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_inference.py -v
```

**Step 3: Implement `src/inference.py`**

```python
"""GLiNER inference wrapper for NER prediction."""
import random
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tokens_to_text_with_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Reconstruct raw text from tokens and compute character offsets.

    Joins tokens with single spaces.

    Args:
        tokens: List of token strings.

    Returns:
        Tuple of (reconstructed text, list of (char_start, char_end) per token).
    """
    offsets = []
    current_pos = 0
    parts = []
    for token in tokens:
        offsets.append((current_pos, current_pos + len(token)))
        parts.append(token)
        current_pos += len(token) + 1  # +1 for space
    return " ".join(parts), offsets


def map_char_spans_to_token_spans(
    char_start: int, char_end: int, token_offsets: list[tuple[int, int]]
) -> tuple[int, int]:
    """Map character-level spans back to token indices.

    Args:
        char_start: Start character index (inclusive).
        char_end: End character index (exclusive).
        token_offsets: List of (char_start, char_end) per token.

    Returns:
        Tuple of (start_token, end_token) where end_token is exclusive.
    """
    start_token = None
    end_token = None
    for i, (tok_start, tok_end) in enumerate(token_offsets):
        if tok_start <= char_start < tok_end:
            start_token = i
        if tok_start < char_end <= tok_end:
            end_token = i + 1
    if start_token is None:
        start_token = 0
    if end_token is None:
        end_token = len(token_offsets)
    return start_token, end_token


def map_gliner_label_to_conll(gliner_label: str, label_map: dict[str, str]) -> str:
    """Map a GLiNER label string to a CoNLL label.

    Args:
        gliner_label: The label returned by GLiNER (e.g., "person").
        label_map: Mapping from GLiNER labels to CoNLL labels.

    Returns:
        CoNLL label string, or the original label if not found.
    """
    return label_map.get(gliner_label, gliner_label)


def load_gliner_model(model_name: str):
    """Load a GLiNER model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Loaded GLiNER model.
    """
    from gliner import GLiNER
    model = GLiNER.from_pretrained(model_name)
    return model


def predict_sentence(
    model,
    tokens: list[str],
    gliner_labels: list[str],
    label_map: dict[str, str],
    threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Run GLiNER inference on a single sentence.

    Args:
        model: Loaded GLiNER model.
        tokens: List of token strings.
        gliner_labels: List of GLiNER label strings (e.g., ["person", "organization"]).
        label_map: Mapping from GLiNER labels to CoNLL labels.
        threshold: Score threshold for predictions.

    Returns:
        List of predicted entity dicts with: text, start_token, end_token, label, score.
    """
    text, token_offsets = tokens_to_text_with_offsets(tokens)

    with torch.no_grad():
        raw_entities = model.predict_entities(text, gliner_labels, threshold=threshold)

    predictions = []
    for ent in raw_entities:
        start_tok, end_tok = map_char_spans_to_token_spans(
            ent["start"], ent["end"], token_offsets
        )
        conll_label = map_gliner_label_to_conll(ent["label"], label_map)
        predictions.append({
            "text": ent["text"],
            "start_token": start_tok,
            "end_token": end_tok,
            "label": conll_label,
            "score": round(ent["score"], 4),
        })

    return predictions
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_inference.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/inference.py tests/test_inference.py
git commit -m "feat: add GLiNER inference wrapper with char-to-token mapping"
```

---

### Task 4: Implement `src/metrics.py` — entity-level evaluation

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

**Step 1: Write tests**

```python
# tests/test_metrics.py
"""Tests for entity-level metrics."""
from src.metrics import (
    compute_entity_metrics,
    classify_errors,
)


def test_compute_entity_metrics_perfect():
    gold = [{"start": 0, "end": 1, "label": "PER"}]
    pred = [{"start_token": 0, "end_token": 1, "label": "PER", "text": "X", "score": 0.9}]
    metrics = compute_entity_metrics(gold, pred)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_entity_metrics_no_predictions():
    gold = [{"start": 0, "end": 1, "label": "PER"}]
    pred = []
    metrics = compute_entity_metrics(gold, pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0


def test_compute_entity_metrics_wrong_label():
    gold = [{"start": 0, "end": 1, "label": "PER"}]
    pred = [{"start_token": 0, "end_token": 1, "label": "ORG", "text": "X", "score": 0.9}]
    metrics = compute_entity_metrics(gold, pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0


def test_classify_errors():
    gold = [
        {"start": 0, "end": 1, "label": "PER"},
        {"start": 5, "end": 7, "label": "LOC"},
        {"start": 10, "end": 12, "label": "ORG"},
    ]
    pred = [
        {"start_token": 0, "end_token": 1, "label": "ORG", "text": "X", "score": 0.9},  # type error
        {"start_token": 5, "end_token": 8, "label": "LOC", "text": "Y", "score": 0.9},  # boundary error
        {"start_token": 20, "end_token": 22, "label": "PER", "text": "Z", "score": 0.9},  # spurious
        # gold[2] is missing
    ]
    errors = classify_errors(gold, pred)
    assert errors["type_error"] == 1
    assert errors["boundary_error"] == 1
    assert errors["missing"] == 1
    assert errors["spurious"] == 1
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metrics.py -v
```

**Step 3: Implement `src/metrics.py`**

```python
"""Entity-level evaluation metrics and error classification."""
from typing import Any


def _normalize_entity(entity: dict) -> tuple[int, int, str]:
    """Normalize entity dict to (start, end, label) tuple."""
    start = entity.get("start", entity.get("start_token"))
    end = entity.get("end", entity.get("end_token"))
    return (start, end, entity["label"])


def compute_entity_metrics(
    gold_entities: list[dict], pred_entities: list[dict]
) -> dict[str, float]:
    """Compute entity-level exact-match precision, recall, F1.

    An entity matches only if (start, end, label) are all identical.

    Args:
        gold_entities: List of gold entity dicts (with start, end, label).
        pred_entities: List of predicted entity dicts (with start_token, end_token, label).

    Returns:
        Dict with precision, recall, f1, tp, fp, fn counts.
    """
    gold_set = {_normalize_entity(e) for e in gold_entities}
    pred_set = {_normalize_entity(e) for e in pred_entities}

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_per_type_metrics(
    gold_entities: list[dict], pred_entities: list[dict], entity_types: list[str]
) -> dict[str, dict[str, float]]:
    """Compute entity-level metrics per entity type.

    Args:
        gold_entities: List of gold entity dicts.
        pred_entities: List of predicted entity dicts.
        entity_types: List of entity type strings to evaluate.

    Returns:
        Dict mapping entity type to metrics dict.
    """
    results = {}
    for etype in entity_types:
        gold_filtered = [e for e in gold_entities if e["label"] == etype]
        pred_filtered = [e for e in pred_entities if e["label"] == etype]
        results[etype] = compute_entity_metrics(gold_filtered, pred_filtered)
    return results


def _spans_overlap(s1_start: int, s1_end: int, s2_start: int, s2_end: int) -> bool:
    """Check if two spans overlap."""
    return s1_start < s2_end and s2_start < s1_end


def classify_errors(
    gold_entities: list[dict], pred_entities: list[dict]
) -> dict[str, int]:
    """Classify prediction errors into categories.

    Categories:
    - type_error: span matches exactly but label differs
    - boundary_error: spans overlap but boundaries differ, same label
    - type_boundary_error: spans overlap, different boundaries AND label
    - missing: gold entity with no overlapping prediction (false negative)
    - spurious: prediction with no overlapping gold entity (false positive)

    Args:
        gold_entities: List of gold entity dicts.
        pred_entities: List of predicted entity dicts.

    Returns:
        Dict with error category counts.
    """
    gold_tuples = [_normalize_entity(e) for e in gold_entities]
    pred_tuples = [_normalize_entity(e) for e in pred_entities]

    gold_matched = set()
    pred_matched = set()
    exact_matches = set()

    errors = {
        "type_error": 0,
        "boundary_error": 0,
        "type_boundary_error": 0,
        "missing": 0,
        "spurious": 0,
    }

    # Find exact matches first
    gold_set = set(gold_tuples)
    pred_set = set(pred_tuples)
    exact = gold_set & pred_set

    for g_idx, g in enumerate(gold_tuples):
        if g in exact:
            for p_idx, p in enumerate(pred_tuples):
                if p == g and p_idx not in pred_matched:
                    gold_matched.add(g_idx)
                    pred_matched.add(p_idx)
                    break

    # Classify non-exact-match pairs
    for g_idx, (gs, ge, gl) in enumerate(gold_tuples):
        if g_idx in gold_matched:
            continue
        found_overlap = False
        for p_idx, (ps, pe, pl) in enumerate(pred_tuples):
            if p_idx in pred_matched:
                continue
            if _spans_overlap(gs, ge, ps, pe):
                found_overlap = True
                gold_matched.add(g_idx)
                pred_matched.add(p_idx)
                if gs == ps and ge == pe:
                    errors["type_error"] += 1
                elif gl == pl:
                    errors["boundary_error"] += 1
                else:
                    errors["type_boundary_error"] += 1
                break
        if not found_overlap:
            errors["missing"] += 1
            gold_matched.add(g_idx)

    for p_idx in range(len(pred_tuples)):
        if p_idx not in pred_matched:
            errors["spurious"] += 1

    return errors
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_metrics.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: add entity-level metrics and error classification"
```

---

### Task 5: Implement `src/noise_analysis.py` — noise attribution

**Files:**
- Create: `src/noise_analysis.py`
- Create: `tests/test_noise_analysis.py`

**Step 1: Write tests**

```python
# tests/test_noise_analysis.py
"""Tests for noise attribution analysis."""
from src.noise_analysis import classify_noise_attribution


def test_noise_attribution_pred_matches_clean_only():
    pred = [{"start_token": 0, "end_token": 1, "label": "PER", "text": "John", "score": 0.9}]
    conll_gold = [{"start": 0, "end": 1, "label": "ORG"}]
    clean_gold = [{"start": 0, "end": 1, "label": "PER"}]
    result = classify_noise_attribution(pred, conll_gold, clean_gold)
    assert result["noise_penalized_correct"] == 1
    assert result["model_learned_noise"] == 0
    assert result["genuine_error"] == 0


def test_noise_attribution_pred_matches_conll_only():
    pred = [{"start_token": 0, "end_token": 1, "label": "ORG", "text": "EU", "score": 0.9}]
    conll_gold = [{"start": 0, "end": 1, "label": "ORG"}]
    clean_gold = [{"start": 0, "end": 1, "label": "MISC"}]
    result = classify_noise_attribution(pred, conll_gold, clean_gold)
    assert result["noise_penalized_correct"] == 0
    assert result["model_learned_noise"] == 1


def test_noise_attribution_genuine_error():
    pred = [{"start_token": 0, "end_token": 1, "label": "LOC", "text": "X", "score": 0.9}]
    conll_gold = [{"start": 0, "end": 1, "label": "PER"}]
    clean_gold = [{"start": 0, "end": 1, "label": "PER"}]
    result = classify_noise_attribution(pred, conll_gold, clean_gold)
    assert result["genuine_error"] == 1
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_noise_analysis.py -v
```

**Step 3: Implement `src/noise_analysis.py`**

```python
"""Noise attribution analysis comparing predictions against CoNLL-03 and CleanCoNLL."""
from typing import Any


def _normalize(entity: dict) -> tuple[int, int, str]:
    """Normalize entity to (start, end, label)."""
    start = entity.get("start", entity.get("start_token"))
    end = entity.get("end", entity.get("end_token"))
    return (start, end, entity["label"])


def classify_noise_attribution(
    pred_entities: list[dict],
    conll_gold: list[dict],
    clean_gold: list[dict],
) -> dict[str, Any]:
    """Classify each prediction by noise attribution.

    For predictions that disagree with at least one gold standard:
    - noise_penalized_correct: pred matches CleanCoNLL but not CoNLL-03
      (label noise in CoNLL-03 unfairly penalizes a correct prediction)
    - model_learned_noise: pred matches CoNLL-03 but not CleanCoNLL
      (model reproduces the noisy annotation)
    - genuine_error: pred matches neither
    - correct_both: pred matches both (true positive on both)

    Args:
        pred_entities: Predicted entities.
        conll_gold: CoNLL-03 gold entities.
        clean_gold: CleanCoNLL gold entities.

    Returns:
        Dict with counts and example lists per category.
    """
    pred_set = {_normalize(e) for e in pred_entities}
    conll_set = {_normalize(e) for e in conll_gold}
    clean_set = {_normalize(e) for e in clean_gold}

    result = {
        "noise_penalized_correct": 0,
        "model_learned_noise": 0,
        "genuine_error": 0,
        "correct_both": 0,
        "examples_noise_penalized": [],
        "examples_learned_noise": [],
        "examples_genuine_error": [],
    }

    for entity in pred_set:
        in_conll = entity in conll_set
        in_clean = entity in clean_set

        example = {"start": entity[0], "end": entity[1], "label": entity[2]}

        if in_conll and in_clean:
            result["correct_both"] += 1
        elif in_clean and not in_conll:
            result["noise_penalized_correct"] += 1
            result["examples_noise_penalized"].append(example)
        elif in_conll and not in_clean:
            result["model_learned_noise"] += 1
            result["examples_learned_noise"].append(example)
        else:
            result["genuine_error"] += 1
            result["examples_genuine_error"].append(example)

    # Also account for gold entities missed by prediction
    # Missed in CoNLL but not CleanCoNLL (or vice versa) is also informative
    conll_missed = conll_set - pred_set
    clean_missed = clean_set - pred_set
    result["missed_conll_only"] = len(conll_missed - clean_missed)
    result["missed_clean_only"] = len(clean_missed - conll_missed)
    result["missed_both"] = len(conll_missed & clean_missed)

    return result


def aggregate_noise_analysis(
    per_sentence_results: list[dict], max_examples: int = 10
) -> dict[str, Any]:
    """Aggregate noise attribution results across all sentences.

    Args:
        per_sentence_results: List of per-sentence classification dicts.
        max_examples: Maximum number of examples to keep per category.

    Returns:
        Aggregated counts and truncated example lists.
    """
    agg = {
        "noise_penalized_correct": 0,
        "model_learned_noise": 0,
        "genuine_error": 0,
        "correct_both": 0,
        "missed_conll_only": 0,
        "missed_clean_only": 0,
        "missed_both": 0,
        "examples_noise_penalized": [],
        "examples_learned_noise": [],
        "examples_genuine_error": [],
    }

    for r in per_sentence_results:
        for key in ["noise_penalized_correct", "model_learned_noise", "genuine_error",
                     "correct_both", "missed_conll_only", "missed_clean_only", "missed_both"]:
            agg[key] += r[key]
        for key in ["examples_noise_penalized", "examples_learned_noise", "examples_genuine_error"]:
            agg[key].extend(r[key])

    # Truncate examples
    for key in ["examples_noise_penalized", "examples_learned_noise", "examples_genuine_error"]:
        agg[key] = agg[key][:max_examples]

    return agg
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_noise_analysis.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/noise_analysis.py tests/test_noise_analysis.py
git commit -m "feat: add noise attribution analysis module"
```

---

### Task 6: Implement `scripts/prepare_data.py` — data pipeline script

**Files:**
- Create: `scripts/prepare_data.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Prepare CoNLL-03 and CleanCoNLL datasets for evaluation.

Parses raw CoNLL-03 files and CleanCoNLL files into standardized JSON format.

Usage:
    python scripts/prepare_data.py --config configs/default.yaml
"""
import argparse
import os
import subprocess
import sys

import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import (
    parse_conll03_file,
    parse_cleanconll_file,
    save_sentences_json,
    setup_cleanconll,
)


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


def main():
    parser = argparse.ArgumentParser(description="Prepare CoNLL-03 and CleanCoNLL data")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--skip-cleanconll-build", action="store_true",
                        help="Skip cloning/building CleanCoNLL (assume already built)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    raw_dir = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_data"]
    cleanconll_repo = config["paths"]["cleanconll_repo"]

    os.makedirs(processed_dir, exist_ok=True)

    # Step 1: Parse CoNLL-03
    print("=" * 60)
    print("Step 1: Parsing CoNLL-03 files")
    print("=" * 60)
    for split_name, filename in CONLL03_SPLITS.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found, skipping {split_name}")
            continue
        sentences = parse_conll03_file(filepath, split_name)
        out_path = os.path.join(processed_dir, f"conll03_{split_name}.json")
        save_sentences_json(sentences, out_path)
        total_entities = sum(len(s["entities"]) for s in sentences)
        print(f"  {split_name}: {len(sentences)} sentences, {total_entities} entities -> {out_path}")

    # Step 2: Build CleanCoNLL
    print("\n" + "=" * 60)
    print("Step 2: Building CleanCoNLL")
    print("=" * 60)

    if not args.skip_cleanconll_build:
        setup_cleanconll(cleanconll_repo, raw_dir)

    # Step 3: Parse CleanCoNLL
    print("\n" + "=" * 60)
    print("Step 3: Parsing CleanCoNLL files")
    print("=" * 60)
    cleanconll_data_dir = os.path.join(cleanconll_repo, "data", "cleanconll")
    for split_name, filename in CLEANCONLL_SPLITS.items():
        filepath = os.path.join(cleanconll_data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  WARNING: {filepath} not found, skipping {split_name}")
            continue
        sentences = parse_cleanconll_file(filepath, split_name)
        out_path = os.path.join(processed_dir, f"cleanconll_{split_name}.json")
        save_sentences_json(sentences, out_path)
        total_entities = sum(len(s["entities"]) for s in sentences)
        print(f"  {split_name}: {len(sentences)} sentences, {total_entities} entities -> {out_path}")

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/prepare_data.py
git commit -m "feat: add data preparation script for CoNLL-03 and CleanCoNLL"
```

---

### Task 7: Implement `scripts/run_inference.py` — GLiNER inference script

**Files:**
- Create: `scripts/run_inference.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Run GLiNER inference on processed datasets.

Loads a GLiNER model and runs NER prediction on test splits of
CoNLL-03 and CleanCoNLL datasets.

Usage:
    python scripts/run_inference.py --config configs/default.yaml
"""
import argparse
import json
import os
import sys

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_sentences_json
from src.inference import set_seed, load_gliner_model, predict_sentence


def main():
    parser = argparse.ArgumentParser(description="Run GLiNER inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--datasets", nargs="+", default=["conll03", "cleanconll"],
                        help="Which datasets to run inference on")
    parser.add_argument("--split", type=str, default="test",
                        help="Which split to evaluate")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]
    os.makedirs(results_dir, exist_ok=True)

    # Build label mapping
    gliner_labels = [lbl["gliner_label"] for lbl in config["labels"]]
    label_map = {lbl["gliner_label"]: lbl["conll_label"] for lbl in config["labels"]}

    # Load model
    print(f"Loading model: {config['model']['name']}")
    model = load_gliner_model(config["model"]["name"])
    threshold = config["model"]["threshold"]
    print(f"Model loaded. Threshold: {threshold}")

    for dataset_name in args.datasets:
        input_path = os.path.join(processed_dir, f"{dataset_name}_{args.split}.json")
        if not os.path.exists(input_path):
            print(f"WARNING: {input_path} not found, skipping")
            continue

        print(f"\nRunning inference on {dataset_name} ({args.split})...")
        sentences = load_sentences_json(input_path)

        all_predictions = []
        for sent in tqdm(sentences, desc=f"  {dataset_name}"):
            preds = predict_sentence(
                model, sent["tokens"], gliner_labels, label_map, threshold
            )
            all_predictions.append({
                "id": sent["id"],
                "tokens": sent["tokens"],
                "predictions": preds,
            })

        out_path = os.path.join(results_dir, f"predictions_{dataset_name}_{args.split}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, indent=2, ensure_ascii=False)
        total_preds = sum(len(p["predictions"]) for p in all_predictions)
        print(f"  Saved {total_preds} predictions across {len(all_predictions)} sentences -> {out_path}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/run_inference.py
git commit -m "feat: add GLiNER inference script"
```

---

### Task 8: Implement `scripts/evaluate.py` — evaluation script

**Files:**
- Create: `scripts/evaluate.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Evaluate GLiNER predictions against gold standard annotations.

Computes entity-level metrics, error classification, and noise attribution.

Usage:
    python scripts/evaluate.py --config configs/default.yaml
"""
import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_utils import load_sentences_json
from src.metrics import compute_entity_metrics, compute_per_type_metrics, classify_errors
from src.noise_analysis import classify_noise_attribution, aggregate_noise_analysis


def evaluate_dataset(sentences, predictions, entity_types):
    """Evaluate predictions against a single gold dataset."""
    all_gold = []
    all_pred = []
    all_errors = {
        "type_error": 0, "boundary_error": 0, "type_boundary_error": 0,
        "missing": 0, "spurious": 0,
    }

    pred_by_id = {p["id"]: p["predictions"] for p in predictions}

    for sent in sentences:
        gold_entities = sent["entities"]
        pred_entities = pred_by_id.get(sent["id"], [])
        all_gold.extend(gold_entities)
        all_pred.extend(pred_entities)

        errors = classify_errors(gold_entities, pred_entities)
        for k, v in errors.items():
            all_errors[k] += v

    overall = compute_entity_metrics(all_gold, all_pred)
    per_type = compute_per_type_metrics(all_gold, all_pred, entity_types)

    return {
        "overall": overall,
        "per_type": per_type,
        "errors": all_errors,
        "num_sentences": len(sentences),
        "num_gold_entities": len(all_gold),
        "num_pred_entities": len(all_pred),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GLiNER predictions")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]
    entity_types = [lbl["conll_label"] for lbl in config["labels"]]

    # Load data
    conll_path = os.path.join(processed_dir, f"conll03_{args.split}.json")
    clean_path = os.path.join(processed_dir, f"cleanconll_{args.split}.json")
    pred_conll_path = os.path.join(results_dir, f"predictions_conll03_{args.split}.json")
    pred_clean_path = os.path.join(results_dir, f"predictions_cleanconll_{args.split}.json")

    conll_sentences = load_sentences_json(conll_path)
    clean_sentences = load_sentences_json(clean_path)

    with open(pred_conll_path) as f:
        pred_conll = json.load(f)
    with open(pred_clean_path) as f:
        pred_clean = json.load(f)

    # Evaluate on CoNLL-03
    print("Evaluating on CoNLL-03...")
    conll_results = evaluate_dataset(conll_sentences, pred_conll, entity_types)
    conll_out = os.path.join(results_dir, f"metrics_conll03_{args.split}.json")
    with open(conll_out, "w") as f:
        json.dump(conll_results, f, indent=2)
    print(f"  Overall F1: {conll_results['overall']['f1']:.4f}")

    # Evaluate on CleanCoNLL
    print("Evaluating on CleanCoNLL...")
    clean_results = evaluate_dataset(clean_sentences, pred_clean, entity_types)
    clean_out = os.path.join(results_dir, f"metrics_cleanconll_{args.split}.json")
    with open(clean_out, "w") as f:
        json.dump(clean_results, f, indent=2)
    print(f"  Overall F1: {clean_results['overall']['f1']:.4f}")

    # Noise attribution
    print("\nRunning noise attribution analysis...")
    pred_by_id = {p["id"]: p["predictions"] for p in pred_conll}
    conll_by_id = {s["id"]: s["entities"] for s in conll_sentences}
    clean_by_id = {s["id"]: s["entities"] for s in clean_sentences}

    per_sentence_noise = []
    for sent_id in pred_by_id:
        if sent_id in conll_by_id and sent_id in clean_by_id:
            result = classify_noise_attribution(
                pred_by_id[sent_id],
                conll_by_id[sent_id],
                clean_by_id[sent_id],
            )
            result["sentence_id"] = sent_id
            per_sentence_noise.append(result)

    noise_agg = aggregate_noise_analysis(per_sentence_noise, max_examples=10)
    noise_out = os.path.join(results_dir, f"noise_analysis_{args.split}.json")
    with open(noise_out, "w") as f:
        json.dump(noise_agg, f, indent=2)

    print(f"  Noise penalized correct predictions: {noise_agg['noise_penalized_correct']}")
    print(f"  Model learned noisy patterns: {noise_agg['model_learned_noise']}")
    print(f"  Genuine errors: {noise_agg['genuine_error']}")
    print(f"  Correct on both: {noise_agg['correct_both']}")

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add evaluation script with metrics and noise attribution"
```

---

### Task 9: Implement `scripts/generate_report.py` — report generation

**Files:**
- Create: `scripts/generate_report.py`

**Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Generate comparison tables and findings report.

Reads evaluation results and produces Markdown + CSV summary artifacts.

Usage:
    python scripts/generate_report.py --config configs/default.yaml
"""
import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_comparison_table(conll_metrics, clean_metrics, entity_types):
    """Build a comparison DataFrame."""
    rows = []

    # Overall
    rows.append({
        "Entity Type": "**Overall**",
        "CoNLL-03 P": f"{conll_metrics['overall']['precision']:.4f}",
        "CoNLL-03 R": f"{conll_metrics['overall']['recall']:.4f}",
        "CoNLL-03 F1": f"{conll_metrics['overall']['f1']:.4f}",
        "CleanCoNLL P": f"{clean_metrics['overall']['precision']:.4f}",
        "CleanCoNLL R": f"{clean_metrics['overall']['recall']:.4f}",
        "CleanCoNLL F1": f"{clean_metrics['overall']['f1']:.4f}",
        "F1 Delta": f"{clean_metrics['overall']['f1'] - conll_metrics['overall']['f1']:+.4f}",
    })

    # Per type
    for etype in entity_types:
        c = conll_metrics["per_type"].get(etype, {})
        cl = clean_metrics["per_type"].get(etype, {})
        delta = cl.get("f1", 0) - c.get("f1", 0)
        rows.append({
            "Entity Type": etype,
            "CoNLL-03 P": f"{c.get('precision', 0):.4f}",
            "CoNLL-03 R": f"{c.get('recall', 0):.4f}",
            "CoNLL-03 F1": f"{c.get('f1', 0):.4f}",
            "CleanCoNLL P": f"{cl.get('precision', 0):.4f}",
            "CleanCoNLL R": f"{cl.get('recall', 0):.4f}",
            "CleanCoNLL F1": f"{cl.get('f1', 0):.4f}",
            "F1 Delta": f"{delta:+.4f}",
        })

    return pd.DataFrame(rows)


def generate_findings(conll_metrics, clean_metrics, noise_analysis, entity_types):
    """Generate findings markdown text."""
    lines = []
    lines.append("# Findings: Fair Evaluation of GLiNER on CleanCoNLL vs CoNLL-03\n")

    lines.append("## 1. Performance Comparison\n")
    conll_f1 = conll_metrics["overall"]["f1"]
    clean_f1 = clean_metrics["overall"]["f1"]
    delta = clean_f1 - conll_f1
    direction = "higher" if delta > 0 else "lower"
    lines.append(
        f"GLiNER (`knowledgator/gliner-multitask-large-v0.5`) achieves an overall "
        f"entity-level F1 of **{conll_f1:.4f}** on CoNLL-03 and **{clean_f1:.4f}** on "
        f"CleanCoNLL ({delta:+.4f}, {direction} on cleaned data).\n"
    )

    lines.append("### Per-Entity-Type Breakdown\n")
    for etype in entity_types:
        c_f1 = conll_metrics["per_type"].get(etype, {}).get("f1", 0)
        cl_f1 = clean_metrics["per_type"].get(etype, {}).get("f1", 0)
        d = cl_f1 - c_f1
        lines.append(f"- **{etype}**: CoNLL-03 F1={c_f1:.4f}, CleanCoNLL F1={cl_f1:.4f} ({d:+.4f})")
    lines.append("")

    lines.append("## 2. Error Category Changes\n")
    conll_err = conll_metrics["errors"]
    clean_err = clean_metrics["errors"]
    lines.append("| Error Type | CoNLL-03 | CleanCoNLL | Delta |")
    lines.append("|---|---|---|---|")
    for err_type in ["type_error", "boundary_error", "type_boundary_error", "missing", "spurious"]:
        c = conll_err.get(err_type, 0)
        cl = clean_err.get(err_type, 0)
        lines.append(f"| {err_type} | {c} | {cl} | {cl - c:+d} |")
    lines.append("")

    lines.append("## 3. Noise Attribution\n")
    lines.append(
        f"- **Noise penalized correct predictions**: {noise_analysis['noise_penalized_correct']} "
        f"(predictions that match CleanCoNLL but are marked wrong by CoNLL-03)\n"
        f"- **Model learned noisy patterns**: {noise_analysis['model_learned_noise']} "
        f"(predictions that match CoNLL-03 but are wrong per CleanCoNLL)\n"
        f"- **Genuine model errors**: {noise_analysis['genuine_error']} "
        f"(wrong on both datasets)\n"
        f"- **Correct on both**: {noise_analysis['correct_both']}\n"
    )

    if noise_analysis.get("examples_noise_penalized"):
        lines.append("### Examples: Noise Penalized Correct Predictions\n")
        for ex in noise_analysis["examples_noise_penalized"][:5]:
            lines.append(f"- Span [{ex['start']}:{ex['end']}] predicted as **{ex['label']}** — "
                         f"matches CleanCoNLL, wrong per CoNLL-03")
        lines.append("")

    lines.append("## 4. Conclusion\n")
    lines.append(
        "Label noise in CoNLL-03 introduces systematic measurement error in NER evaluation. "
        f"Of all entity-level disagreements, **{noise_analysis['noise_penalized_correct']}** "
        "cases represent correct model predictions that are unfairly penalized by noisy gold "
        "annotations. CleanCoNLL provides a fairer evaluation benchmark by correcting these "
        "annotation errors, leading to more accurate assessment of model capabilities.\n"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    results_dir = config["paths"]["results"]
    entity_types = [lbl["conll_label"] for lbl in config["labels"]]

    # Load metrics
    with open(os.path.join(results_dir, f"metrics_conll03_{args.split}.json")) as f:
        conll_metrics = json.load(f)
    with open(os.path.join(results_dir, f"metrics_cleanconll_{args.split}.json")) as f:
        clean_metrics = json.load(f)
    with open(os.path.join(results_dir, f"noise_analysis_{args.split}.json")) as f:
        noise_analysis = json.load(f)

    # Build comparison table
    df = build_comparison_table(conll_metrics, clean_metrics, entity_types)
    csv_path = os.path.join(results_dir, "comparison_table.csv")
    md_table_path = os.path.join(results_dir, "comparison_table.md")
    df.to_csv(csv_path, index=False)
    df.to_markdown(md_table_path, index=False)
    print(f"Comparison table saved to {csv_path} and {md_table_path}")

    # Generate findings
    findings = generate_findings(conll_metrics, clean_metrics, noise_analysis, entity_types)
    findings_path = os.path.join(results_dir, "findings.md")
    with open(findings_path, "w") as f:
        f.write(findings)
    print(f"Findings report saved to {findings_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/generate_report.py
git commit -m "feat: add report generation script"
```

---

### Task 10: Create `run_all.sh` and `README.md`

**Files:**
- Create: `run_all.sh`
- Create: `README.md`

**Step 1: Write `run_all.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

echo "=========================================="
echo "GLiNER CleanCoNLL Evaluation Pipeline"
echo "Config: $CONFIG"
echo "=========================================="

echo ""
echo "[1/4] Preparing data..."
python scripts/prepare_data.py --config "$CONFIG"

echo ""
echo "[2/4] Running GLiNER inference..."
python scripts/run_inference.py --config "$CONFIG"

echo ""
echo "[3/4] Evaluating predictions..."
python scripts/evaluate.py --config "$CONFIG"

echo ""
echo "[4/4] Generating report..."
python scripts/generate_report.py --config "$CONFIG"

echo ""
echo "=========================================="
echo "Pipeline complete! Results in results/"
echo "=========================================="
```

**Step 2: Write `README.md`**

See implementation for full content. Covers: project overview, setup, data preparation, running the pipeline, output files, project structure.

**Step 3: Commit**

```bash
chmod +x run_all.sh
git add run_all.sh README.md
git commit -m "feat: add run_all.sh pipeline runner and README"
```

---

### Task 11: Push to GitHub

**Step 1: Add remote and push**

```bash
git remote add origin https://github.com/rusydi16/PBML_Open-Type_NER_w_GLiNER_on_CleanCoNLL.git
git branch -M main
git push -u origin main
```
