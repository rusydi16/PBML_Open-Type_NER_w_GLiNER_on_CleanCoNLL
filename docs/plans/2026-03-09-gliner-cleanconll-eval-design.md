# Design: Fair Evaluation of Open-Type NER with GLiNER on CleanCoNLL

**Date:** 2026-03-09
**Status:** Approved

## Goal

Compare GLiNER NER evaluation on CoNLL-03 vs CleanCoNLL to quantify how label noise affects measured performance and error patterns.

## Decisions

- **Model:** `gliner-multitask-large-v0.5`
- **Architecture:** Standalone scripts in `scripts/` with shared helpers in `src/`. Chained by `run_all.sh`.
- **CoNLL-03 source:** User places raw files manually in `data/raw/`
- **CleanCoNLL source:** Official patch approach (clone repo, apply corrections)
- **Prediction format:** Span list with token offsets
- **Eval splits:** Test only
- **Noise examples:** Top 10 per category

## Repository Structure

```
Ver_1/
├── configs/default.yaml
├── data/raw/                  # gitignored, user places eng.train/testa/testb
├── scripts/
│   ├── prepare_data.py
│   ├── run_inference.py
│   ├── evaluate.py
│   └── generate_report.py
├── src/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── inference.py
│   ├── metrics.py
│   └── noise_analysis.py
├── results/                   # gitignored except .gitkeep
├── requirements.txt
├── run_all.sh
├── .gitignore
└── README.md
```

## Pipeline Steps

### 1. Data Preparation (`scripts/prepare_data.py`)

- Parse CoNLL-03 column format from `data/raw/eng.{train,testa,testb}`
- Output standardized JSON per split: `data/processed/{conll03,cleanconll}_{train,dev,test}.json`
- Each sentence:
  ```json
  {
    "id": "test-0001",
    "tokens": ["EU", "rejects", ...],
    "ner_tags": ["B-ORG", "O", ...],
    "entities": [{"text": "EU", "start": 0, "end": 1, "label": "ORG"}]
  }
  ```
- CleanCoNLL: clone official repo, apply correction TSV to parsed CoNLL-03 data
- Output files share identical sentence IDs and tokens, differing only in NER tags

### 2. GLiNER Inference (`scripts/run_inference.py`)

- Load model via `gliner` Python package
- Label mapping from YAML config:
  - PER → "person", ORG → "organization", LOC → "location", MISC → "miscellaneous"
- Reconstruct raw text from tokens, track char-to-token offsets
- Run `model.predict_entities(text, labels, threshold)` per sentence
- Map predicted char spans back to token indices
- Save predictions as JSON with fields: text, start_token, end_token, label, score
- Deterministic: set seeds, `torch.no_grad()`

### 3. Evaluation (`scripts/evaluate.py`)

**Entity-level exact-match metrics:**
- P/R/F1 overall and per-type (PER/ORG/LOC/MISC)
- Exact match = same (start_token, end_token) AND same label

**Error categorization (for non-matches):**
- Type error: span matches, label differs
- Boundary error: overlapping span, different boundaries, same label
- Type+Boundary error: overlapping span, different boundaries AND label
- Missing (FN): gold entity with no overlapping prediction
- Spurious (FP): prediction with no overlapping gold entity

**Noise attribution:**
- Compare pred vs CoNLL-03 vs CleanCoNLL per entity
- Categories:
  - `pred == cleanconll, pred != conll03` → noise penalized correct prediction
  - `pred == conll03, pred != cleanconll` → model learned noisy pattern
  - `pred != both` → genuine model error
- Output: counts + top 10 examples per category

### 4. Report Generation (`scripts/generate_report.py`)

- `results/comparison_table.csv` + `results/comparison_table.md`: side-by-side P/R/F1
- `results/findings.md`: 1-2 page summary covering:
  - Performance differences
  - Error category changes
  - Noise attribution summary
  - Conclusion on fair evaluation with CleanCoNLL

## Config (`configs/default.yaml`)

```yaml
model:
  name: "gliner-multitask-large-v0.5"
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

## Dependencies

- Python 3.10+
- gliner
- transformers
- torch
- numpy
- pandas
- pyyaml
- seqeval (for reference metrics validation)

## Execution

```bash
bash run_all.sh
# Equivalent to:
# python scripts/prepare_data.py --config configs/default.yaml
# python scripts/run_inference.py --config configs/default.yaml
# python scripts/evaluate.py --config configs/default.yaml
# python scripts/generate_report.py --config configs/default.yaml
```
