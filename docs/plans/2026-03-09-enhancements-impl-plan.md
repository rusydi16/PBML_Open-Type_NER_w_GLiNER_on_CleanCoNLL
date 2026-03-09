# Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add statistical significance testing, model size ablation, and fine-tuning comparison to the GLiNER CleanCoNLL evaluation project.

**Architecture:** New src modules and scripts alongside existing pipeline. `run_all.sh` gains `--full`, `--bootstrap`, `--ablation`, `--finetune` flags.

**Tech Stack:** Python 3.10+, numpy (bootstrap), gliner[training] + accelerate (fine-tuning)

---

### Task 1: Implement `src/statistical_tests.py` with tests

**Files:**
- Create: `src/statistical_tests.py`
- Create: `tests/test_statistical_tests.py`

**Step 1: Write the failing tests**

```python
# tests/test_statistical_tests.py
"""Tests for bootstrap statistical significance testing."""
import pytest
from src.statistical_tests import bootstrap_entity_f1, paired_bootstrap_test


def _make_sentence(gold_entities, pred_entities, sid="s-0001"):
    """Helper to create a (gold_sentence, pred_entry) pair."""
    gold_sent = {"id": sid, "tokens": ["a", "b", "c"], "entities": gold_entities, "ner_tags": []}
    pred_entry = {"id": sid, "tokens": ["a", "b", "c"], "predictions": pred_entities}
    return gold_sent, pred_entry


class TestBootstrapEntityF1:
    def test_perfect_predictions(self):
        gold_ents = [{"start": 0, "end": 1, "label": "PER"}]
        pred_ents = [{"start_token": 0, "end_token": 1, "label": "PER", "text": "a", "score": 0.9}]
        pairs = [_make_sentence(gold_ents, pred_ents, f"s-{i:04d}") for i in range(50)]
        gold_sents = [p[0] for p in pairs]
        pred_entries = [p[1] for p in pairs]
        result = bootstrap_entity_f1(gold_sents, pred_entries, n_iterations=100, seed=42)
        assert result["mean"] == pytest.approx(1.0, abs=0.01)
        assert result["ci_lower"] >= 0.95
        assert result["ci_upper"] <= 1.01

    def test_returns_correct_keys(self):
        gold_ents = [{"start": 0, "end": 1, "label": "PER"}]
        pred_ents = [{"start_token": 0, "end_token": 1, "label": "ORG", "text": "a", "score": 0.9}]
        pairs = [_make_sentence(gold_ents, pred_ents, f"s-{i:04d}") for i in range(20)]
        gold_sents = [p[0] for p in pairs]
        pred_entries = [p[1] for p in pairs]
        result = bootstrap_entity_f1(gold_sents, pred_entries, n_iterations=50, seed=42)
        assert "mean" in result
        assert "std" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]


class TestPairedBootstrapTest:
    def test_identical_datasets(self):
        gold_ents = [{"start": 0, "end": 1, "label": "PER"}]
        pred_ents = [{"start_token": 0, "end_token": 1, "label": "PER", "text": "a", "score": 0.9}]
        pairs = [_make_sentence(gold_ents, pred_ents, f"s-{i:04d}") for i in range(50)]
        gold_a = [p[0] for p in pairs]
        pred_a = [p[1] for p in pairs]
        result = paired_bootstrap_test(gold_a, pred_a, gold_a, pred_a, n_iterations=100, seed=42)
        assert "delta_mean" in result
        assert "p_value" in result
        assert result["delta_mean"] == pytest.approx(0.0, abs=0.01)

    def test_returns_correct_keys(self):
        gold_ents = [{"start": 0, "end": 1, "label": "PER"}]
        pred_ents = [{"start_token": 0, "end_token": 1, "label": "PER", "text": "a", "score": 0.9}]
        pairs = [_make_sentence(gold_ents, pred_ents, f"s-{i:04d}") for i in range(20)]
        gold_sents = [p[0] for p in pairs]
        pred_entries = [p[1] for p in pairs]
        result = paired_bootstrap_test(gold_sents, pred_entries, gold_sents, pred_entries, n_iterations=50, seed=42)
        for key in ["delta_mean", "delta_std", "delta_ci_lower", "delta_ci_upper", "p_value"]:
            assert key in result
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_statistical_tests.py -v
```
Expected: FAIL (ImportError)

**Step 3: Implement `src/statistical_tests.py`**

```python
"""Bootstrap resampling for statistical significance testing of NER metrics."""
import numpy as np
from src.metrics import compute_entity_metrics


def _compute_f1_for_sample(gold_sentences, pred_entries, indices):
    """Compute entity-level F1 for a bootstrap sample of sentence indices."""
    all_gold = []
    all_pred = []
    pred_by_id = {p["id"]: p["predictions"] for p in pred_entries}
    for idx in indices:
        sent = gold_sentences[idx]
        all_gold.extend(sent["entities"])
        all_pred.extend(pred_by_id.get(sent["id"], []))
    metrics = compute_entity_metrics(all_gold, all_pred)
    return metrics["f1"]


def bootstrap_entity_f1(gold_sentences, pred_entries, n_iterations=1000, seed=42, confidence=0.95):
    """Compute bootstrapped confidence interval for entity-level F1.

    Args:
        gold_sentences: List of gold sentence dicts (with entities).
        pred_entries: List of prediction dicts (with id and predictions).
        n_iterations: Number of bootstrap iterations.
        seed: Random seed.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Dict with mean, std, ci_lower, ci_upper.
    """
    rng = np.random.RandomState(seed)
    n = len(gold_sentences)
    f1_scores = []
    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        f1 = _compute_f1_for_sample(gold_sentences, pred_entries, indices)
        f1_scores.append(f1)
    f1_scores = np.array(f1_scores)
    alpha = 1 - confidence
    return {
        "mean": round(float(np.mean(f1_scores)), 4),
        "std": round(float(np.std(f1_scores)), 4),
        "ci_lower": round(float(np.percentile(f1_scores, 100 * alpha / 2)), 4),
        "ci_upper": round(float(np.percentile(f1_scores, 100 * (1 - alpha / 2))), 4),
    }


def paired_bootstrap_test(
    gold_a, pred_a, gold_b, pred_b, n_iterations=1000, seed=42, confidence=0.95
):
    """Paired bootstrap test comparing F1 between two evaluation setups.

    Tests whether F1(dataset_b) - F1(dataset_a) is significantly different from 0.

    Args:
        gold_a, pred_a: Gold and predictions for dataset A (e.g., CoNLL-03).
        gold_b, pred_b: Gold and predictions for dataset B (e.g., CleanCoNLL).
        n_iterations: Number of bootstrap iterations.
        seed: Random seed.
        confidence: Confidence level.

    Returns:
        Dict with delta_mean, delta_std, delta_ci_lower, delta_ci_upper, p_value.
        p_value is the proportion of bootstrap deltas <= 0 (one-sided test).
    """
    rng = np.random.RandomState(seed)
    n = min(len(gold_a), len(gold_b))
    deltas = []
    for _ in range(n_iterations):
        indices = rng.choice(n, size=n, replace=True)
        f1_a = _compute_f1_for_sample(gold_a, pred_a, indices)
        f1_b = _compute_f1_for_sample(gold_b, pred_b, indices)
        deltas.append(f1_b - f1_a)
    deltas = np.array(deltas)
    alpha = 1 - confidence
    p_value = float(np.mean(deltas <= 0))
    return {
        "delta_mean": round(float(np.mean(deltas)), 4),
        "delta_std": round(float(np.std(deltas)), 4),
        "delta_ci_lower": round(float(np.percentile(deltas, 100 * alpha / 2)), 4),
        "delta_ci_upper": round(float(np.percentile(deltas, 100 * (1 - alpha / 2))), 4),
        "p_value": round(p_value, 4),
    }
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_statistical_tests.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/statistical_tests.py tests/test_statistical_tests.py
git commit -m "feat: add bootstrap statistical significance testing module"
```

---

### Task 2: Integrate bootstrap into `scripts/evaluate.py`

**Files:**
- Modify: `scripts/evaluate.py`

**Step 1: Add `--bootstrap` flag and `--n-bootstrap` arg to the argument parser**

After the `--split` argument, add:

```python
parser.add_argument(
    "--bootstrap", action="store_true",
    help="Run bootstrap significance testing (slower)",
)
parser.add_argument(
    "--n-bootstrap", type=int, default=1000,
    help="Number of bootstrap iterations (default: 1000)",
)
```

**Step 2: Add bootstrap logic at the end of main(), before the final print**

After the noise attribution section, add:

```python
# Bootstrap significance testing (optional)
if args.bootstrap:
    from src.statistical_tests import bootstrap_entity_f1, paired_bootstrap_test

    print(f"\n{'='*60}")
    print(f"Bootstrap Significance Testing (n={args.n_bootstrap})")
    print(f"{'='*60}")

    for dataset in datasets:
        pred_list = pred[dataset]
        gold_list = gold[dataset]
        bs = bootstrap_entity_f1(gold_list, pred_list, n_iterations=args.n_bootstrap, seed=config["seed"])
        bs_path = os.path.join(results_dir, f"bootstrap_{dataset}_{split}.json")
        with open(bs_path, "w", encoding="utf-8") as f:
            json.dump(bs, f, indent=2)
        print(f"  {dataset} F1: {bs['mean']:.4f} [{bs['ci_lower']:.4f}, {bs['ci_upper']:.4f}]")

    # Paired test
    sig = paired_bootstrap_test(
        gold["conll03"], pred["conll03"],
        gold["cleanconll"], pred["cleanconll"],
        n_iterations=args.n_bootstrap, seed=config["seed"],
    )
    sig_path = os.path.join(results_dir, f"significance_test_{split}.json")
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(sig, f, indent=2)
    print(f"  Delta (CleanCoNLL - CoNLL-03): {sig['delta_mean']:+.4f} [{sig['delta_ci_lower']:+.4f}, {sig['delta_ci_upper']:+.4f}]")
    print(f"  p-value: {sig['p_value']:.4f}")
```

**Step 3: Commit**

```bash
git add scripts/evaluate.py
git commit -m "feat: add --bootstrap flag to evaluate.py for significance testing"
```

---

### Task 3: Create `configs/ablation.yaml` and `scripts/run_ablation.py`

**Files:**
- Create: `configs/ablation.yaml`
- Create: `scripts/run_ablation.py`

**Step 1: Create `configs/ablation.yaml`**

```yaml
models:
  - name: "urchade/gliner_small-v2.1"
    short_name: "small"
    params: "44M"
  - name: "urchade/gliner_medium-v2.1"
    short_name: "medium"
    params: "86M"
  - name: "urchade/gliner_large-v2.1"
    short_name: "large"
    params: "304M"

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
  processed_data: "data/processed"
  results: "results"

seed: 42
```

**Step 2: Create `scripts/run_ablation.py`**

```python
#!/usr/bin/env python3
"""Run model size ablation: inference + evaluation across multiple GLiNER variants.

Usage:
    python scripts/run_ablation.py --config configs/ablation.yaml
"""
import argparse
import json
import os
import sys

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_sentences_json
from src.inference import load_gliner_model, predict_sentence, set_seed
from src.metrics import compute_entity_metrics, compute_per_type_metrics, classify_errors
from src.noise_analysis import classify_noise_attribution, aggregate_noise_analysis


def run_inference_for_model(model, sentences, gliner_labels, label_map, threshold):
    """Run inference on all sentences with a loaded model."""
    import torch
    results = []
    total = 0
    for sent in tqdm(sentences, desc="  Predicting"):
        preds = predict_sentence(model, sent["tokens"], gliner_labels, label_map, threshold)
        total += len(preds)
        results.append({"id": sent["id"], "tokens": sent["tokens"], "predictions": preds})
    return results, total


def evaluate_predictions(gold_sentences, pred_entries, entity_types):
    """Evaluate predictions against gold. Returns metrics dict."""
    pred_by_id = {p["id"]: p["predictions"] for p in pred_entries}
    all_gold = []
    all_pred = []
    error_counts = {"type_error": 0, "boundary_error": 0, "type_boundary_error": 0, "missing": 0, "spurious": 0}
    for sent in gold_sentences:
        gold_ents = sent["entities"]
        pred_ents = pred_by_id.get(sent["id"], [])
        all_gold.extend(gold_ents)
        all_pred.extend(pred_ents)
        errs = classify_errors(gold_ents, pred_ents)
        for k in error_counts:
            error_counts[k] += errs[k]
    overall = compute_entity_metrics(all_gold, all_pred)
    per_type = compute_per_type_metrics(all_gold, all_pred, entity_types)
    return {"overall": overall, "per_type": per_type, "errors": error_counts}


def main():
    parser = argparse.ArgumentParser(description="Run model size ablation study")
    parser.add_argument("--config", default="configs/ablation.yaml")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    gliner_labels = [l["gliner_label"] for l in config["labels"]]
    label_map = {l["gliner_label"]: l["conll_label"] for l in config["labels"]}
    entity_types = [l["conll_label"] for l in config["labels"]]
    threshold = config["threshold"]
    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]

    datasets = ["conll03", "cleanconll"]
    gold = {}
    for ds in datasets:
        path = os.path.join(processed_dir, f"{ds}_{args.split}.json")
        gold[ds] = load_sentences_json(path)
        print(f"Loaded {ds}: {len(gold[ds])} sentences")

    ablation_results = []

    for model_info in config["models"]:
        model_name = model_info["name"]
        short_name = model_info["short_name"]
        params = model_info["params"]
        out_dir = os.path.join(results_dir, "ablation", short_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({params})")
        print(f"{'='*60}")

        model = load_gliner_model(model_name)

        row = {"model": short_name, "params": params}

        for ds in datasets:
            print(f"\n  Dataset: {ds}")
            preds, n_preds = run_inference_for_model(model, gold[ds], gliner_labels, label_map, threshold)

            # Save predictions
            pred_path = os.path.join(out_dir, f"predictions_{ds}_{args.split}.json")
            with open(pred_path, "w") as f:
                json.dump(preds, f, indent=2, ensure_ascii=False)

            # Evaluate
            metrics = evaluate_predictions(gold[ds], preds, entity_types)
            metrics_path = os.path.join(out_dir, f"metrics_{ds}_{args.split}.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            row[f"{ds}_f1"] = metrics["overall"]["f1"]
            row[f"{ds}_p"] = metrics["overall"]["precision"]
            row[f"{ds}_r"] = metrics["overall"]["recall"]
            print(f"    F1: {metrics['overall']['f1']:.4f}")

        # Noise attribution
        pred_by_id = {p["id"]: p["predictions"] for p in preds}  # uses last dataset's preds (cleanconll)
        # Reload conll03 preds for noise analysis
        conll_pred_path = os.path.join(out_dir, f"predictions_conll03_{args.split}.json")
        with open(conll_pred_path) as f:
            conll_preds = json.load(f)
        conll_pred_by_id = {p["id"]: p["predictions"] for p in conll_preds}
        gold_conll_by_id = {s["id"]: s["entities"] for s in gold["conll03"]}
        gold_clean_by_id = {s["id"]: s["entities"] for s in gold["cleanconll"]}
        common_ids = set(conll_pred_by_id) & set(gold_conll_by_id) & set(gold_clean_by_id)

        per_sent = []
        for sid in sorted(common_ids):
            r = classify_noise_attribution(conll_pred_by_id[sid], gold_conll_by_id[sid], gold_clean_by_id[sid])
            per_sent.append(r)
        noise_agg = aggregate_noise_analysis(per_sent, max_examples=5)
        noise_path = os.path.join(out_dir, f"noise_analysis_{args.split}.json")
        with open(noise_path, "w") as f:
            json.dump(noise_agg, f, indent=2)

        row["noise_penalized"] = noise_agg["noise_penalized_correct"]
        row["f1_delta"] = round(row.get("cleanconll_f1", 0) - row.get("conll03_f1", 0), 4)

        ablation_results.append(row)
        del model  # Free memory before loading next model

    # Generate ablation table
    import pandas as pd
    df = pd.DataFrame(ablation_results)
    table_path = os.path.join(results_dir, "ablation_table.csv")
    md_path = os.path.join(results_dir, "ablation_table.md")
    df.to_csv(table_path, index=False)
    df.to_markdown(md_path, index=False)
    print(f"\nAblation table saved to {table_path} and {md_path}")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add configs/ablation.yaml scripts/run_ablation.py
git commit -m "feat: add model size ablation script and config"
```

---

### Task 4: Implement `src/finetune.py` with tests

**Files:**
- Create: `src/finetune.py`
- Create: `tests/test_finetune.py`

**Step 1: Write failing tests**

```python
# tests/test_finetune.py
"""Tests for fine-tuning data conversion utilities."""
from src.finetune import convert_sentence_to_gliner_format, convert_dataset_to_gliner_format


class TestConvertSentenceToGlinerFormat:
    def test_basic_conversion(self):
        sentence = {
            "id": "test-0001",
            "tokens": ["John", "lives", "in", "New", "York"],
            "entities": [
                {"text": "John", "start": 0, "end": 1, "label": "PER"},
                {"text": "New York", "start": 3, "end": 5, "label": "LOC"},
            ],
            "ner_tags": ["B-PER", "O", "O", "B-LOC", "I-LOC"],
        }
        conll_to_gliner = {"PER": "person", "LOC": "location", "ORG": "organization", "MISC": "miscellaneous"}
        result = convert_sentence_to_gliner_format(sentence, conll_to_gliner)
        assert result["tokenized_text"] == ["John", "lives", "in", "New", "York"]
        assert [0, 0, "person"] in result["ner"]
        assert [3, 4, "location"] in result["ner"]

    def test_empty_entities(self):
        sentence = {
            "id": "test-0002",
            "tokens": ["The", "cat", "sat"],
            "entities": [],
            "ner_tags": ["O", "O", "O"],
        }
        conll_to_gliner = {"PER": "person"}
        result = convert_sentence_to_gliner_format(sentence, conll_to_gliner)
        assert result["tokenized_text"] == ["The", "cat", "sat"]
        assert result["ner"] == []

    def test_exclusive_to_inclusive_end(self):
        sentence = {
            "id": "test-0003",
            "tokens": ["A", "B", "C"],
            "entities": [{"text": "A B C", "start": 0, "end": 3, "label": "ORG"}],
            "ner_tags": ["B-ORG", "I-ORG", "I-ORG"],
        }
        conll_to_gliner = {"ORG": "organization"}
        result = convert_sentence_to_gliner_format(sentence, conll_to_gliner)
        # end 3 (exclusive) -> 2 (inclusive)
        assert [0, 2, "organization"] in result["ner"]


class TestConvertDatasetToGlinerFormat:
    def test_converts_multiple_sentences(self):
        sentences = [
            {"id": "s-1", "tokens": ["A"], "entities": [{"text": "A", "start": 0, "end": 1, "label": "PER"}], "ner_tags": ["B-PER"]},
            {"id": "s-2", "tokens": ["B"], "entities": [], "ner_tags": ["O"]},
        ]
        conll_to_gliner = {"PER": "person"}
        result = convert_dataset_to_gliner_format(sentences, conll_to_gliner)
        assert len(result) == 2
        assert result[0]["ner"] == [[0, 0, "person"]]
        assert result[1]["ner"] == []
```

**Step 2: Run tests to verify they fail**

```bash
python3 -m pytest tests/test_finetune.py -v
```

**Step 3: Implement `src/finetune.py`**

```python
"""Fine-tuning utilities: data conversion and GLiNER training wrapper."""
import json
import os
from typing import Any


def convert_sentence_to_gliner_format(sentence: dict, conll_to_gliner: dict[str, str]) -> dict[str, Any]:
    """Convert a sentence dict to GLiNER training format.

    Our format: entities with exclusive end indices and CoNLL labels (PER, ORG, etc.)
    GLiNER format: inclusive end indices and natural language labels (person, organization, etc.)

    Args:
        sentence: Sentence dict with tokens and entities.
        conll_to_gliner: Mapping from CoNLL labels to GLiNER natural language labels.

    Returns:
        Dict with tokenized_text and ner fields.
    """
    ner = []
    for entity in sentence["entities"]:
        start = entity["start"]
        end = entity["end"] - 1  # Convert exclusive to inclusive
        gliner_label = conll_to_gliner.get(entity["label"], entity["label"])
        ner.append([start, end, gliner_label])
    return {
        "tokenized_text": sentence["tokens"],
        "ner": ner,
    }


def convert_dataset_to_gliner_format(sentences: list[dict], conll_to_gliner: dict[str, str]) -> list[dict]:
    """Convert a list of sentences to GLiNER training format."""
    return [convert_sentence_to_gliner_format(s, conll_to_gliner) for s in sentences]


def save_gliner_training_data(data: list[dict], filepath: str) -> None:
    """Save GLiNER training data as JSON."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def finetune_gliner(
    model_name: str,
    train_data: list[dict],
    output_dir: str,
    max_steps: int = 2000,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    seed: int = 42,
    eval_data: list[dict] | None = None,
):
    """Fine-tune a GLiNER model.

    Args:
        model_name: HuggingFace model name to start from.
        train_data: Training data in GLiNER format.
        output_dir: Where to save the fine-tuned model.
        max_steps: Maximum training steps.
        learning_rate: Learning rate for encoder.
        batch_size: Per-device training batch size.
        seed: Random seed.
        eval_data: Optional evaluation data.
    """
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_name)

    train_kwargs = {
        "train_data": train_data,
        "output_dir": output_dir,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": batch_size,
        "seed": seed,
        "save_steps": max_steps,  # Save only at the end
        "logging_steps": 100,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
    }

    if eval_data:
        train_kwargs["eval_data"] = eval_data
        train_kwargs["evaluation_strategy"] = "steps"
        train_kwargs["eval_steps"] = 500

    model.train_model(**train_kwargs)
    return model
```

**Step 4: Run tests**

```bash
python3 -m pytest tests/test_finetune.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/finetune.py tests/test_finetune.py
git commit -m "feat: add fine-tuning data conversion and training wrapper"
```

---

### Task 5: Create `scripts/run_finetune.py`

**Files:**
- Create: `scripts/run_finetune.py`
- Create: `configs/finetune.yaml`

**Step 1: Create `configs/finetune.yaml`**

```yaml
base_model: "urchade/gliner_medium-v2.1"

training:
  max_steps: 2000
  learning_rate: 1e-5
  batch_size: 8
  warmup_ratio: 0.1

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
  processed_data: "data/processed"
  results: "results"
  models: "models"

seed: 42
threshold: 0.5
```

**Step 2: Create `scripts/run_finetune.py`**

```python
#!/usr/bin/env python3
"""Fine-tune GLiNER on CoNLL-03 vs CleanCoNLL and evaluate both on CleanCoNLL test set.

Usage:
    python scripts/run_finetune.py --config configs/finetune.yaml
"""
import argparse
import json
import os
import sys

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_sentences_json
from src.finetune import convert_dataset_to_gliner_format, finetune_gliner
from src.inference import load_gliner_model, predict_sentence, set_seed
from src.metrics import compute_entity_metrics, compute_per_type_metrics, classify_errors
from src.noise_analysis import classify_noise_attribution, aggregate_noise_analysis


def evaluate_model(model, gold_sentences, gliner_labels, label_map, entity_types, threshold):
    """Run inference and evaluate a model on gold sentences."""
    pred_entries = []
    for sent in tqdm(gold_sentences, desc="  Evaluating"):
        preds = predict_sentence(model, sent["tokens"], gliner_labels, label_map, threshold)
        pred_entries.append({"id": sent["id"], "tokens": sent["tokens"], "predictions": preds})

    pred_by_id = {p["id"]: p["predictions"] for p in pred_entries}
    all_gold = []
    all_pred = []
    error_counts = {"type_error": 0, "boundary_error": 0, "type_boundary_error": 0, "missing": 0, "spurious": 0}

    for sent in gold_sentences:
        all_gold.extend(sent["entities"])
        all_pred.extend(pred_by_id.get(sent["id"], []))
        errs = classify_errors(sent["entities"], pred_by_id.get(sent["id"], []))
        for k in error_counts:
            error_counts[k] += errs[k]

    overall = compute_entity_metrics(all_gold, all_pred)
    per_type = compute_per_type_metrics(all_gold, all_pred, entity_types)
    return {"overall": overall, "per_type": per_type, "errors": error_counts}, pred_entries


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GLiNER and compare")
    parser.add_argument("--config", default="configs/finetune.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, just evaluate existing models")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])

    # Label mappings
    gliner_labels = [l["gliner_label"] for l in config["labels"]]
    label_map = {l["gliner_label"]: l["conll_label"] for l in config["labels"]}
    conll_to_gliner = {l["conll_label"]: l["gliner_label"] for l in config["labels"]}
    entity_types = [l["conll_label"] for l in config["labels"]]
    threshold = config["threshold"]

    processed_dir = config["paths"]["processed_data"]
    results_dir = config["paths"]["results"]
    models_dir = config["paths"]["models"]

    finetune_results_dir = os.path.join(results_dir, "finetune")
    os.makedirs(finetune_results_dir, exist_ok=True)

    # Load eval data (CleanCoNLL test set)
    eval_gold = load_sentences_json(os.path.join(processed_dir, f"cleanconll_{args.split}.json"))
    print(f"Evaluation set: CleanCoNLL {args.split} ({len(eval_gold)} sentences)")

    training_configs = [
        {"name": "conll03", "train_file": f"conll03_train.json", "label": "CoNLL-03 (noisy)"},
        {"name": "cleanconll", "train_file": f"cleanconll_train.json", "label": "CleanCoNLL (clean)"},
    ]

    ft_results = []

    for tc in training_configs:
        model_dir = os.path.join(models_dir, f"finetuned_{tc['name']}")
        print(f"\n{'='*60}")
        print(f"Training: {tc['label']}")
        print(f"{'='*60}")

        if not args.skip_training:
            # Load and convert training data
            train_sentences = load_sentences_json(os.path.join(processed_dir, tc["train_file"]))
            train_data = convert_dataset_to_gliner_format(train_sentences, conll_to_gliner)
            print(f"  Training sentences: {len(train_data)}")

            # Fine-tune
            model = finetune_gliner(
                model_name=config["base_model"],
                train_data=train_data,
                output_dir=model_dir,
                max_steps=config["training"]["max_steps"],
                learning_rate=config["training"]["learning_rate"],
                batch_size=config["training"]["batch_size"],
                seed=config["seed"],
            )
            print(f"  Model saved to: {model_dir}")
        else:
            print(f"  Loading existing model from: {model_dir}")
            model = load_gliner_model(model_dir)

        # Evaluate on CleanCoNLL test
        print(f"\n  Evaluating on CleanCoNLL {args.split}...")
        metrics, pred_entries = evaluate_model(model, eval_gold, gliner_labels, label_map, entity_types, threshold)

        # Save
        metrics_path = os.path.join(finetune_results_dir, f"metrics_{tc['name']}_{args.split}.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        pred_path = os.path.join(finetune_results_dir, f"predictions_{tc['name']}_{args.split}.json")
        with open(pred_path, "w") as f:
            json.dump(pred_entries, f, indent=2, ensure_ascii=False)

        print(f"  F1: {metrics['overall']['f1']:.4f}")

        ft_results.append({
            "training_data": tc["label"],
            "eval_f1": metrics["overall"]["f1"],
            "eval_p": metrics["overall"]["precision"],
            "eval_r": metrics["overall"]["recall"],
            "type_errors": metrics["errors"]["type_error"],
            "boundary_errors": metrics["errors"]["boundary_error"],
            "missing": metrics["errors"]["missing"],
            "spurious": metrics["errors"]["spurious"],
        })

        del model

    # Generate comparison table
    import pandas as pd
    df = pd.DataFrame(ft_results)
    table_path = os.path.join(finetune_results_dir, "finetune_table.csv")
    md_path = os.path.join(finetune_results_dir, "finetune_table.md")
    df.to_csv(table_path, index=False)
    df.to_markdown(md_path, index=False)
    print(f"\nFine-tuning comparison saved to {table_path}")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add configs/finetune.yaml scripts/run_finetune.py
git commit -m "feat: add fine-tuning comparison script and config"
```

---

### Task 6: Update `run_all.sh`, `requirements.txt`, `.gitignore`, and `README.md`

**Files:**
- Modify: `run_all.sh`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Modify: `README.md`

**Step 1: Update `run_all.sh` to support flags**

Replace the entire file with:

```bash
#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
ABLATION_CONFIG="${ABLATION_CONFIG:-configs/ablation.yaml}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/finetune.yaml}"

# Parse flags
RUN_BOOTSTRAP=false
RUN_ABLATION=false
RUN_FINETUNE=false
RUN_FULL=false

for arg in "$@"; do
    case $arg in
        --full) RUN_FULL=true ;;
        --bootstrap) RUN_BOOTSTRAP=true ;;
        --ablation) RUN_ABLATION=true ;;
        --finetune) RUN_FINETUNE=true ;;
        *) CONFIG="$arg" ;;
    esac
done

if $RUN_FULL; then
    RUN_BOOTSTRAP=true
    RUN_ABLATION=true
    RUN_FINETUNE=true
fi

echo "=========================================="
echo "GLiNER CleanCoNLL Evaluation Pipeline"
echo "=========================================="

echo ""
echo "[1/4] Preparing data..."
python scripts/prepare_data.py --config "$CONFIG"

echo ""
echo "[2/4] Running GLiNER inference..."
python scripts/run_inference.py --config "$CONFIG"

echo ""
echo "[3/4] Evaluating predictions..."
if $RUN_BOOTSTRAP; then
    python scripts/evaluate.py --config "$CONFIG" --bootstrap
else
    python scripts/evaluate.py --config "$CONFIG"
fi

echo ""
echo "[4/4] Generating report..."
python scripts/generate_report.py --config "$CONFIG"

if $RUN_ABLATION; then
    echo ""
    echo "[Extra] Running model size ablation..."
    python scripts/run_ablation.py --config "$ABLATION_CONFIG"
fi

if $RUN_FINETUNE; then
    echo ""
    echo "[Extra] Running fine-tuning comparison..."
    python scripts/run_finetune.py --config "$FINETUNE_CONFIG"
fi

echo ""
echo "=========================================="
echo "Pipeline complete! Results in results/"
echo "=========================================="
```

**Step 2: Update `requirements.txt`** — add `accelerate>=0.20.0` at the end.

**Step 3: Update `.gitignore`** — add `models/` directory.

**Step 4: Update `README.md`** — add sections for the three new capabilities:
- Bootstrap significance testing (`--bootstrap` flag)
- Model size ablation (`scripts/run_ablation.py`)
- Fine-tuning comparison (`scripts/run_finetune.py`)
- Full pipeline: `bash run_all.sh --full`

**Step 5: Commit**

```bash
git add run_all.sh requirements.txt .gitignore README.md
git commit -m "feat: update pipeline runner, deps, and docs for enhancements"
```

---

### Task 7: Push to GitHub

**Step 1: Push**

```bash
git push origin main
```
