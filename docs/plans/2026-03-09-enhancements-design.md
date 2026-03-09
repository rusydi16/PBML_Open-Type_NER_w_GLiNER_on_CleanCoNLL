# Design: Project Enhancements — Statistical Tests, Ablation, Fine-tuning

**Date:** 2026-03-09
**Status:** Approved

## Goal

Add three enhancements to strengthen the evaluation study:
1. Statistical significance testing (bootstrap resampling)
2. Model size ablation (small/medium/large)
3. Fine-tuning comparison (train on CoNLL-03 vs CleanCoNLL)

## Architecture

New scripts alongside existing pipeline. `run_all.sh --full` runs everything.

## Enhancement 1: Statistical Significance Testing

**New file:** `src/statistical_tests.py`

- Bootstrap resampling (N=1000 iterations, configurable)
- Resample sentences with replacement
- Compute entity-level F1 per bootstrap sample for both datasets
- Compute paired difference (CleanCoNLL F1 - CoNLL-03 F1) per iteration
- Output: 95% CI for each dataset's F1, 95% CI for delta, p-value

**Integration:** `scripts/evaluate.py` gains `--bootstrap` flag (default off).

**Output files:**
- `results/bootstrap_conll03.json` — {mean, std, ci_lower, ci_upper}
- `results/bootstrap_cleanconll.json`
- `results/significance_test.json` — {delta_mean, delta_ci, p_value}

## Enhancement 2: Model Size Ablation

**New file:** `scripts/run_ablation.py`

**Models:**
- `urchade/gliner_small-v2.1` (44M params)
- `urchade/gliner_medium-v2.1` (86M params)
- `urchade/gliner_large-v2.1` (304M params)

**Process per model:**
1. Load model
2. Run inference on both test sets (CoNLL-03 + CleanCoNLL)
3. Run evaluation (metrics + noise attribution)
4. Save to `results/ablation/{model_short_name}/`

**New config:** `configs/ablation.yaml` — model list with names and param counts.

**Output:** `results/ablation_table.md` — comparison across model sizes showing F1 delta and noise impact.

## Enhancement 3: Fine-tuning Comparison

**New files:**
- `src/finetune.py` — data conversion + training wrapper
- `scripts/run_finetune.py` — CLI script

**Data conversion:**
- Our format: `{"tokens": [...], "entities": [{"start": 0, "end": 2, "label": "PER"}]}`
- GLiNER format: `{"tokenized_text": [...], "ner": [[0, 1, "person"], ...]}`
- End index: our format is exclusive, GLiNER is inclusive
- Labels: map CoNLL labels (PER) to natural language (person)

**Training:**
- Base model: `urchade/gliner_medium-v2.1` (best quality/resource on RTX 3060)
- Model A: fine-tuned on CoNLL-03 train split
- Model B: fine-tuned on CleanCoNLL train split
- Hyperparams: lr=1e-5, batch_size=8, max_steps=2000, seed=42
- Save to `models/finetuned_conll03/` and `models/finetuned_cleanconll/`

**Evaluation:**
- Evaluate both models on CleanCoNLL test set
- Use existing evaluation pipeline
- Save to `results/finetune/`

**Output:** `results/finetune_table.md` — training data vs clean eval F1.

## Updated run_all.sh

```bash
bash run_all.sh              # basic pipeline (same as before)
bash run_all.sh --full       # basic + bootstrap + ablation + finetune
bash run_all.sh --bootstrap  # basic + bootstrap only
bash run_all.sh --ablation   # basic + ablation only
bash run_all.sh --finetune   # basic + finetune only
```

## New Dependencies

- `gliner[training]` (for fine-tuning, adds accelerate etc.)
- No other new dependencies (bootstrap uses numpy which is already required)

## Updated requirements.txt

Add: `accelerate>=0.20.0` (needed by gliner training)
