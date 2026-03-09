# Fair Evaluation of Open-Type NER with GLiNER on CleanCoNLL

## Overview

This project evaluates [GLiNER](https://github.com/urchade/GLiNER), a generalist model for named entity recognition, on both the original CoNLL-2003 dataset and the [CleanCoNLL](https://github.com/flairNLP/CleanCoNLL) revision. By comparing performance across the two annotation sets, we measure how label noise in the original CoNLL-2003 gold standard affects reported scores and whether cleaning the annotations changes the ranking or perceived quality of an open-type NER system.

## Prerequisites

- Python 3.10 or higher
- Git

A CUDA-capable GPU is strongly recommended for running GLiNER inference at reasonable speed.

## Setup

1. **Clone this repository**

   ```bash
   git clone https://github.com/rusydi16/PBML_Open-Type_NER_w_GLiNER_on_CleanCoNLL.git
   cd PBML_Open-Type_NER_w_GLiNER_on_CleanCoNLL
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/Mac
   # .venv\Scripts\activate    # Windows
   pip install -r requirements.txt
   ```

3. **Place CoNLL-2003 data files**

   Copy the original CoNLL-2003 files into `data/raw/`:

   ```
   data/raw/eng.train
   data/raw/eng.testa
   data/raw/eng.testb
   ```

   These files are not distributed with this repository. You must obtain them from the original CoNLL-2003 shared task.

## Building CleanCoNLL

The `prepare_data.py` script clones the official CleanCoNLL repository into `data/cleanconll_repo/`. After that, you need to run their build script manually to produce the cleaned dataset:

```bash
cd data/cleanconll_repo

# Copy your CoNLL-2003 files to the location expected by CleanCoNLL.
# Check their README for the exact expected paths, typically:
cp ../raw/eng.train .
cp ../raw/eng.testa .
cp ../raw/eng.testb .

# Run the CleanCoNLL build script
bash create_cleanconll_from_conll03.sh

# Return to the project root
cd ../..
```

Once the build script finishes, `prepare_data.py` will be able to read and align both the original and cleaned annotations.

## Running the Pipeline

### Basic Pipeline

Run the core evaluation pipeline (data prep, inference, evaluation, report):

```bash
bash run_all.sh
```

### Full Pipeline (with all enhancements)

Run everything including bootstrap significance testing, model size ablation, and fine-tuning comparison:

```bash
bash run_all.sh --full
```

### Individual Enhancement Flags

```bash
bash run_all.sh --bootstrap   # Add statistical significance testing
bash run_all.sh --ablation    # Add model size ablation study
bash run_all.sh --finetune    # Add fine-tuning comparison
```

Flags can be combined: `bash run_all.sh --bootstrap --ablation`

## Individual Scripts

Each stage can also be run on its own:

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/prepare_data.py` | Loads and aligns CoNLL-2003 and CleanCoNLL data into a unified format under `data/processed/`. | `python scripts/prepare_data.py --config configs/default.yaml` |
| `scripts/run_inference.py` | Runs GLiNER inference on the processed data and writes predictions. | `python scripts/run_inference.py --config configs/default.yaml` |
| `scripts/evaluate.py` | Computes precision, recall, F1, error classification, and noise attribution. Add `--bootstrap` for significance testing. | `python scripts/evaluate.py --config configs/default.yaml --bootstrap` |
| `scripts/generate_report.py` | Produces summary tables and a findings report from the evaluation results. | `python scripts/generate_report.py --config configs/default.yaml` |
| `scripts/run_ablation.py` | Runs inference and evaluation across small/medium/large GLiNER variants. | `python scripts/run_ablation.py --config configs/ablation.yaml` |
| `scripts/run_finetune.py` | Fine-tunes GLiNER on CoNLL-03 vs CleanCoNLL training data, evaluates both on CleanCoNLL test. | `python scripts/run_finetune.py --config configs/finetune.yaml` |

All scripts accept `--config <path>` to override the default configuration.

## Output Files

After a successful pipeline run, the `results/` directory will contain:

**Core outputs:**
- `metrics_conll03_test.json` / `metrics_cleanconll_test.json` — P/R/F1 and error counts
- `noise_analysis_test.json` — Noise attribution analysis
- `comparison_table.csv` / `comparison_table.md` — Side-by-side comparison
- `findings.md` — Summary report

**Bootstrap (with `--bootstrap`):**
- `bootstrap_conll03_test.json` / `bootstrap_cleanconll_test.json` — F1 confidence intervals
- `significance_test_test.json` — Paired bootstrap test results with p-value

**Ablation (with `--ablation`):**
- `ablation/small/`, `ablation/medium/`, `ablation/large/` — Per-model predictions and metrics
- `ablation_table.csv` / `ablation_table.md` — Model size comparison

**Fine-tuning (with `--finetune`):**
- `finetune/metrics_conll03_test.json` / `finetune/metrics_cleanconll_test.json` — Fine-tuned model metrics
- `finetune/finetune_table.csv` / `finetune/finetune_table.md` — Training data comparison

## Project Structure

```
.
├── configs/
│   ├── default.yaml          # Default pipeline configuration
│   ├── ablation.yaml         # Model size ablation configuration
│   └── finetune.yaml         # Fine-tuning configuration
├── data/
│   ├── raw/                  # Original CoNLL-2003 files (user-provided)
│   ├── processed/            # Aligned data produced by prepare_data.py
│   └── cleanconll_repo/      # Cloned CleanCoNLL repository
├── models/                   # Fine-tuned models (generated)
├── results/                  # Pipeline outputs (metrics, reports)
├── scripts/
│   ├── prepare_data.py       # Stage 1: data preparation
│   ├── run_inference.py      # Stage 2: GLiNER inference
│   ├── evaluate.py           # Stage 3: metric computation + bootstrap
│   ├── generate_report.py    # Stage 4: report generation
│   ├── run_ablation.py       # Model size ablation study
│   └── run_finetune.py       # Fine-tuning comparison
├── src/
│   ├── __init__.py
│   ├── data_utils.py         # Data loading and alignment utilities
│   ├── inference.py          # GLiNER wrapper
│   ├── metrics.py            # Evaluation metric implementations
│   ├── noise_analysis.py     # Label noise comparison utilities
│   ├── statistical_tests.py  # Bootstrap significance testing
│   └── finetune.py           # Fine-tuning data conversion and training
├── tests/
│   ├── __init__.py
│   ├── test_data_utils.py
│   ├── test_inference.py
│   ├── test_metrics.py
│   ├── test_noise_analysis.py
│   ├── test_statistical_tests.py
│   └── test_finetune.py
├── requirements.txt
├── run_all.sh                # Full pipeline runner
└── README.md
```

## Configuration

All pipeline behaviour is controlled through YAML config files:

### `configs/default.yaml` (core pipeline)

| Field | Description |
|-------|-------------|
| `model.name` | HuggingFace model identifier for GLiNER (default: `knowledgator/gliner-multitask-large-v0.5`) |
| `model.threshold` | Confidence threshold for entity predictions (default: `0.5`) |
| `labels` | Mapping between GLiNER open-type labels and CoNLL entity labels (PER, ORG, LOC, MISC) |
| `paths.*` | Directories for raw data, processed data, results, CleanCoNLL repo |
| `seed` | Random seed for reproducibility |

### `configs/ablation.yaml` (model size ablation)

| Field | Description |
|-------|-------------|
| `models` | List of GLiNER model variants with name, short_name, and param count |
| `threshold` | Shared confidence threshold |

### `configs/finetune.yaml` (fine-tuning)

| Field | Description |
|-------|-------------|
| `base_model` | Base model to fine-tune (default: `urchade/gliner_medium-v2.1`) |
| `training.*` | max_steps, learning_rate, batch_size, warmup_ratio |
| `paths.models` | Directory to save fine-tuned models |

## Models

- **Core pipeline:** `knowledgator/gliner-multitask-large-v0.5` — latest multitask NER model
- **Ablation:** `urchade/gliner_small-v2.1` (44M), `urchade/gliner_medium-v2.1` (86M), `urchade/gliner_large-v2.1` (304M)
- **Fine-tuning base:** `urchade/gliner_medium-v2.1` — best quality-to-resource tradeoff for RTX 3060

All models perform open-type NER, accepting arbitrary entity type labels at inference time. A GPU is recommended; inference on CPU is possible but significantly slower.
