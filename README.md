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
   git clone <repo-url>
   cd Ver_1
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
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

Run the full pipeline end-to-end with the default configuration:

```bash
bash run_all.sh
```

Or specify a custom config file:

```bash
bash run_all.sh configs/my_config.yaml
```

The pipeline executes four stages in order: data preparation, GLiNER inference, evaluation, and report generation.

## Individual Scripts

Each stage can also be run on its own:

| Script | Description | Example |
|--------|-------------|---------|
| `scripts/prepare_data.py` | Loads and aligns CoNLL-2003 and CleanCoNLL data into a unified format under `data/processed/`. | `python scripts/prepare_data.py --config configs/default.yaml` |
| `scripts/run_inference.py` | Runs GLiNER inference on the processed data and writes predictions. | `python scripts/run_inference.py --config configs/default.yaml` |
| `scripts/evaluate.py` | Computes precision, recall, F1 and other metrics against both annotation sets. | `python scripts/evaluate.py --config configs/default.yaml` |
| `scripts/generate_report.py` | Produces summary tables and a final report from the evaluation results. | `python scripts/generate_report.py --config configs/default.yaml` |

All scripts accept `--config <path>` to override the default configuration.

## Output Files

After a successful pipeline run, the `results/` directory will contain:

- Evaluation metrics (precision, recall, F1) for both CoNLL-2003 and CleanCoNLL annotations
- Per-entity-type breakdowns
- Noise analysis comparing score differences between the original and cleaned datasets
- A summary report

## Project Structure

```
Ver_1/
├── configs/
│   └── default.yaml          # Default pipeline configuration
├── data/
│   ├── raw/                   # Original CoNLL-2003 files (user-provided)
│   ├── processed/             # Aligned data produced by prepare_data.py
│   └── cleanconll_repo/       # Cloned CleanCoNLL repository
├── results/                   # Pipeline outputs (metrics, reports)
├── scripts/
│   ├── prepare_data.py        # Stage 1: data preparation
│   ├── run_inference.py       # Stage 2: GLiNER inference
│   ├── evaluate.py            # Stage 3: metric computation
│   └── generate_report.py     # Stage 4: report generation
├── src/
│   ├── __init__.py
│   ├── data_utils.py          # Data loading and alignment utilities
│   ├── inference.py           # GLiNER wrapper
│   ├── metrics.py             # Evaluation metric implementations
│   └── noise_analysis.py      # Label noise comparison utilities
├── tests/
│   ├── __init__.py
│   ├── test_data_utils.py
│   ├── test_inference.py
│   ├── test_metrics.py
│   └── test_noise_analysis.py
├── requirements.txt
├── run_all.sh                 # Full pipeline runner
└── README.md
```

## Configuration

All pipeline behaviour is controlled through a YAML config file. The default is `configs/default.yaml`:

| Field | Description |
|-------|-------------|
| `model.name` | HuggingFace model identifier for GLiNER (default: `knowledgator/gliner-multitask-large-v0.5`) |
| `model.threshold` | Confidence threshold for entity predictions (default: `0.5`) |
| `labels` | Mapping between GLiNER open-type labels and CoNLL entity labels (PER, ORG, LOC, MISC) |
| `paths.raw_data` | Directory containing original CoNLL-2003 files |
| `paths.processed_data` | Directory for processed/aligned data |
| `paths.results` | Directory for evaluation outputs |
| `paths.cleanconll_repo` | Directory where the CleanCoNLL repo is cloned |
| `seed` | Random seed for reproducibility |

## Model

This project uses **gliner-multitask-large-v0.5** by Knowledgator. The model performs open-type NER, meaning it accepts arbitrary entity type labels as input at inference time rather than being restricted to a fixed label set. A GPU is recommended; inference on CPU is possible but significantly slower.
