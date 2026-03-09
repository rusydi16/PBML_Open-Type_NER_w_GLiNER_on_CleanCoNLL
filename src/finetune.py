"""Fine-tuning utilities: data format conversion and GLiNER training wrapper."""

import json
from typing import Any


def convert_sentence_to_gliner_format(
    sentence: dict[str, Any],
    conll_to_gliner: dict[str, str],
) -> dict[str, Any]:
    """Convert one sentence from our format to GLiNER format.

    Input sentence has keys ``tokens`` and ``entities`` where each entity has
    ``start`` (inclusive) and ``end`` (exclusive) token indices plus a ``label``
    using CoNLL tag names (e.g. ``PER``).

    Returns a dict with ``tokenized_text`` (list of tokens) and ``ner``
    (list of ``[start, end_inclusive, gliner_label]`` triples).
    """
    ner: list[list[Any]] = []
    for entity in sentence["entities"]:
        start = entity["start"]
        end_inclusive = entity["end"] - 1
        gliner_label = conll_to_gliner[entity["label"]]
        ner.append([start, end_inclusive, gliner_label])
    return {
        "tokenized_text": sentence["tokens"],
        "ner": ner,
    }


def convert_dataset_to_gliner_format(
    sentences: list[dict[str, Any]],
    conll_to_gliner: dict[str, str],
) -> list[dict[str, Any]]:
    """Convert a list of sentences to GLiNER format."""
    return [
        convert_sentence_to_gliner_format(s, conll_to_gliner)
        for s in sentences
    ]


def save_gliner_training_data(data: list[dict[str, Any]], filepath: str) -> None:
    """Save GLiNER-formatted training data as JSON."""
    with open(filepath, "w") as f:
        json.dump(data, f)


def finetune_gliner(
    model_name: str,
    train_data: list[dict[str, Any]],
    output_dir: str,
    max_steps: int = 2000,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    seed: int = 42,
    eval_data: list[dict[str, Any]] | None = None,
) -> Any:
    """Fine-tune a GLiNER model.

    Imports ``gliner`` inside the function to avoid loading the heavy
    dependency at module import time.

    Returns the trained model.
    """
    from gliner import GLiNER  # type: ignore[import-untyped]

    model = GLiNER.from_pretrained(model_name)

    train_params = {
        "num_steps": max_steps,
        "train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "save_directory": output_dir,
    }
    if eval_data is not None:
        train_params["val_data_dir"] = eval_data

    model.train_model(train_data, **train_params)
    return model
