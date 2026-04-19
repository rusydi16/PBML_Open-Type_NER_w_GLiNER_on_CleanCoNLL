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
    eval_data: list[dict[str, Any]],
    output_dir: str,
    max_steps: int = 2000,
    learning_rate: float = 1e-5,
    batch_size: int = 8,
    seed: int = 42,
    save_steps: int | None = None,
    save_total_limit: int = 3,
    warmup_ratio: float = 0.1,
) -> Any:
    """Fine-tune a GLiNER model using the 0.2+ API.

    GLiNER 0.2.26 routes training through a Hugging Face Trainer; both
    ``train_dataset`` and ``eval_dataset`` are required positional arguments,
    and hyperparameters are passed as ``**training_kwargs`` that
    ``BaseGLiNER.create_training_args`` forwards to ``TrainingArguments``.

    Imports ``gliner`` inside the function to keep it out of module import.
    Returns the trained model so the caller can run inference or save it.
    """
    from gliner import GLiNER  # type: ignore[import-untyped]

    model = GLiNER.from_pretrained(model_name)

    # Auto-derive a reasonable checkpoint cadence so a mid-training crash still
    # leaves a usable snapshot on disk.
    if save_steps is None:
        save_steps = max(max_steps // 4, 250) if max_steps >= 500 else max_steps

    model.train_model(
        train_dataset=train_data,
        eval_dataset=eval_data,
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=max_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        seed=seed,
    )
    return model
