"""Baseline NER wrappers for non-GLiNER models.

Each baseline exposes ``predict_sentence(model_bundle, tokens) -> list[dict]``
matching the prediction format used by the GLiNER pipeline:

    {"text": str, "start_token": int, "end_token": int,
     "label": str, "score": float}

This lets ``scripts/evaluate.py`` and the ablation infrastructure consume
baseline predictions without any special handling.
"""

from typing import Any

from src.inference import map_char_spans_to_token_spans, tokens_to_text_with_offsets


# HuggingFace token-classification models like ``dslim/bert-base-NER`` return
# labels such as ``B-PER`` with the CoNLL short-form after ``aggregation_strategy``.
# Since that already matches our canonical labels (PER/ORG/LOC/MISC) we don't
# rebuild a mapping — this is here as a hook if other baselines need it.
def map_hf_label_to_conll(label: str) -> str:
    # aggregation_strategy="simple" removes the B-/I- prefix; handle both cases.
    if label.startswith(("B-", "I-")):
        label = label[2:]
    return label


def load_hf_ner_pipeline(model_name: str, device: int | str | None = None) -> Any:
    """Load a HuggingFace token-classification pipeline with span aggregation."""
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    # aggregation_strategy="simple" merges sub-word pieces into entity spans
    # and returns character offsets, which we then map back to token indices.
    return pipeline(
        task="ner",
        model=model_name,
        aggregation_strategy="simple",
        device=device,
    )


def predict_sentence_hf(
    nlp: Any,
    tokens: list[str],
    allowed_labels: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Run a HF NER pipeline on a whitespace-joined sentence.

    Character offsets from the pipeline are mapped back to token indices using
    the same logic as the GLiNER wrapper. Predictions whose mapped label is not
    in ``allowed_labels`` (when provided) are dropped; this keeps the set of
    evaluated labels aligned with CoNLL's PER/ORG/LOC/MISC.
    """
    text, token_offsets = tokens_to_text_with_offsets(tokens)
    raw = nlp(text)

    results: list[dict[str, Any]] = []
    for ent in raw:
        label = map_hf_label_to_conll(ent.get("entity_group", ent.get("entity", "")))
        if allowed_labels is not None and label not in allowed_labels:
            continue
        start_tok, end_tok = map_char_spans_to_token_spans(
            int(ent["start"]), int(ent["end"]), token_offsets
        )
        results.append(
            {
                "text": str(ent.get("word", text[int(ent["start"]):int(ent["end"])])),
                "start_token": start_tok,
                "end_token": end_tok,
                "label": label,
                "score": float(ent.get("score", 0.0)),
            }
        )
    return results
