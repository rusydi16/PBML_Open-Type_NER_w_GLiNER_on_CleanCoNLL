"""Inference utilities for GLiNER-based NER."""

import random
from typing import Any, Dict, List, Optional, Tuple


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility (random, numpy, torch, cuda)."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def tokens_to_text_with_offsets(
    tokens: List[str],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Join tokens with spaces and return (text, list of (char_start, char_end) per token)."""
    offsets: List[Tuple[int, int]] = []
    pos = 0
    for token in tokens:
        start = pos
        end = pos + len(token)
        offsets.append((start, end))
        pos = end + 1  # +1 for the space separator
    text = " ".join(tokens)
    return text, offsets


def map_char_spans_to_token_spans(
    char_start: int,
    char_end: int,
    token_offsets: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """Map a character-level span to token indices.

    Returns (start_token, end_token) where end_token is exclusive.
    """
    start_token: Optional[int] = None
    end_token: Optional[int] = None

    for i, (tok_start, tok_end) in enumerate(token_offsets):
        # Token overlaps with the character span
        if tok_end > char_start and tok_start < char_end:
            if start_token is None:
                start_token = i
            end_token = i + 1

    if start_token is None:
        return (0, 0)
    return (start_token, end_token)


def map_gliner_label_to_conll(gliner_label: str, label_map: Dict[str, str]) -> str:
    """Map a GLiNER label to a CoNLL label using a dict. Returns original if not found."""
    return label_map.get(gliner_label, gliner_label)


def load_gliner_model(model_name: str, device: str | None = None) -> Any:
    """Load a GLiNER model and move it to GPU if CUDA is available.

    GLiNER.from_pretrained returns a CPU-resident model by default. Without an
    explicit .to('cuda') call, inference stays on CPU even when a GPU is
    available — several-fold slower. Pass ``device`` to override.
    """
    import torch
    from gliner import GLiNER

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GLiNER.from_pretrained(model_name)
    model = model.to(device)
    print(f"  Model placed on device: {device}")
    return model


def predict_sentence(
    model: Any,
    tokens: List[str],
    gliner_labels: List[str],
    label_map: Dict[str, str],
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Run GLiNER inference on a sentence.

    Returns list of dicts with keys: text, start_token, end_token, label, score.
    When ``label_map`` collapses multiple GLiNER prompts into the same CoNLL
    label (e.g. ``"country"``, ``"city"`` → ``LOC``) or when different prompts
    yield conflicting labels on the same span (e.g. ``"country: Japan"`` vs
    ``"sports team: Japan"``), only the highest-scoring prediction per
    ``(start_token, end_token)`` span is kept so the evaluator isn't
    double-counting.
    """
    import torch

    text, token_offsets = tokens_to_text_with_offsets(tokens)

    with torch.no_grad():
        entities = model.predict_entities(
            text, gliner_labels, threshold=threshold
        )

    by_span: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for ent in entities:
        char_start = ent["start"]
        char_end = ent["end"]
        start_tok, end_tok = map_char_spans_to_token_spans(
            char_start, char_end, token_offsets
        )
        mapped_label = map_gliner_label_to_conll(ent["label"], label_map)
        candidate = {
            "text": ent["text"],
            "start_token": start_tok,
            "end_token": end_tok,
            "label": mapped_label,
            "score": ent["score"],
        }
        key = (start_tok, end_tok)
        existing = by_span.get(key)
        if existing is None or candidate["score"] > existing["score"]:
            by_span[key] = candidate

    return list(by_span.values())
