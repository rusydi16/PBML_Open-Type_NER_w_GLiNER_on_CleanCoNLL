"""Tests for src/inference.py (pure utility functions only, no gliner required)."""

import pytest

from src.inference import (
    map_char_spans_to_token_spans,
    map_gliner_label_to_conll,
    tokens_to_text_with_offsets,
)


# ---------------------------------------------------------------------------
# tokens_to_text_with_offsets
# ---------------------------------------------------------------------------

class TestTokensToTextWithOffsets:
    def test_single_token(self):
        text, offsets = tokens_to_text_with_offsets(["hello"])
        assert text == "hello"
        assert offsets == [(0, 5)]

    def test_multiple_tokens(self):
        text, offsets = tokens_to_text_with_offsets(["EU", "rejects", "German", "call"])
        assert text == "EU rejects German call"
        assert len(offsets) == 4
        assert offsets[0] == (0, 2)
        assert offsets[1] == (3, 10)
        assert offsets[2] == (11, 17)
        assert offsets[3] == (18, 22)

    def test_empty_list(self):
        text, offsets = tokens_to_text_with_offsets([])
        assert text == ""
        assert offsets == []

    def test_offsets_cover_correct_substrings(self):
        tokens = ["The", "quick", "brown", "fox"]
        text, offsets = tokens_to_text_with_offsets(tokens)
        for token, (start, end) in zip(tokens, offsets):
            assert text[start:end] == token


# ---------------------------------------------------------------------------
# map_char_spans_to_token_spans
# ---------------------------------------------------------------------------

class TestMapCharSpansToTokenSpans:
    def test_exact_single_token(self):
        offsets = [(0, 2), (3, 10), (11, 17), (18, 22)]
        # "EU" starts at 0, ends at 2
        start, end = map_char_spans_to_token_spans(0, 2, offsets)
        assert start == 0
        assert end == 1

    def test_exact_multi_token(self):
        offsets = [(0, 2), (3, 10), (11, 17), (18, 22)]
        # "German call" starts at 11, ends at 22
        start, end = map_char_spans_to_token_spans(11, 22, offsets)
        assert start == 2
        assert end == 4

    def test_partial_overlap(self):
        offsets = [(0, 3), (4, 9), (10, 15)]
        # A span that starts inside first token and ends inside second
        start, end = map_char_spans_to_token_spans(1, 7, offsets)
        assert start == 0
        assert end == 2

    def test_last_token(self):
        offsets = [(0, 3), (4, 7)]
        start, end = map_char_spans_to_token_spans(4, 7, offsets)
        assert start == 1
        assert end == 2


# ---------------------------------------------------------------------------
# map_gliner_label_to_conll
# ---------------------------------------------------------------------------

class TestMapGlinerLabelToConll:
    def test_known_label(self):
        label_map = {"person": "PER", "location": "LOC"}
        assert map_gliner_label_to_conll("person", label_map) == "PER"

    def test_unknown_label_returns_original(self):
        label_map = {"person": "PER"}
        assert map_gliner_label_to_conll("organization", label_map) == "organization"

    def test_empty_map(self):
        assert map_gliner_label_to_conll("person", {}) == "person"
