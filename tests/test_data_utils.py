"""Tests for src/data_utils.py."""

import json
import os
import tempfile

import pytest

from src.data_utils import (
    bio_tags_to_entities,
    load_sentences_json,
    parse_conll03_file,
    parse_cleanconll_file,
    save_sentences_json,
)


# ---------------------------------------------------------------------------
# bio_tags_to_entities
# ---------------------------------------------------------------------------

class TestBioTagsToEntities:
    def test_simple_entity(self):
        tokens = ["John", "lives", "in", "New", "York"]
        tags = ["B-PER", "O", "O", "B-LOC", "I-LOC"]
        entities = bio_tags_to_entities(tokens, tags)
        assert entities == [
            {"text": "John", "start": 0, "end": 1, "label": "PER"},
            {"text": "New York", "start": 3, "end": 5, "label": "LOC"},
        ]

    def test_all_o_tags(self):
        tokens = ["the", "cat", "sat"]
        tags = ["O", "O", "O"]
        entities = bio_tags_to_entities(tokens, tags)
        assert entities == []

    def test_consecutive_b_tags(self):
        tokens = ["John", "Mary", "ran"]
        tags = ["B-PER", "B-PER", "O"]
        entities = bio_tags_to_entities(tokens, tags)
        assert entities == [
            {"text": "John", "start": 0, "end": 1, "label": "PER"},
            {"text": "Mary", "start": 1, "end": 2, "label": "PER"},
        ]

    def test_i_tag_different_label_starts_new_entity(self):
        tokens = ["John", "Inc"]
        tags = ["B-PER", "I-ORG"]
        entities = bio_tags_to_entities(tokens, tags)
        assert entities == [
            {"text": "John", "start": 0, "end": 1, "label": "PER"},
            {"text": "Inc", "start": 1, "end": 2, "label": "ORG"},
        ]

    def test_empty_input(self):
        assert bio_tags_to_entities([], []) == []


# ---------------------------------------------------------------------------
# parse_conll03_file
# ---------------------------------------------------------------------------

SAMPLE_CONLL03 = """\
-DOCSTART- -X- -X- O

EU NNP I-NP B-ORG
rejects VBZ I-VP O
German JJ I-NP B-MISC
call NN I-NP O

Peter NNP I-NP B-PER
Blackburn NNP I-NP I-PER
"""


class TestParseConll03File:
    def test_parse_conll03(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(SAMPLE_CONLL03)
            f.flush()
            tmp_path = f.name

        try:
            sentences = parse_conll03_file(tmp_path, "train")
            assert len(sentences) == 2

            s0 = sentences[0]
            assert s0["id"] == "train-0001"
            assert s0["tokens"] == ["EU", "rejects", "German", "call"]
            assert s0["ner_tags"] == ["B-ORG", "O", "B-MISC", "O"]
            assert {"text": "EU", "start": 0, "end": 1, "label": "ORG"} in s0["entities"]
            assert {"text": "German", "start": 2, "end": 3, "label": "MISC"} in s0["entities"]

            s1 = sentences[1]
            assert s1["id"] == "train-0002"
            assert s1["tokens"] == ["Peter", "Blackburn"]
            assert s1["ner_tags"] == ["B-PER", "I-PER"]
            assert s1["entities"] == [
                {"text": "Peter Blackburn", "start": 0, "end": 2, "label": "PER"}
            ]
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# parse_cleanconll_file
# ---------------------------------------------------------------------------

SAMPLE_CLEANCONLL = """\
-DOCSTART-\t-X-\t-X-\t-X-\tO

EU\tNNP\tQ458\tB-ORG\tB-ORG
rejects\tVBZ\t-\tO\tO

"""


class TestParseCleanConllFile:
    def test_parse_cleanconll(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(SAMPLE_CLEANCONLL)
            f.flush()
            tmp_path = f.name

        try:
            sentences = parse_cleanconll_file(tmp_path, "test")
            assert len(sentences) == 1
            s0 = sentences[0]
            assert s0["id"] == "test-0001"
            assert s0["tokens"] == ["EU", "rejects"]
            assert s0["ner_tags"] == ["B-ORG", "O"]
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# save / load JSON round-trip
# ---------------------------------------------------------------------------

class TestJsonRoundTrip:
    def test_save_and_load(self):
        sentences = [
            {"id": "train-0001", "tokens": ["a"], "ner_tags": ["O"], "entities": []}
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name

        try:
            save_sentences_json(sentences, tmp_path)
            loaded = load_sentences_json(tmp_path)
            assert loaded == sentences
        finally:
            os.unlink(tmp_path)
