"""Tests for src/finetune.py."""

import json
import os
import tempfile

import pytest

from src.finetune import (
    convert_sentence_to_gliner_format,
    convert_dataset_to_gliner_format,
    save_gliner_training_data,
)


CONLL_TO_GLINER = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
    "MISC": "miscellaneous",
}


# ---------------------------------------------------------------------------
# convert_sentence_to_gliner_format
# ---------------------------------------------------------------------------

class TestConvertSentenceToGlinerFormat:
    def test_basic_conversion(self):
        sentence = {
            "tokens": ["John", "lives", "in", "New", "York"],
            "entities": [
                {"start": 0, "end": 1, "label": "PER"},
                {"start": 3, "end": 5, "label": "LOC"},
            ],
        }
        result = convert_sentence_to_gliner_format(sentence, CONLL_TO_GLINER)
        assert result["tokenized_text"] == ["John", "lives", "in", "New", "York"]
        assert result["ner"] == [
            [0, 0, "person"],
            [3, 4, "location"],
        ]

    def test_empty_entities(self):
        sentence = {
            "tokens": ["the", "cat", "sat"],
            "entities": [],
        }
        result = convert_sentence_to_gliner_format(sentence, CONLL_TO_GLINER)
        assert result["tokenized_text"] == ["the", "cat", "sat"]
        assert result["ner"] == []

    def test_exclusive_to_inclusive_end(self):
        """Entity end index is exclusive in our format, inclusive in GLiNER."""
        sentence = {
            "tokens": ["The", "European", "Union", "met"],
            "entities": [
                {"start": 1, "end": 3, "label": "ORG"},
            ],
        }
        result = convert_sentence_to_gliner_format(sentence, CONLL_TO_GLINER)
        # end=3 exclusive -> end=2 inclusive
        assert result["ner"] == [[1, 2, "organization"]]

    def test_single_token_entity(self):
        """A single-token entity with end=start+1 should have inclusive end=start."""
        sentence = {
            "tokens": ["Alice", "ran"],
            "entities": [
                {"start": 0, "end": 1, "label": "PER"},
            ],
        }
        result = convert_sentence_to_gliner_format(sentence, CONLL_TO_GLINER)
        assert result["ner"] == [[0, 0, "person"]]


# ---------------------------------------------------------------------------
# convert_dataset_to_gliner_format
# ---------------------------------------------------------------------------

class TestConvertDatasetToGlinerFormat:
    def test_multiple_sentences(self):
        sentences = [
            {
                "tokens": ["John", "ran"],
                "entities": [{"start": 0, "end": 1, "label": "PER"}],
            },
            {
                "tokens": ["in", "Paris"],
                "entities": [{"start": 1, "end": 2, "label": "LOC"}],
            },
        ]
        results = convert_dataset_to_gliner_format(sentences, CONLL_TO_GLINER)
        assert len(results) == 2
        assert results[0]["tokenized_text"] == ["John", "ran"]
        assert results[0]["ner"] == [[0, 0, "person"]]
        assert results[1]["tokenized_text"] == ["in", "Paris"]
        assert results[1]["ner"] == [[1, 1, "location"]]

    def test_empty_dataset(self):
        results = convert_dataset_to_gliner_format([], CONLL_TO_GLINER)
        assert results == []


# ---------------------------------------------------------------------------
# save_gliner_training_data
# ---------------------------------------------------------------------------

class TestSaveGlinerTrainingData:
    def test_save_and_load(self):
        data = [
            {"tokenized_text": ["a", "b"], "ner": [[0, 0, "person"]]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.json")
            save_gliner_training_data(data, filepath)
            with open(filepath, "r") as f:
                loaded = json.load(f)
            assert loaded == data
