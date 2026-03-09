"""Tests for src/metrics.py."""

import pytest

from src.metrics import (
    _normalize_entity,
    _spans_overlap,
    classify_errors,
    compute_entity_metrics,
    compute_per_type_metrics,
)


# ---------------------------------------------------------------------------
# _normalize_entity
# ---------------------------------------------------------------------------

class TestNormalizeEntity:
    def test_gold_format(self):
        entity = {"start": 0, "end": 3, "label": "PER"}
        assert _normalize_entity(entity) == (0, 3, "PER")

    def test_pred_format(self):
        entity = {"start_token": 1, "end_token": 4, "label": "LOC",
                  "text": "New York", "score": 0.95}
        assert _normalize_entity(entity) == (1, 4, "LOC")


# ---------------------------------------------------------------------------
# _spans_overlap
# ---------------------------------------------------------------------------

class TestSpansOverlap:
    def test_overlapping(self):
        assert _spans_overlap(0, 3, 2, 5) is True

    def test_non_overlapping(self):
        assert _spans_overlap(0, 2, 3, 5) is False

    def test_adjacent_no_overlap(self):
        assert _spans_overlap(0, 2, 2, 4) is False

    def test_contained(self):
        assert _spans_overlap(1, 5, 2, 4) is True


# ---------------------------------------------------------------------------
# compute_entity_metrics
# ---------------------------------------------------------------------------

class TestComputeEntityMetrics:
    def test_perfect_match(self):
        gold = [
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 3, "end": 5, "label": "LOC"},
        ]
        pred = [
            {"start_token": 0, "end_token": 1, "label": "PER", "text": "John", "score": 0.9},
            {"start_token": 3, "end_token": 5, "label": "LOC", "text": "New York", "score": 0.8},
        ]
        m = compute_entity_metrics(gold, pred)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0
        assert m["tp"] == 2
        assert m["fp"] == 0
        assert m["fn"] == 0

    def test_no_predictions(self):
        gold = [
            {"start": 0, "end": 1, "label": "PER"},
        ]
        pred = []
        m = compute_entity_metrics(gold, pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["tp"] == 0
        assert m["fp"] == 0
        assert m["fn"] == 1

    def test_no_gold(self):
        gold = []
        pred = [
            {"start_token": 0, "end_token": 1, "label": "PER", "text": "X", "score": 0.5},
        ]
        m = compute_entity_metrics(gold, pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["tp"] == 0
        assert m["fp"] == 1
        assert m["fn"] == 0

    def test_wrong_label(self):
        gold = [{"start": 0, "end": 2, "label": "PER"}]
        pred = [{"start_token": 0, "end_token": 2, "label": "LOC", "text": "X", "score": 0.5}]
        m = compute_entity_metrics(gold, pred)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0
        assert m["tp"] == 0
        assert m["fp"] == 1
        assert m["fn"] == 1

    def test_partial_match(self):
        gold = [
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 3, "end": 5, "label": "LOC"},
        ]
        pred = [
            {"start_token": 0, "end_token": 1, "label": "PER", "text": "John", "score": 0.9},
        ]
        m = compute_entity_metrics(gold, pred)
        assert m["precision"] == 1.0
        assert m["recall"] == 0.5
        assert m["tp"] == 1
        assert m["fp"] == 0
        assert m["fn"] == 1

    def test_both_empty(self):
        m = compute_entity_metrics([], [])
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0


# ---------------------------------------------------------------------------
# compute_per_type_metrics
# ---------------------------------------------------------------------------

class TestComputePerTypeMetrics:
    def test_per_type_breakdown(self):
        gold = [
            {"start": 0, "end": 1, "label": "PER"},
            {"start": 3, "end": 5, "label": "LOC"},
        ]
        pred = [
            {"start_token": 0, "end_token": 1, "label": "PER", "text": "John", "score": 0.9},
        ]
        result = compute_per_type_metrics(gold, pred, ["PER", "LOC"])
        assert result["PER"]["f1"] == 1.0
        assert result["LOC"]["recall"] == 0.0
        assert result["LOC"]["precision"] == 0.0


# ---------------------------------------------------------------------------
# classify_errors
# ---------------------------------------------------------------------------

class TestClassifyErrors:
    def test_type_error(self):
        """Span matches exactly but label differs."""
        gold = [{"start": 0, "end": 2, "label": "PER"}]
        pred = [{"start_token": 0, "end_token": 2, "label": "LOC", "text": "X", "score": 0.5}]
        errors = classify_errors(gold, pred)
        assert errors["type_error"] == 1
        assert errors["boundary_error"] == 0
        assert errors["missing"] == 0
        assert errors["spurious"] == 0

    def test_boundary_error(self):
        """Spans overlap, different boundaries, same label."""
        gold = [{"start": 0, "end": 3, "label": "PER"}]
        pred = [{"start_token": 1, "end_token": 3, "label": "PER", "text": "X", "score": 0.5}]
        errors = classify_errors(gold, pred)
        assert errors["boundary_error"] == 1
        assert errors["type_error"] == 0

    def test_type_boundary_error(self):
        """Spans overlap, different boundaries AND label."""
        gold = [{"start": 0, "end": 3, "label": "PER"}]
        pred = [{"start_token": 1, "end_token": 3, "label": "LOC", "text": "X", "score": 0.5}]
        errors = classify_errors(gold, pred)
        assert errors["type_boundary_error"] == 1

    def test_missing(self):
        """Gold entity with no overlapping prediction."""
        gold = [{"start": 0, "end": 2, "label": "PER"}]
        pred = [{"start_token": 5, "end_token": 7, "label": "LOC", "text": "X", "score": 0.5}]
        errors = classify_errors(gold, pred)
        assert errors["missing"] == 1
        assert errors["spurious"] == 1

    def test_spurious(self):
        """Prediction with no overlapping gold."""
        gold = []
        pred = [{"start_token": 0, "end_token": 2, "label": "PER", "text": "X", "score": 0.5}]
        errors = classify_errors(gold, pred)
        assert errors["spurious"] == 1
        assert errors["missing"] == 0

    def test_mixed_errors(self):
        """Combination of different error types."""
        gold = [
            {"start": 0, "end": 2, "label": "PER"},   # type error (matched with pred[0])
            {"start": 5, "end": 8, "label": "LOC"},   # missing (no overlap)
        ]
        pred = [
            {"start_token": 0, "end_token": 2, "label": "ORG", "text": "X", "score": 0.5},  # type error
            {"start_token": 10, "end_token": 12, "label": "PER", "text": "Y", "score": 0.5},  # spurious
        ]
        errors = classify_errors(gold, pred)
        assert errors["type_error"] == 1
        assert errors["missing"] == 1
        assert errors["spurious"] == 1
