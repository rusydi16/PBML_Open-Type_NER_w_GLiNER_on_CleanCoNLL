"""Tests for src/statistical_tests.py."""

import pytest

from src.statistical_tests import bootstrap_entity_f1, paired_bootstrap_test


def _make_perfect_data(n_sentences=30):
    """Create n_sentences of synthetic data with perfect predictions.

    Each sentence has one gold entity and one matching pred entity.
    """
    gold_sentences = []
    pred_entries = []
    for i in range(n_sentences):
        gold_sentences.append({
            "id": i,
            "entities": [{"start": 0, "end": 2, "label": "PER"}],
        })
        pred_entries.append({
            "id": i,
            "entities": [
                {"start_token": 0, "end_token": 2, "label": "PER",
                 "text": "John", "score": 0.99},
            ],
        })
    return gold_sentences, pred_entries


def _make_partial_data(n_sentences=30):
    """Create data where predictions miss half the entities.

    Each sentence has two gold entities but only one matching prediction.
    """
    gold_sentences = []
    pred_entries = []
    for i in range(n_sentences):
        gold_sentences.append({
            "id": i,
            "entities": [
                {"start": 0, "end": 2, "label": "PER"},
                {"start": 5, "end": 7, "label": "LOC"},
            ],
        })
        pred_entries.append({
            "id": i,
            "entities": [
                {"start_token": 0, "end_token": 2, "label": "PER",
                 "text": "John", "score": 0.95},
            ],
        })
    return gold_sentences, pred_entries


# ---------------------------------------------------------------------------
# bootstrap_entity_f1
# ---------------------------------------------------------------------------

class TestBootstrapEntityF1:
    def test_perfect_predictions_mean_near_one(self):
        """With perfect predictions, mean F1 should be ~1.0."""
        gold, pred = _make_perfect_data(50)
        result = bootstrap_entity_f1(gold, pred, n_iterations=500, seed=42)
        assert abs(result["mean"] - 1.0) < 0.01

    def test_perfect_predictions_tight_ci(self):
        """With perfect predictions, CI should be very tight around 1.0."""
        gold, pred = _make_perfect_data(50)
        result = bootstrap_entity_f1(gold, pred, n_iterations=500, seed=42)
        assert result["ci_lower"] >= 0.99
        assert result["ci_upper"] <= 1.01

    def test_returns_correct_keys(self):
        gold, pred = _make_perfect_data(30)
        result = bootstrap_entity_f1(gold, pred, n_iterations=100, seed=42)
        expected_keys = {"mean", "std", "ci_lower", "ci_upper"}
        assert set(result.keys()) == expected_keys

    def test_ci_lower_le_mean_le_ci_upper(self):
        gold, pred = _make_partial_data(30)
        result = bootstrap_entity_f1(gold, pred, n_iterations=500, seed=42)
        assert result["ci_lower"] <= result["mean"] <= result["ci_upper"]

    def test_partial_predictions_f1_below_one(self):
        """With partial predictions, mean F1 should be well below 1.0."""
        gold, pred = _make_partial_data(50)
        result = bootstrap_entity_f1(gold, pred, n_iterations=500, seed=42)
        assert result["mean"] < 0.9

    def test_reproducibility_with_same_seed(self):
        gold, pred = _make_partial_data(30)
        r1 = bootstrap_entity_f1(gold, pred, n_iterations=200, seed=123)
        r2 = bootstrap_entity_f1(gold, pred, n_iterations=200, seed=123)
        assert r1 == r2


# ---------------------------------------------------------------------------
# paired_bootstrap_test
# ---------------------------------------------------------------------------

class TestPairedBootstrapTest:
    def test_identical_datasets_delta_near_zero(self):
        """Comparing identical setups should give delta ≈ 0."""
        gold, pred = _make_perfect_data(50)
        result = paired_bootstrap_test(
            gold, pred, gold, pred, n_iterations=500, seed=42
        )
        assert abs(result["delta_mean"]) < 0.01

    def test_returns_correct_keys(self):
        gold, pred = _make_perfect_data(30)
        result = paired_bootstrap_test(
            gold, pred, gold, pred, n_iterations=100, seed=42
        )
        expected_keys = {
            "delta_mean", "delta_std", "delta_ci_lower",
            "delta_ci_upper", "p_value",
        }
        assert set(result.keys()) == expected_keys

    def test_better_system_positive_delta(self):
        """System B (perfect) vs system A (partial) should give positive delta."""
        gold_a, pred_a = _make_partial_data(50)
        gold_b, pred_b = _make_perfect_data(50)
        result = paired_bootstrap_test(
            gold_a, pred_a, gold_b, pred_b, n_iterations=500, seed=42
        )
        assert result["delta_mean"] > 0.0
        assert result["p_value"] < 0.05

    def test_delta_ci_contains_mean(self):
        gold, pred = _make_partial_data(30)
        gold2, pred2 = _make_perfect_data(30)
        result = paired_bootstrap_test(
            gold, pred, gold2, pred2, n_iterations=500, seed=42
        )
        assert result["delta_ci_lower"] <= result["delta_mean"] <= result["delta_ci_upper"]

    def test_identical_pvalue_high(self):
        """For identical systems, p_value should be high (not significant)."""
        gold, pred = _make_partial_data(50)
        result = paired_bootstrap_test(
            gold, pred, gold, pred, n_iterations=500, seed=42
        )
        assert result["p_value"] >= 0.4
