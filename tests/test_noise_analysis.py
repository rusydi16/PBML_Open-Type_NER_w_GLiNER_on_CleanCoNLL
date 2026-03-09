"""Tests for src/noise_analysis.py."""

import pytest

from src.noise_analysis import classify_noise_attribution, aggregate_noise_analysis


# ---------------------------------------------------------------------------
# classify_noise_attribution
# ---------------------------------------------------------------------------

class TestClassifyNoiseAttribution:
    def test_correct_both(self):
        """Prediction matches both CoNLL and CleanCoNLL gold."""
        pred = [{"start_token": 0, "end_token": 2, "label": "PER", "text": "X", "score": 0.9}]
        conll_gold = [{"start": 0, "end": 2, "label": "PER"}]
        clean_gold = [{"start": 0, "end": 2, "label": "PER"}]
        result = classify_noise_attribution(pred, conll_gold, clean_gold)
        assert result["correct_both"] == 1
        assert result["noise_penalized_correct"] == 0
        assert result["model_learned_noise"] == 0
        assert result["genuine_error"] == 0

    def test_noise_penalized_correct(self):
        """Prediction matches CleanCoNLL but NOT CoNLL-03 (noise unfairly penalizes)."""
        pred = [{"start_token": 0, "end_token": 2, "label": "PER", "text": "X", "score": 0.9}]
        conll_gold = [{"start": 0, "end": 2, "label": "LOC"}]  # different label in noisy
        clean_gold = [{"start": 0, "end": 2, "label": "PER"}]  # matches pred
        result = classify_noise_attribution(pred, conll_gold, clean_gold)
        assert result["noise_penalized_correct"] == 1
        assert result["correct_both"] == 0
        assert result["model_learned_noise"] == 0
        assert result["genuine_error"] == 0
        assert len(result["noise_penalized_correct_examples"]) == 1
        assert result["noise_penalized_correct_examples"][0] == {"start": 0, "end": 2, "label": "PER"}

    def test_model_learned_noise(self):
        """Prediction matches CoNLL-03 but NOT CleanCoNLL (model learned noisy pattern)."""
        pred = [{"start_token": 0, "end_token": 2, "label": "LOC", "text": "X", "score": 0.9}]
        conll_gold = [{"start": 0, "end": 2, "label": "LOC"}]  # matches pred
        clean_gold = [{"start": 0, "end": 2, "label": "PER"}]  # different label in clean
        result = classify_noise_attribution(pred, conll_gold, clean_gold)
        assert result["model_learned_noise"] == 1
        assert result["correct_both"] == 0
        assert result["noise_penalized_correct"] == 0
        assert result["genuine_error"] == 0
        assert len(result["model_learned_noise_examples"]) == 1
        assert result["model_learned_noise_examples"][0] == {"start": 0, "end": 2, "label": "LOC"}

    def test_genuine_error(self):
        """Prediction matches neither CoNLL nor CleanCoNLL."""
        pred = [{"start_token": 0, "end_token": 2, "label": "ORG", "text": "X", "score": 0.9}]
        conll_gold = [{"start": 0, "end": 2, "label": "PER"}]
        clean_gold = [{"start": 0, "end": 2, "label": "LOC"}]
        result = classify_noise_attribution(pred, conll_gold, clean_gold)
        assert result["genuine_error"] == 1
        assert result["correct_both"] == 0
        assert result["noise_penalized_correct"] == 0
        assert result["model_learned_noise"] == 0
        assert len(result["genuine_error_examples"]) == 1
        assert result["genuine_error_examples"][0] == {"start": 0, "end": 2, "label": "ORG"}

    def test_missed_entities(self):
        """Gold entities not matched by any prediction."""
        pred = []
        conll_gold = [{"start": 0, "end": 2, "label": "PER"}]
        clean_gold = [{"start": 0, "end": 2, "label": "PER"}]
        result = classify_noise_attribution(pred, conll_gold, clean_gold)
        assert result["missed_both"] == 1
        assert result["missed_conll_only"] == 0
        assert result["missed_clean_only"] == 0

    def test_missed_conll_only(self):
        """Entity in CoNLL gold but not in CleanCoNLL gold, missed by pred."""
        pred = []
        conll_gold = [{"start": 0, "end": 2, "label": "PER"}]
        clean_gold = []
        result = classify_noise_attribution(pred, conll_gold, clean_gold)
        assert result["missed_conll_only"] == 1
        assert result["missed_clean_only"] == 0
        assert result["missed_both"] == 0

    def test_missed_clean_only(self):
        """Entity in CleanCoNLL gold but not in CoNLL gold, missed by pred."""
        pred = []
        conll_gold = []
        clean_gold = [{"start": 0, "end": 2, "label": "PER"}]
        result = classify_noise_attribution(pred, conll_gold, clean_gold)
        assert result["missed_clean_only"] == 1
        assert result["missed_conll_only"] == 0
        assert result["missed_both"] == 0

    def test_mixed_scenario(self):
        """Multiple predictions with different attribution categories."""
        pred = [
            {"start_token": 0, "end_token": 2, "label": "PER", "text": "A", "score": 0.9},
            {"start_token": 3, "end_token": 5, "label": "LOC", "text": "B", "score": 0.8},
        ]
        conll_gold = [
            {"start": 0, "end": 2, "label": "PER"},
            {"start": 3, "end": 5, "label": "ORG"},
        ]
        clean_gold = [
            {"start": 0, "end": 2, "label": "PER"},
            {"start": 3, "end": 5, "label": "LOC"},
        ]
        result = classify_noise_attribution(pred, conll_gold, clean_gold)
        assert result["correct_both"] == 1       # PER matches both
        assert result["noise_penalized_correct"] == 1  # LOC matches clean only


# ---------------------------------------------------------------------------
# aggregate_noise_analysis
# ---------------------------------------------------------------------------

class TestAggregateNoiseAnalysis:
    def test_basic_aggregation(self):
        """Sum counts across multiple sentence results."""
        r1 = {
            "correct_both": 1, "noise_penalized_correct": 0,
            "model_learned_noise": 1, "genuine_error": 0,
            "missed_conll_only": 0, "missed_clean_only": 0, "missed_both": 0,
            "noise_penalized_correct_examples": [],
            "model_learned_noise_examples": [{"start": 0, "end": 2, "label": "LOC"}],
            "genuine_error_examples": [],
        }
        r2 = {
            "correct_both": 2, "noise_penalized_correct": 1,
            "model_learned_noise": 0, "genuine_error": 1,
            "missed_conll_only": 1, "missed_clean_only": 0, "missed_both": 1,
            "noise_penalized_correct_examples": [{"start": 5, "end": 7, "label": "PER"}],
            "model_learned_noise_examples": [],
            "genuine_error_examples": [{"start": 8, "end": 10, "label": "ORG"}],
        }
        agg = aggregate_noise_analysis([r1, r2])
        assert agg["correct_both"] == 3
        assert agg["noise_penalized_correct"] == 1
        assert agg["model_learned_noise"] == 1
        assert agg["genuine_error"] == 1
        assert agg["missed_conll_only"] == 1
        assert agg["missed_clean_only"] == 0
        assert agg["missed_both"] == 1
        assert len(agg["model_learned_noise_examples"]) == 1
        assert len(agg["noise_penalized_correct_examples"]) == 1
        assert len(agg["genuine_error_examples"]) == 1

    def test_truncation(self):
        """Examples should be truncated to max_examples."""
        results = []
        for i in range(15):
            results.append({
                "correct_both": 0, "noise_penalized_correct": 1,
                "model_learned_noise": 0, "genuine_error": 0,
                "missed_conll_only": 0, "missed_clean_only": 0, "missed_both": 0,
                "noise_penalized_correct_examples": [{"start": i, "end": i + 1, "label": "PER"}],
                "model_learned_noise_examples": [],
                "genuine_error_examples": [],
            })
        agg = aggregate_noise_analysis(results, max_examples=10)
        assert agg["noise_penalized_correct"] == 15
        assert len(agg["noise_penalized_correct_examples"]) == 10

    def test_truncation_custom_limit(self):
        """Custom max_examples is respected."""
        results = []
        for i in range(5):
            results.append({
                "correct_both": 0, "noise_penalized_correct": 0,
                "model_learned_noise": 0, "genuine_error": 1,
                "missed_conll_only": 0, "missed_clean_only": 0, "missed_both": 0,
                "noise_penalized_correct_examples": [],
                "model_learned_noise_examples": [],
                "genuine_error_examples": [{"start": i, "end": i + 1, "label": "ORG"}],
            })
        agg = aggregate_noise_analysis(results, max_examples=3)
        assert agg["genuine_error"] == 5
        assert len(agg["genuine_error_examples"]) == 3
