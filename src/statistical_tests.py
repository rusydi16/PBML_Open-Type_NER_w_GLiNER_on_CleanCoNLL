"""Bootstrap resampling for statistical significance testing of NER metrics."""

import numpy as np

from src.metrics import compute_entity_metrics


def _compute_f1_for_sample(gold_sentences, pred_entries, indices):
    """Compute entity-level F1 for a bootstrap sample of sentence indices.

    Parameters
    ----------
    gold_sentences : list[dict]
        Each dict has keys "id" and "entities" (list of gold entity dicts).
    pred_entries : list[dict]
        Each dict has keys "id" and "entities" (list of pred entity dicts).
    indices : array-like of int
        Sentence indices (into gold_sentences) to include in this sample.

    Returns
    -------
    float
        The entity-level F1 score for the sampled sentences.
    """
    pred_by_id = {entry["id"]: entry["entities"] for entry in pred_entries}

    all_gold = []
    all_pred = []
    for idx in indices:
        sent = gold_sentences[idx]
        all_gold.extend(sent["entities"])
        all_pred.extend(pred_by_id.get(sent["id"], []))

    metrics = compute_entity_metrics(all_gold, all_pred)
    return metrics["f1"]


def bootstrap_entity_f1(
    gold_sentences,
    pred_entries,
    n_iterations=1000,
    seed=42,
    confidence=0.95,
):
    """Compute bootstrapped confidence interval for entity-level F1.

    Parameters
    ----------
    gold_sentences : list[dict]
        Each dict has "id" and "entities".
    pred_entries : list[dict]
        Each dict has "id" and "entities".
    n_iterations : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).

    Returns
    -------
    dict
        Keys: "mean", "std", "ci_lower", "ci_upper" (all rounded to 4 decimals).
    """
    rng = np.random.RandomState(seed)
    n = len(gold_sentences)

    scores = np.empty(n_iterations)
    for i in range(n_iterations):
        indices = rng.randint(0, n, size=n)
        scores[i] = _compute_f1_for_sample(gold_sentences, pred_entries, indices)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(scores, 100 * alpha / 2))
    ci_upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))

    return {
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores)), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def paired_bootstrap_test(
    gold_a,
    pred_a,
    gold_b,
    pred_b,
    n_iterations=1000,
    seed=42,
    confidence=0.95,
):
    """Paired bootstrap test comparing two NER setups.

    Uses the same bootstrap indices for both systems so the comparison
    is paired.

    Parameters
    ----------
    gold_a, pred_a : list[dict]
        Gold and predictions for system A.
    gold_b, pred_b : list[dict]
        Gold and predictions for system B.
    n_iterations : int
        Number of bootstrap iterations.
    seed : int
        Random seed.
    confidence : float
        Confidence level for delta CI.

    Returns
    -------
    dict
        Keys: "delta_mean", "delta_std", "delta_ci_lower", "delta_ci_upper",
        "p_value". Delta = f1_b - f1_a.
    """
    rng = np.random.RandomState(seed)
    n = len(gold_a)

    deltas = np.empty(n_iterations)
    for i in range(n_iterations):
        indices = rng.randint(0, n, size=n)
        f1_a = _compute_f1_for_sample(gold_a, pred_a, indices)
        f1_b = _compute_f1_for_sample(gold_b, pred_b, indices)
        deltas[i] = f1_b - f1_a

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    p_value = float(np.mean(deltas <= 0))

    return {
        "delta_mean": round(float(np.mean(deltas)), 4),
        "delta_std": round(float(np.std(deltas)), 4),
        "delta_ci_lower": round(ci_lower, 4),
        "delta_ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
    }
