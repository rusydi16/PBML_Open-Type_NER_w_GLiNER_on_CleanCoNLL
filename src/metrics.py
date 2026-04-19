"""Entity-level evaluation metrics and error classification."""


def _normalize_entity(entity):
    """Extract (start, end, label) from either gold or pred format.

    Gold entities have keys: start, end, label.
    Pred entities have keys: start_token, end_token, label (plus text, score).
    """
    start = entity.get("start", entity.get("start_token"))
    end = entity.get("end", entity.get("end_token"))
    label = entity["label"]
    return (start, end, label)


def _spans_overlap(s1_start, s1_end, s2_start, s2_end):
    """Return True if the two spans overlap (exclusive end boundaries)."""
    return s1_start < s2_end and s2_start < s1_end


def compute_entity_metrics(gold_entities, pred_entities):
    """Entity-level exact-match precision, recall, and F1.

    Returns dict with keys: precision, recall, f1, tp, fp, fn.
    All float values rounded to 4 decimal places.
    """
    gold_set = {_normalize_entity(e) for e in gold_entities}
    pred_set = {_normalize_entity(e) for e in pred_entities}

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
           if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_per_type_metrics(gold_entities, pred_entities, entity_types):
    """Per-type breakdown of entity metrics.

    Returns dict mapping entity type string to metrics dict.
    """
    results = {}
    for etype in entity_types:
        gold_filtered = [e for e in gold_entities if e["label"] == etype]
        pred_filtered = [e for e in pred_entities if e["label"] == etype]
        results[etype] = compute_entity_metrics(gold_filtered, pred_filtered)
    return results


def compute_entity_metrics_aggregated(per_sentence_pairs):
    """Entity-level P/R/F1 aggregated across multiple sentences.

    Takes an iterable of ``(gold_entities, pred_entities)`` tuples — one per
    sentence — and sums TP/FP/FN over the sentences. This is the correct
    multi-sentence aggregation: using ``compute_entity_metrics`` on a flat
    list that crosses sentence boundaries silently deduplicates entities
    whose ``(start, end, label)`` tuples happen to match across sentences
    (e.g. two sentences each with ``PER`` at tokens 0-2), which drastically
    undercounts TP/FP/FN.
    """
    total_tp = total_fp = total_fn = 0
    for gold, pred in per_sentence_pairs:
        gold_set = {_normalize_entity(e) for e in gold}
        pred_set = {_normalize_entity(e) for e in pred}
        total_tp += len(gold_set & pred_set)
        total_fp += len(pred_set - gold_set)
        total_fn += len(gold_set - pred_set)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def compute_per_type_metrics_aggregated(per_sentence_pairs, entity_types):
    """Per-type P/R/F1 aggregated across multiple sentences."""
    # Materialize once so we can iterate per type without re-consuming.
    pairs_list = list(per_sentence_pairs)
    results = {}
    for etype in entity_types:
        filtered = [
            ([e for e in gold if e["label"] == etype],
             [e for e in pred if e["label"] == etype])
            for gold, pred in pairs_list
        ]
        results[etype] = compute_entity_metrics_aggregated(filtered)
    return results


def classify_errors(gold_entities, pred_entities):
    """Classify non-matching entity pairs into error categories.

    Returns dict with counts for:
      type_error, boundary_error, type_boundary_error, missing, spurious.
    """
    gold_normed = [_normalize_entity(e) for e in gold_entities]
    pred_normed = [_normalize_entity(e) for e in pred_entities]

    gold_set = set(gold_normed)
    pred_set = set(pred_normed)

    # Only consider non-exact-match entities
    unmatched_gold = [g for g in gold_normed if g not in pred_set]
    unmatched_pred = [p for p in pred_normed if p not in gold_set]

    counts = {
        "type_error": 0,
        "boundary_error": 0,
        "type_boundary_error": 0,
        "missing": 0,
        "spurious": 0,
    }

    gold_matched = set()
    pred_matched = set()

    for gi, g in enumerate(unmatched_gold):
        g_start, g_end, g_label = g
        for pi, p in enumerate(unmatched_pred):
            if pi in pred_matched:
                continue
            p_start, p_end, p_label = p
            if not _spans_overlap(g_start, g_end, p_start, p_end):
                continue

            same_span = (g_start == p_start and g_end == p_end)
            same_label = (g_label == p_label)

            if same_span and not same_label:
                counts["type_error"] += 1
            elif not same_span and same_label:
                counts["boundary_error"] += 1
            elif not same_span and not same_label:
                counts["type_boundary_error"] += 1

            gold_matched.add(gi)
            pred_matched.add(pi)
            break

    # Gold entities with no overlapping prediction at all
    for gi, g in enumerate(unmatched_gold):
        if gi not in gold_matched:
            counts["missing"] += 1

    # Pred entities with no overlapping gold at all
    for pi, p in enumerate(unmatched_pred):
        if pi not in pred_matched:
            counts["spurious"] += 1

    return counts
