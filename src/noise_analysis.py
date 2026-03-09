"""Noise attribution analysis comparing predictions against CoNLL-03 and CleanCoNLL."""


def _normalize_entity(entity):
    """Extract (start, end, label) from either gold or pred format."""
    start = entity.get("start", entity.get("start_token"))
    end = entity.get("end", entity.get("end_token"))
    label = entity["label"]
    return (start, end, label)


def classify_noise_attribution(pred_entities, conll_gold, clean_gold):
    """For each predicted entity, classify by noise attribution.

    Categories:
      - correct_both: pred matches both CoNLL and CleanCoNLL
      - noise_penalized_correct: pred matches CleanCoNLL but NOT CoNLL-03
      - model_learned_noise: pred matches CoNLL-03 but NOT CleanCoNLL
      - genuine_error: pred matches neither

    Also counts missed entities (gold entities not predicted):
      - missed_both: in both gold sets but not predicted
      - missed_conll_only: only in CoNLL gold, not predicted
      - missed_clean_only: only in CleanCoNLL gold, not predicted

    Returns dict with counts and example lists for the 3 error categories.
    """
    pred_set = {_normalize_entity(e) for e in pred_entities}
    conll_set = {_normalize_entity(e) for e in conll_gold}
    clean_set = {_normalize_entity(e) for e in clean_gold}

    counts = {
        "correct_both": 0,
        "noise_penalized_correct": 0,
        "model_learned_noise": 0,
        "genuine_error": 0,
        "missed_conll_only": 0,
        "missed_clean_only": 0,
        "missed_both": 0,
    }
    noise_penalized_correct_examples = []
    model_learned_noise_examples = []
    genuine_error_examples = []

    for p in pred_set:
        in_conll = p in conll_set
        in_clean = p in clean_set

        if in_conll and in_clean:
            counts["correct_both"] += 1
        elif in_clean and not in_conll:
            counts["noise_penalized_correct"] += 1
            noise_penalized_correct_examples.append(
                {"start": p[0], "end": p[1], "label": p[2]}
            )
        elif in_conll and not in_clean:
            counts["model_learned_noise"] += 1
            model_learned_noise_examples.append(
                {"start": p[0], "end": p[1], "label": p[2]}
            )
        else:
            counts["genuine_error"] += 1
            genuine_error_examples.append(
                {"start": p[0], "end": p[1], "label": p[2]}
            )

    # Missed entities: gold entities not in predictions
    all_gold = conll_set | clean_set
    for g in all_gold:
        if g in pred_set:
            continue
        in_conll = g in conll_set
        in_clean = g in clean_set
        if in_conll and in_clean:
            counts["missed_both"] += 1
        elif in_conll and not in_clean:
            counts["missed_conll_only"] += 1
        elif in_clean and not in_conll:
            counts["missed_clean_only"] += 1

    counts["noise_penalized_correct_examples"] = noise_penalized_correct_examples
    counts["model_learned_noise_examples"] = model_learned_noise_examples
    counts["genuine_error_examples"] = genuine_error_examples

    return counts


def aggregate_noise_analysis(per_sentence_results, max_examples=10):
    """Aggregate noise analysis results across sentences.

    Sums all count fields and collects examples, truncated to max_examples
    per category.
    """
    count_keys = [
        "correct_both", "noise_penalized_correct", "model_learned_noise",
        "genuine_error", "missed_conll_only", "missed_clean_only", "missed_both",
    ]
    example_keys = [
        "noise_penalized_correct_examples",
        "model_learned_noise_examples",
        "genuine_error_examples",
    ]

    agg = {k: 0 for k in count_keys}
    agg.update({k: [] for k in example_keys})

    for result in per_sentence_results:
        for k in count_keys:
            agg[k] += result[k]
        for k in example_keys:
            agg[k].extend(result[k])

    # Truncate examples
    for k in example_keys:
        agg[k] = agg[k][:max_examples]

    return agg
