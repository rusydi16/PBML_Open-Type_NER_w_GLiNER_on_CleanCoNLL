"""Exploratory data analysis helpers for CoNLL-03 / CleanCoNLL comparison."""

from collections import Counter
from typing import Any


def basic_stats(sentences: list[dict[str, Any]]) -> dict[str, Any]:
    """Sentence/token/entity counts and simple averages for one split."""
    n_sent = len(sentences)
    n_tok = sum(len(s["tokens"]) for s in sentences)
    entities = [e for s in sentences for e in s["entities"]]
    n_ent = len(entities)
    ent_lens = [e["end"] - e["start"] for e in entities]
    avg_sent_len = n_tok / n_sent if n_sent else 0.0
    avg_ent_per_sent = n_ent / n_sent if n_sent else 0.0
    avg_ent_len = sum(ent_lens) / n_ent if n_ent else 0.0
    return {
        "sentences": n_sent,
        "tokens": n_tok,
        "entities": n_ent,
        "avg_sent_len": round(avg_sent_len, 2),
        "avg_ent_per_sent": round(avg_ent_per_sent, 2),
        "avg_ent_len": round(avg_ent_len, 2),
    }


def entity_type_counts(sentences: list[dict[str, Any]]) -> dict[str, int]:
    """Count entities per type label."""
    counter: Counter = Counter()
    for s in sentences:
        for e in s["entities"]:
            counter[e["label"]] += 1
    return dict(counter)


def entity_length_histogram(
    sentences: list[dict[str, Any]], max_bin: int = 10
) -> dict[str, int]:
    """Histogram of entity lengths (in tokens). Lengths >= max_bin bucketed as '>=N'."""
    hist: Counter = Counter()
    for s in sentences:
        for e in s["entities"]:
            length = e["end"] - e["start"]
            key = str(length) if length < max_bin else f">={max_bin}"
            hist[key] += 1
    ordered: dict[str, int] = {}
    for i in range(1, max_bin):
        k = str(i)
        if k in hist:
            ordered[k] = hist[k]
    overflow_key = f">={max_bin}"
    if overflow_key in hist:
        ordered[overflow_key] = hist[overflow_key]
    return ordered


def _spans_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


def categorize_sentence_deltas(
    sent_a: dict[str, Any],
    sent_b: dict[str, Any],
) -> dict[str, int]:
    """Classify entity differences between two aligned sentence annotations.

    Returns counts for: ``exact_match``, ``type_changed``, ``boundary_changed``,
    ``removed`` (in A only), ``added`` (in B only). Assumes the two sentences
    share identical tokens, so token indices are directly comparable.
    """
    a_ents = sent_a["entities"]
    b_ents = sent_b["entities"]

    a_set = {(e["start"], e["end"], e["label"]) for e in a_ents}
    b_set = {(e["start"], e["end"], e["label"]) for e in b_ents}
    exact = a_set & b_set

    a_rem = [e for e in a_ents if (e["start"], e["end"], e["label"]) not in exact]
    b_rem = [e for e in b_ents if (e["start"], e["end"], e["label"]) not in exact]

    matched_b: set[int] = set()
    type_changed = 0
    boundary_changed = 0
    removed = 0

    for ea in a_rem:
        # 1) same span, different label
        found = False
        for bi, eb in enumerate(b_rem):
            if bi in matched_b:
                continue
            if ea["start"] == eb["start"] and ea["end"] == eb["end"]:
                type_changed += 1
                matched_b.add(bi)
                found = True
                break
        if found:
            continue
        # 2) same label, overlapping span
        for bi, eb in enumerate(b_rem):
            if bi in matched_b:
                continue
            if ea["label"] == eb["label"] and _spans_overlap(
                ea["start"], ea["end"], eb["start"], eb["end"]
            ):
                boundary_changed += 1
                matched_b.add(bi)
                found = True
                break
        if found:
            continue
        removed += 1

    added = len(b_rem) - len(matched_b)

    return {
        "exact_match": len(exact),
        "type_changed": type_changed,
        "boundary_changed": boundary_changed,
        "removed": removed,
        "added": added,
    }


def categorize_sentence_deltas_with_examples(
    sent_a: dict[str, Any],
    sent_b: dict[str, Any],
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    """Same as ``categorize_sentence_deltas`` but also returns per-category examples.

    Example dicts carry enough context to quote in a report: tokens, span,
    before/after label, etc.
    """
    counts = {
        "exact_match": 0,
        "type_changed": 0,
        "boundary_changed": 0,
        "removed": 0,
        "added": 0,
    }
    examples: dict[str, list[dict[str, Any]]] = {
        "type_changed": [],
        "boundary_changed": [],
        "removed": [],
        "added": [],
    }

    tokens = sent_a["tokens"]

    a_ents = sent_a["entities"]
    b_ents = sent_b["entities"]
    a_set = {(e["start"], e["end"], e["label"]) for e in a_ents}
    b_set = {(e["start"], e["end"], e["label"]) for e in b_ents}
    exact = a_set & b_set
    counts["exact_match"] = len(exact)

    a_rem = [e for e in a_ents if (e["start"], e["end"], e["label"]) not in exact]
    b_rem = [e for e in b_ents if (e["start"], e["end"], e["label"]) not in exact]
    matched_b: set[int] = set()

    for ea in a_rem:
        # type_changed
        hit = None
        for bi, eb in enumerate(b_rem):
            if bi in matched_b:
                continue
            if ea["start"] == eb["start"] and ea["end"] == eb["end"]:
                hit = ("type_changed", bi, eb)
                break
        if hit is None:
            for bi, eb in enumerate(b_rem):
                if bi in matched_b:
                    continue
                if ea["label"] == eb["label"] and _spans_overlap(
                    ea["start"], ea["end"], eb["start"], eb["end"]
                ):
                    hit = ("boundary_changed", bi, eb)
                    break
        if hit is None:
            counts["removed"] += 1
            examples["removed"].append(
                {
                    "sentence": " ".join(tokens),
                    "span": " ".join(tokens[ea["start"]:ea["end"]]),
                    "label_a": ea["label"],
                }
            )
        else:
            cat, bi, eb = hit
            counts[cat] += 1
            matched_b.add(bi)
            examples[cat].append(
                {
                    "sentence": " ".join(tokens),
                    "span_a": " ".join(tokens[ea["start"]:ea["end"]]),
                    "span_b": " ".join(tokens[eb["start"]:eb["end"]]),
                    "label_a": ea["label"],
                    "label_b": eb["label"],
                }
            )

    for bi, eb in enumerate(b_rem):
        if bi in matched_b:
            continue
        counts["added"] += 1
        examples["added"].append(
            {
                "sentence": " ".join(tokens),
                "span": " ".join(tokens[eb["start"]:eb["end"]]),
                "label_b": eb["label"],
            }
        )

    return counts, examples


def aggregate_deltas(
    aligned_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    examples_per_category: int = 5,
) -> dict[str, Any]:
    """Aggregate delta counts across many aligned sentence pairs.

    Returns total counts, per-type counts, and a small sample of examples.
    """
    totals: Counter = Counter()
    by_type: dict[str, Counter] = {}
    examples: dict[str, list[dict[str, Any]]] = {
        "type_changed": [],
        "boundary_changed": [],
        "removed": [],
        "added": [],
    }

    for sa, sb in aligned_pairs:
        counts, ex = categorize_sentence_deltas_with_examples(sa, sb)
        for k, v in counts.items():
            totals[k] += v

        # Per-type breakdown: track label from whichever side applies
        a_set = {(e["start"], e["end"], e["label"]) for e in sa["entities"]}
        b_set = {(e["start"], e["end"], e["label"]) for e in sb["entities"]}
        for s_tup in a_set & b_set:
            by_type.setdefault(s_tup[2], Counter())["exact_match"] += 1
        for ea in sa["entities"]:
            if (ea["start"], ea["end"], ea["label"]) in a_set & b_set:
                continue
            # attribute to original label (side A)
            by_type.setdefault(ea["label"], Counter())["changed_or_removed_a"] += 1
        for eb in sb["entities"]:
            if (eb["start"], eb["end"], eb["label"]) in a_set & b_set:
                continue
            by_type.setdefault(eb["label"], Counter())["changed_or_added_b"] += 1

        for k, lst in ex.items():
            remaining = examples_per_category - len(examples[k])
            if remaining > 0 and lst:
                examples[k].extend(lst[:remaining])

    return {
        "totals": dict(totals),
        "by_type": {k: dict(v) for k, v in by_type.items()},
        "examples": examples,
    }
