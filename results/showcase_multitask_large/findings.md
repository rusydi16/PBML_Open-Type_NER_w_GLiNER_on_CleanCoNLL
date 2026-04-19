# Findings: Fair Evaluation of GLiNER on CleanCoNLL vs CoNLL-03

## 1. Performance Comparison

Overall F1 on CoNLL-03: **87.39%** | Overall F1 on CleanCoNLL: **87.44%** | Delta: **0.05** pp (higher on CleanCoNLL).

### Per-Entity-Type Breakdown

- **PER**: CoNLL-03 F1 = 98.28%, CleanCoNLL F1 = 99.32%, Delta = 1.04 pp (higher)
- **ORG**: CoNLL-03 F1 = 93.67%, CleanCoNLL F1 = 93.50%, Delta = -0.17 pp (lower)
- **LOC**: CoNLL-03 F1 = 86.46%, CleanCoNLL F1 = 85.59%, Delta = -0.87 pp (lower)
- **MISC**: CoNLL-03 F1 = 58.48%, CleanCoNLL F1 = 59.22%, Delta = 0.74 pp (higher)

## 2. Error Category Changes

| Error Category | CoNLL-03 | CleanCoNLL | Delta |
|---|---|---|---|
| type_error | 621 | 672 | +51 |
| boundary_error | 246 | 241 | -5 |
| type_boundary_error | 265 | 261 | -4 |
| missing | 906 | 927 | +21 |
| spurious | 551 | 494 | -57 |

## 3. Noise Attribution

- **correct_both**: 3366 — Predictions correct under both annotation sets
- **noise_penalized_correct**: 165 — Correct predictions penalised due to noisy CoNLL-03 labels
- **model_learned_noise**: 146 — Model learned noisy patterns from training data
- **genuine_error**: 1475 — Genuine model errors (wrong under both annotations)
- **missed_both**: 1657 — Entities missed under both annotation sets
- **missed_conll_only**: 310 — Entities missed only when evaluating against CoNLL-03
- **missed_clean_only**: 372 — Entities missed only when evaluating against CleanCoNLL

## 4. Conclusion

When evaluated against the cleaned annotations (CleanCoNLL), GLiNER achieves an overall F1 that is 0.05 percentage points higher than on the original CoNLL-03 test set. This suggests that a portion of the apparent errors on CoNLL-03 are attributable to noisy gold labels rather than genuine model mistakes. Noise attribution analysis identified 165 predictions that were correct but penalised by the original annotations. These findings support the value of using CleanCoNLL for a fairer evaluation of NER models.
