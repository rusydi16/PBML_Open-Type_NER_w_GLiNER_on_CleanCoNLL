# Findings: Fair Evaluation of GLiNER on CleanCoNLL vs CoNLL-03

## 1. Performance Comparison

Overall F1 on CoNLL-03: **62.57%** | Overall F1 on CleanCoNLL: **59.10%** | Delta: **-3.47** pp (lower on CleanCoNLL).

### Per-Entity-Type Breakdown

- **PER**: CoNLL-03 F1 = 82.15%, CleanCoNLL F1 = 82.98%, Delta = 0.83 pp (higher)
- **ORG**: CoNLL-03 F1 = 46.63%, CleanCoNLL F1 = 43.07%, Delta = -3.56 pp (lower)
- **LOC**: CoNLL-03 F1 = 70.78%, CleanCoNLL F1 = 64.64%, Delta = -6.14 pp (lower)
- **LOC**: CoNLL-03 F1 = 70.78%, CleanCoNLL F1 = 64.64%, Delta = -6.14 pp (lower)
- **LOC**: CoNLL-03 F1 = 70.78%, CleanCoNLL F1 = 64.64%, Delta = -6.14 pp (lower)
- **MISC**: CoNLL-03 F1 = 26.42%, CleanCoNLL F1 = 26.85%, Delta = 0.43 pp (higher)
- **MISC**: CoNLL-03 F1 = 26.42%, CleanCoNLL F1 = 26.85%, Delta = 0.43 pp (higher)
- **MISC**: CoNLL-03 F1 = 26.42%, CleanCoNLL F1 = 26.85%, Delta = 0.43 pp (higher)
- **MISC**: CoNLL-03 F1 = 26.42%, CleanCoNLL F1 = 26.85%, Delta = 0.43 pp (higher)
- **MISC**: CoNLL-03 F1 = 26.42%, CleanCoNLL F1 = 26.85%, Delta = 0.43 pp (higher)
- **MISC**: CoNLL-03 F1 = 26.42%, CleanCoNLL F1 = 26.85%, Delta = 0.43 pp (higher)

## 2. Error Category Changes

| Error Category | CoNLL-03 | CleanCoNLL | Delta |
|---|---|---|---|
| type_error | 845 | 1108 | +263 |
| boundary_error | 256 | 247 | -9 |
| type_boundary_error | 255 | 249 | -6 |
| missing | 507 | 525 | +18 |
| spurious | 1310 | 1244 | -66 |

## 3. Noise Attribution

- **correct_both**: 3409 — Predictions correct under both annotation sets
- **noise_penalized_correct**: 89 — Correct predictions penalised due to noisy CoNLL-03 labels
- **model_learned_noise**: 282 — Model learned noisy patterns from training data
- **genuine_error**: 2511 — Genuine model errors (wrong under both annotations)
- **missed_both**: 1614 — Entities missed under both annotation sets
- **missed_conll_only**: 174 — Entities missed only when evaluating against CoNLL-03
- **missed_clean_only**: 448 — Entities missed only when evaluating against CleanCoNLL

## 4. Conclusion

When evaluated against CleanCoNLL, GLiNER's overall F1 is 3.47 percentage points lower than on the original CoNLL-03 test set. This indicates that some predictions the model gets 'right' on CoNLL-03 may actually be matching noisy labels. Noise attribution analysis found 282 cases where the model appears to have learned noisy patterns.
