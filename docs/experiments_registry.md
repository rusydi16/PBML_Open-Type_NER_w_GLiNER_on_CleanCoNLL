# Registry Eksperimen

Dokumen terpusat untuk semua eksperimen di repo ini. Setiap entry mencantumkan
**prompt label yang dipakai GLiNER** secara eksplisit agar hasil selalu traceable
ke konfigurasi yang menghasilkannya.

Urutan di bawah sesuai urutan kronologis di mana eksperimen dijalankan.

---

## 1. Primary Pipeline (RQ1, RQ2, RQ3)

**Tujuan:** Mengukur performa GLiNER pada CoNLL-03 (noisy) vs CleanCoNLL (clean) dengan
label standar 4-tipe.

- **Config**: [`configs/default.yaml`](../configs/default.yaml)
- **Model**: `urchade/gliner_medium-v2.1` (86 M, DeBERTa-v3 base)
- **Inference threshold**: 0.5
- **Prompt labels** (4 prompts, 1-to-1 mapping ke CoNLL):

  | GLiNER prompt | CoNLL label |
  |---|---|
  | `person` | PER |
  | `organization` | ORG |
  | `location` | LOC |
  | `miscellaneous` | MISC |

- **Output**: [`results/`](../results) — `metrics_*_test.json`, `predictions_*_test.json`,
  `noise_analysis_test.json`, `comparison_table.md`, `findings.md`
- **Bootstrap significance**: [`results/significance_test_test.json`](../results/significance_test_test.json)
- **Ringkasan hasil** (test split, F1):

  | | CoNLL | CleanCoNLL | Δ |
  |---|---|---|---|
  | Overall | 61.26 | 58.71 | **−2.55** (p=1.0, CI [−0.034, −0.020]) |
  | PER | 80.92 | 81.88 | +0.96 |
  | ORG | 47.74 | 46.40 | −1.34 |
  | LOC | 65.80 | 61.57 | −4.23 |
  | MISC | 0.00 | 0.00 | 0.00 |

---

## 2. Model-Size Ablation (Opsi Penguatan Riset 2)

**Tujuan:** Apakah kapasitas model (44 M → 304 M) mempengaruhi sensitivitas terhadap label noise?

- **Config**: [`configs/ablation.yaml`](../configs/ablation.yaml)
- **Models**:
  - `urchade/gliner_small-v2.1` (44 M)
  - `urchade/gliner_medium-v2.1` (86 M)
  - `urchade/gliner_large-v2.1` (304 M)
- **Prompt labels**: **sama persis dengan Primary** (4 single prompts)
- **Output**: [`results/ablation/<size>/`](../results/ablation/), tabel ringkasan
  [`ablation_table.md`](../results/ablation_table.md)
- **Ringkasan hasil** (CoNLL / CleanCoNLL test, F1):

  | Model | CoNLL | Clean | Δ | noise_penalized |
  |---|---|---|---|---|
  | small (44 M) | 60.51 | 58.09 | −2.42 | 117 |
  | medium (86 M) | 61.26 | 58.71 | **−2.55** | 107 |
  | large (304 M) | 54.80 | 52.45 | −2.35 | 123 |

- **Finding**: F1 tidak monotonic terhadap kapasitas — medium > small > large. Large
  over-predicts (TP=3677, FP=4095).

---

## 3. Showcase: Multitask-Large (Archived)

**Tujuan:** Baseline SOTA zero-shot NER untuk perspective.

- **Config**: snapshot di `results/showcase_multitask_large/` (archived, tidak dipakai aktif)
- **Model**: `knowledgator/gliner-multitask-large-v0.5` (~440 M, multitask-tuned variant)
- **Prompt labels**: **sama dengan Primary** (4 single prompts)
- **Output**: [`results/showcase_multitask_large/`](../results/showcase_multitask_large/)
- **Ringkasan hasil**:

  | | CoNLL F1 | Clean F1 | Δ |
  |---|---|---|---|
  | Overall | 65.99 | 65.79 | −0.20 |
  | MISC | 6.55 | 6.81 | +0.26 |

- **Catatan**: dipindahkan ke archive setelah primary diganti ke `gliner_medium-v2.1` agar
  konsistensi antara core/ablation/fine-tune terjaga pada satu keluarga model.

---

## 4. Non-GLiNER Baseline (Feedback sidang: baseline selain GLiNER)

**Tujuan:** Bandingkan model open-type vs supervised yang dilatih eksplisit di CoNLL noisy.

- **Config**: [`configs/baseline.yaml`](../configs/baseline.yaml)
- **Model**: `dslim/bert-base-NER` (110 M, BERT-base fine-tuned pada CoNLL-03 train)
- **Framework**: HuggingFace `pipeline("ner", aggregation_strategy="simple")`
- **Prompt labels**: **tidak pakai prompt** (supervised, label diprediksi langsung sebagai IOB tags).
  Aggregation pipeline mengeluarkan `entity_group` yang sudah berupa `PER/ORG/LOC/MISC`
  (lihat [`src/baseline.py`](../src/baseline.py) — prefix `B-`/`I-` distripping otomatis).
- **Output**: [`results/baseline/bert-ner/`](../results/baseline/bert-ner/),
  [`baseline_table.md`](../results/baseline_table.md)
- **Ringkasan hasil**:

  | | CoNLL F1 | Clean F1 | Δ |
  |---|---|---|---|
  | Overall | 85.58 | 83.10 | **−2.48** |
  | PER | 80.34 | 80.86 | +0.52 |
  | ORG | 88.10 | 83.85 | −4.25 |
  | LOC | 92.00 | 86.13 | −5.87 |
  | MISC | **78.79** | **80.85** | +2.06 |

- **Finding**: BERT-NER dilatih di CoNLL noisy → delta negatif saat di-eval di Clean.
  Tapi MISC tetap tinggi (79 %) karena supervised training melihat 3438 gold MISC.

---

## 5. Fine-tune GLiNER (Opsi Penguatan Riset 3 / RQ4)

**Tujuan:** Apakah melatih GLiNER pada CleanCoNLL menghasilkan model yang lebih baik di
dunia nyata dibanding melatih pada CoNLL-03 noisy?

- **Config**: [`configs/finetune.yaml`](../configs/finetune.yaml)
- **Base model**: `urchade/gliner_medium-v2.1` (86 M)
- **Training hyperparameters**: max_steps=2000, learning_rate=1e-5, batch_size=8,
  warmup_ratio=0.1, seed=42
- **Prompt labels** (saat training dan evaluasi, sama persis dengan Primary):

  | GLiNER prompt | CoNLL label |
  |---|---|
  | `person` | PER |
  | `organization` | ORG |
  | `location` | LOC |
  | `miscellaneous` | MISC |

- **Dua training conditions** (beda hanya training data):
  1. `finetuned_conll03` — trained on `conll03_train.json` + eval on `conll03_dev.json`
  2. `finetuned_cleanconll` — trained on `cleanconll_train.json` + eval on `cleanconll_dev.json`
- **Evaluasi akhir**: kedua model di-test pada **CleanCoNLL test** (gold bersih)
- **Output**: [`results/finetune/`](../results/finetune/), [`finetune_table.md`](../results/finetune/finetune_table.md)
- **Ringkasan hasil** (eval pada CleanCoNLL test):

  | Training data | F1 | ΔF1 vs CoNLL-trained |
  |---|---|---|
  | CoNLL-03 (noisy) | 89.22 | — |
  | **CleanCoNLL (clean)** | **93.04** | **+3.82** |

- **Finding**: Training pada data bersih memberi lompatan +3.82 F1 di real-world evaluation.
  Bukti langsung untuk RQ4.

---

## 6. Rich-Labels Experiment (Eksplorasi Label Engineering)

**Tujuan:** Bisakah MISC F1 yang 0 % ditingkatkan tanpa retraining, dengan prompt yang
lebih spesifik?

- **Config**: [`configs/rich_labels.yaml`](../configs/rich_labels.yaml)
- **Model**: `urchade/gliner_medium-v2.1` (sama dengan Primary)
- **Inference threshold**: 0.5
- **Prompt labels** (13 prompts, many-to-one mapping):

  | GLiNER prompt | CoNLL label |
  |---|---|
  | `person` | PER |
  | `company` | ORG |
  | `sports team` | ORG |
  | `political party` | ORG |
  | `institution` | ORG |
  | `country` | LOC |
  | `city` | LOC |
  | `geographical region` | LOC |
  | `nationality` | MISC |
  | `event` | MISC |
  | `sporting event` | MISC |
  | `title of work` | MISC |
  | `religion` | MISC |
  | `language` | MISC |

- **Dedup logic** ([`src/inference.py`](../src/inference.py)): kalau >1 prompt memprediksi
  span yang sama, disimpan satu dengan skor tertinggi
- **Output**: [`results_rich_labels/`](../results_rich_labels/)
- **Ringkasan hasil** (CoNLL / CleanCoNLL test, F1):

  | Label | Primary | Rich | Δ |
  |---|---|---|---|
  | PER | 80.92 / 81.88 | 81.38 / 82.20 | +0.46 / +0.32 |
  | ORG | 47.74 / 46.40 | 44.87 / 42.12 | **−2.87 / −4.28** |
  | LOC | 65.80 / 61.57 | **71.41 / 65.02** | +5.61 / +3.45 |
  | MISC | 0.00 / 0.00 | **27.37 / 27.95** | **+27.37 / +27.95** |
  | Overall | 61.26 / 58.71 | **62.78 / 59.36** | +1.52 / +0.65 |

- **Finding**: MISC dari 0 → 27 dengan prompt spesifik. LOC +5.6 dengan split
  country/city/region. ORG regresi −2.9 s/d −4.3 — akibat kompetisi dedup, bukan split
  ORG itu sendiri (terbukti di experiment 7).

---

## 7. Hybrid-Labels Experiment (Single ORG + Rich LOC/MISC)

**Tujuan:** Bisakah ORG dipulihkan dengan kembali ke single prompt `"organization"`
sambil tetap memakai prompt spesifik untuk LOC dan MISC?

- **Config**: [`configs/hybrid_labels.yaml`](../configs/hybrid_labels.yaml)
- **Model**: `urchade/gliner_medium-v2.1`
- **Prompt labels** (10 prompts, many-to-one):

  | GLiNER prompt | CoNLL label |
  |---|---|
  | `person` | PER |
  | `organization` | ORG (single, dikembalikan) |
  | `country` | LOC |
  | `city` | LOC |
  | `geographical region` | LOC |
  | `nationality` | MISC |
  | `event` | MISC |
  | `sporting event` | MISC |
  | `title of work` | MISC |
  | `religion` | MISC |
  | `language` | MISC |

- **Output**: [`results_hybrid_labels/`](../results_hybrid_labels/)
- **Ringkasan hasil** (CoNLL / CleanCoNLL test, F1):

  | Label | Primary | Rich | **Hybrid** | Winner |
  |---|---|---|---|---|
  | PER | 80.92 / 81.88 | 81.38 / 82.20 | **82.15 / 82.98** | Hybrid |
  | ORG | **47.74 / 46.40** | 44.87 / 42.12 | 46.63 / 43.07 | Primary |
  | LOC | 65.80 / 61.57 | **71.41 / 65.02** | 70.78 / 64.64 | Rich |
  | MISC | 0 / 0 | **27.37 / 27.95** | 26.42 / 26.85 | Rich |
  | Overall | 61.26 / 58.71 | **62.78 / 59.36** | 62.57 / 59.10 | Rich |

- **Finding utama**: ORG **tidak sepenuhnya pulih** (46.63 vs primary 47.74) meski prompt
  ORG kembali ke single. Penyebab: prompt tetangga (`country`, `city`, `nationality`,
  `event`) mencuri prediksi ORG via dedup highest-score. Regresi ORG di experiment 6
  bukan karena split prompt ORG sendiri, tapi karena kompetisi lintas-tipe.
- **PER diuntungkan** — Hybrid terbaik di PER kedua dataset (82.15 dan 82.98).
- **Overall: Rich tetap terbaik**, Hybrid tidak worth kompleksitasnya jika fokus overall F1.

---

## Ringkasan Cross-experiment

| Experiment | Model | Jumlah prompts | Overall F1 (CoNLL) | Overall F1 (Clean) |
|---|---|---|---|---|
| Primary | medium | 4 | 61.26 | 58.71 |
| Ablation small | small | 4 | 60.51 | 58.09 |
| Ablation medium | medium | 4 | 61.26 | 58.71 |
| Ablation large | large | 4 | 54.80 | 52.45 |
| Showcase multitask-large | mt-large | 4 | 65.99 | 65.79 |
| Baseline BERT-NER | bert-base | supervised | 85.58 | 83.10 |
| Fine-tune CoNLL | medium | 4 | — | 89.22 |
| **Fine-tune CleanCoNLL** | medium | 4 | — | **93.04** |
| Rich labels | medium | 13 | **62.78** | **59.36** |
| Hybrid labels | medium | 10 | 62.57 | 59.10 |

**Best zero-shot GLiNER pada keluarga v2.1**: Rich labels (62.78 CoNLL, 59.36 Clean).
**Best overall**: Fine-tune CleanCoNLL (93.04 Clean).

## Cara Re-run Eksperimen Apa Saja

Setiap eksperimen dapat di-rerun dengan satu command sesuai config-nya:

```powershell
# 1. Primary core pipeline
python run_all.py --bootstrap

# 2. Ablation
python scripts/run_ablation.py --config configs/ablation.yaml

# 4. Baseline BERT-NER
python scripts/run_baseline.py --config configs/baseline.yaml

# 5. Fine-tune
python scripts/run_finetune.py --config configs/finetune.yaml

# 6. Rich labels
python scripts/run_inference.py --config configs/rich_labels.yaml --force
python scripts/evaluate.py --config configs/rich_labels.yaml
python scripts/generate_report.py --config configs/rich_labels.yaml

# 7. Hybrid labels
python scripts/run_inference.py --config configs/hybrid_labels.yaml --force
python scripts/evaluate.py --config configs/hybrid_labels.yaml
python scripts/generate_report.py --config configs/hybrid_labels.yaml
```

Semua resume-aware — interrupted run otomatis lanjut dari checkpoint.
