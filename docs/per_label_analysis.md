# Analisis Per-Label: Dimana GLiNER Kuat dan Lemah?

Dokumen ini membongkar skor keseluruhan menjadi per-tipe entitas untuk menjawab:
kenapa GLiNER zero-shot jauh di bawah BERT supervised? Di mana gap paling lebar?
Apakah gap itu bug atau karakteristik intrinsik open-type NER?

## 1. Tabel F1 per Label (CoNLL-03 test)

| Model | PER | ORG | LOC | MISC | Overall |
|---|---|---|---|---|---|
| GLiNER small (44 M) | 81.43 | 46.49 | 65.63 | **1.19** | 60.51 |
| GLiNER medium (86 M, *primary*) | 80.92 | 47.74 | 65.80 | **0.00** | 61.26 |
| GLiNER large (304 M) | 76.15 | 46.69 | 63.02 | **1.19** | 54.80 |
| GLiNER multitask-large (~440 M) | 90.21 | 60.52 | 66.33 | **6.55** | 65.99 |
| BERT-NER (CoNLL-fine-tuned) | 80.34 | **88.10** | **92.00** | **78.79** | **85.58** |

## 2. Tabel F1 per Label (CleanCoNLL test)

| Model | PER | ORG | LOC | MISC | Overall |
|---|---|---|---|---|---|
| GLiNER small | 82.31 | 45.74 | 61.28 | 1.05 | 58.09 |
| GLiNER medium (*primary*) | 81.88 | 46.40 | 61.57 | 0.00 | 58.71 |
| GLiNER large | 76.84 | 45.53 | 58.47 | 1.34 | 52.45 |
| GLiNER multitask-large | 91.69 | 60.64 | 66.60 | 6.81 | 65.79 |
| BERT-NER | 80.86 | 83.85 | 86.13 | 80.85 | 83.10 |

## 3. Pola yang Muncul

### 3.1 PER (Person) — GLiNER kompetitif
GLiNER multitask-large (91.7) bahkan **sedikit lebih tinggi** dari BERT-NER (80.3-80.9).
Label `"person"` punya makna semantik jelas tanpa ambiguitas budaya. Open-type model
memanfaatkan pengetahuan umum tentang nama manusia dari training data luas.

### 3.2 LOC (Location) — gap moderate
GLiNER 58-66 vs BERT 86-92 → gap ~25 poin. Label `"location"` punya ambiguitas:
- Mencakup negara, kota, wilayah, fitur alam — semuanya OK
- Tapi juga tertarik ke "venue" seperti stadion, alamat jalan yang CoNLL tidak anggap LOC
- Dan kebingungan dengan ORG (tim olahraga sering disebut dengan nama kota)

### 3.3 ORG (Organization) — gap besar
GLiNER 45-60 vs BERT 84-88 → gap ~30-40 poin. Masalah inti:
- Di berita olahraga, "JAPAN" / "CHINA" adalah **tim nasional (ORG)**, bukan negara (LOC)
- GLiNER dengan prompt `"organization"` cenderung map ke LOC karena context spatial
- BERT tahu pattern CoNLL — bahwa di artikel sports, country names sering = teams

Ini divalidasi oleh [docs/eda_summary.md](eda_summary.md) delta analysis:
> train split: LOC 795 changed/removed (terbanyak), ORG 847 added (terbanyak).
> Cleaning CoNLL → CleanCoNLL terutama memindahkan label-label country-as-team dari LOC ke ORG.

### 3.4 MISC — bottleneck utama GLiNER
GLiNER 0-7% vs BERT 79-81% → gap ~75 poin, **terbesar**.

**Penyebab:** MISC di CoNLL-03 didefinisikan secara **eksklusi** — "bukan PER, bukan ORG,
bukan LOC". Isinya heterogen:
- Nationalities/demonyms: "English", "Italian"
- Events: "World Cup", "Holocaust"
- Titles of works: "F.A. Cup"
- Religions, languages, ordinals

GLiNER dengan prompt string `"miscellaneous"` menerjemahkan secara harfiah ke arti
"berbagai/lain-lain" (kata bahasa Inggris umum), bukan definisi kategori spesifik CoNLL.
Hasilnya GLiNER memprediksi MISC pada kata-kata random:
- `"goals"` (common noun)
- `"MISERY"` (kata dalam judul)
- `"DIVISION"` (kata umum dalam konteks sports)

**BERT-NER 79% F1** karena dilatih **supervised** pada 3438 gold MISC di train split —
tidak bergantung pada semantik kata "miscellaneous", cukup pelajari pattern empiris.

## 4. Statistik Detail MISC (CoNLL test, gold=702 MISC entitas)

| Model | # pred MISC | TP | FP | FN | P (%) | R (%) | F1 (%) |
|---|---|---|---|---|---|---|---|
| GLiNER small | 138 | 5 | 133 | 697 | 3.62 | 0.71 | 1.19 |
| GLiNER medium | 36 | 0 | 36 | 702 | 0 | 0 | 0 |
| GLiNER large | 972 | 10 | 962 | 692 | 1.03 | 1.42 | 1.19 |
| GLiNER multitask-large | 305 | 33 | 272 | 669 | 10.82 | 4.70 | 6.55 |
| BERT-NER | 853 | **587** | 201 | 115 | 74.49 | 83.62 | **78.79** |

Observasi:
- GLiNER medium memprediksi hanya 36 MISC — paling konservatif, tapi zero TP
- GLiNER large memprediksi 972 MISC (over-predict) — banyak noise, tetap nyaris zero correct
- GLiNER multitask-large sedang, F1 6.55% — masih jauh dari BERT
- Hanya BERT yang match definisi CoNLL MISC

## 5. Apakah Bisa Diperbaiki? — Label Engineering

Hipotesis: ganti prompt label GLiNER dari `"miscellaneous"` menjadi label-label spesifik
yang union-nya mencakup MISC:

```yaml
# Eksperimen di configs/rich_labels.yaml
labels:
  - gliner_label: "person"                  → PER
  - gliner_label: "company"                 → ORG
  - gliner_label: "sports team"             → ORG
  - gliner_label: "political party"         → ORG
  - gliner_label: "country"                 → LOC
  - gliner_label: "city"                    → LOC
  - gliner_label: "geographical region"     → LOC
  - gliner_label: "nationality"             → MISC
  - gliner_label: "event"                   → MISC
  - gliner_label: "sporting event"          → MISC
  - gliner_label: "title of work"           → MISC
  - gliner_label: "religion"                → MISC
  - gliner_label: "language"                → MISC
```

Tiap prompt di-kirim ke GLiNER, hasil prediksi di-map ke label CoNLL (PER/ORG/LOC/MISC),
lalu dedupe overlap span (keep highest score) agar tidak double-count.

**Expected**: MISC F1 naik dari ~0% ke ~30-50% (tidak sampai BERT karena GLiNER tetap
zero-shot), ORG & LOC juga bisa naik karena prompt lebih spesifik.

**Cost**: inference dengan 13 label prompts vs 4 → lebih lambat (~3× lama), tapi masih
manageable di GPU (~15-20 menit untuk core pipeline).

### Hasil Label Engineering (primary = `urchade/gliner_medium-v2.1`)

Output di `results_rich_labels/`. Config di [configs/rich_labels.yaml](../configs/rich_labels.yaml).

**CoNLL-03 test:**

| Label | Primary (single prompt) | Rich labels | Delta |
|---|---|---|---|
| PER | 80.92 | 81.38 | **+0.46** |
| ORG | 47.74 | 44.87 | **−2.87** |
| LOC | 65.80 | 71.41 | **+5.61** |
| MISC | **0.00** | **27.37** | **+27.37** |
| Overall | 61.26 | 62.78 | **+1.52** |

**CleanCoNLL test:**

| Label | Primary | Rich labels | Delta |
|---|---|---|---|
| PER | 81.88 | 82.20 | +0.32 |
| ORG | 46.40 | 42.12 | **−4.28** |
| LOC | 61.57 | 65.02 | +3.45 |
| MISC | **0.00** | **27.95** | **+27.95** |
| Overall | 58.71 | 59.36 | +0.65 |

### Interpretasi

**Yang berhasil:**

1. **MISC naik dari 0 % ke ~27 %** — prompt spesifik (`nationality`, `event`, `sporting event`,
   `title of work`, `religion`, `language`) menggantikan satu kata ambigu `miscellaneous`
   membuat GLiNER bisa mengidentifikasi entitas yang sesuai definisi CoNLL MISC. Tidak
   mencapai BERT (79 %) karena masih zero-shot, tapi jauh lebih baik dari baseline.
2. **LOC naik +3.5 s/d +5.6 poin** — prompt `country` + `city` + `geographical region` lebih
   kuat daripada satu kata `location` yang terlalu umum. Boundary dengan stadion/alamat
   juga lebih jelas.
3. **PER stabil** — label `person` sudah well-defined, tidak butuh engineering.

**Yang regresif:**

- **ORG turun −2.9 s/d −4.3 poin.** Splitting `organization` menjadi 4 label spesifik
  (`company`, `sports team`, `political party`, `institution`) ternyata **merugikan ORG**.
  Kemungkinan penyebab:
  - Prompt `"sports team"` dan prompt `"country"` saling tumpang tindih pada entitas
    seperti "JAPAN" di konteks sports. Dedup by highest score kadang salah pilih.
  - Prompt `"company"` tidak mencakup entitas seperti stock exchanges atau trade
    organizations yang ada di ORG CoNLL.
  - Model GLiNER zero-shot mungkin kurang tepat mengenali semua 4 sub-kategori ORG.

**Net result:** +1.52 poin di CoNLL, +0.65 poin di Clean. Overall positif tapi tidak
spektakuler karena ORG regresi membatalkan sebagian gain LOC dan MISC.

### Hybrid Experiment (Single ORG, Rich LOC + MISC)

Config: [configs/hybrid_labels.yaml](../configs/hybrid_labels.yaml). Hipotesis: kembalikan
`organization` ke single prompt untuk memulihkan ORG F1, tapi tetap pakai prompt spesifik
untuk LOC dan MISC. Output di `results_hybrid_labels/`.

**CoNLL-03 test:**

| Label | Primary | Rich | **Hybrid** | Winner |
|---|---|---|---|---|
| PER | 80.92 | 81.38 | **82.15** | Hybrid |
| ORG | **47.74** | 44.87 | 46.63 | Primary |
| LOC | 65.80 | **71.41** | 70.78 | Rich |
| MISC | 0.00 | **27.37** | 26.42 | Rich |
| Overall | 61.26 | **62.78** | 62.57 | Rich |

**CleanCoNLL test:**

| Label | Primary | Rich | **Hybrid** | Winner |
|---|---|---|---|---|
| PER | 81.88 | 82.20 | **82.98** | Hybrid |
| ORG | **46.40** | 42.12 | 43.07 | Primary |
| LOC | 61.57 | **65.02** | 64.64 | Rich |
| MISC | 0.00 | **27.95** | 26.85 | Rich |
| Overall | 58.71 | **59.36** | 59.10 | Rich |

**Finding penting:**

1. **ORG tidak sepenuhnya pulih di Hybrid** (46.63 vs primary 47.74). Walau prompt ORG
   kembali ke single `"organization"`, prompt tetangga (`country`, `city`, `nationality`,
   `event`) masih mencuri sebagian prediksi ORG via dedup highest-score.

2. **Interpretasi**: regresi ORG di Rich **bukan** karena split prompt ORG itu sendiri,
   tapi karena **kompetisi dedupe dengan prompt LOC/MISC yang lebih spesifik**. Span seperti
   "JAPAN" yang ambigu (team vs country) lebih sering menang oleh `country` daripada
   `organization` karena prompt `country` memberi confidence lebih tinggi di zero-shot.

3. **PER diuntungkan Hybrid** (82.15 dan 82.98 best di kedua dataset) — tanpa kompetitor
   spesifik, PER `person` prompt dominan jelas.

### Kesimpulan prompt strategy

Tiga strategi punya trade-off yang beda:

| Strategi | Kelebihan | Kekurangan |
|---|---|---|
| **Primary (4 single prompts)** | ORG tertinggi (47.7) | MISC 0 %, LOC terendah |
| **Rich (13 prompts, semua split)** | MISC naik ke 27 %, LOC terbaik | ORG turun paling dalam (44.9) |
| **Hybrid (single ORG, rich LOC+MISC)** | Kompromi | Tidak mengalahkan Rich di overall; PER sedikit unggul |

**Untuk thesis defense**: gunakan semua tiga sebagai demo label-sensitivity.
Rekomendasi highest-overall = Rich. Rekomendasi highest-ORG = Primary. Hybrid sebagai
baseline "moderately engineered" yang tidak worth kompleksitasnya. Ini justru menguatkan
narasi **prompt engineering adalah dimensi ketiga fair evaluation** — setelah gold quality
dan training data quality.

### Ide lanjutan (belum dijalankan)

1. **Prompt tuning per-tipe**: coba variasi seperti `"named organization"`, `"football club"`,
   `"corporation"` untuk ORG saja.
2. **Tanpa dedup**: terima prediksi overlap, biarkan evaluasi yang memilih match terbaik.
3. **Label ensemble voting**: jalankan multiple configs, predict jika ≥2 config setuju.
4. **Fine-tune pakai rich labels**: train model untuk memahami prompt spesifik, gabung
   benefit open-type + supervised.

Untuk thesis, hasil ini sudah cukup menarik untuk dibahas di bagian **Discussion: Fair
Evaluation pada Open-type NER**. Poin utamanya: evaluation fair membutuhkan tidak hanya
gold bersih (CleanCoNLL), tapi juga **prompt engineering** yang sesuai dengan domain —
jika prompt tidak tepat, penalty bisa salah-teratribusi ke "noise dataset" padahal
sebetulnya prompt-nya yang tidak cocok.

---

Hasil experiment label engineering ditambahkan pada dokumen ini setelah run selesai.

## 6. Implikasi untuk Laporan

1. **PER tidak perlu khawatir** — GLiNER zero-shot kompetitif dengan supervised BERT.
2. **ORG dan LOC gap karena ambiguity domain-specific** — "JAPAN sebagai tim" butuh
   training data CoNLL-style. Ini bisa dikurangi dengan label engineering lebih spesifik
   (sports team, country) atau fine-tune kecil.
3. **MISC adalah limitasi intrinsik open-type zero-shot** — kategori yang didefinisikan
   secara eksklusi tidak bisa di-prompt dengan satu kata. Label engineering membantu
   sebagian, tapi untuk F1 tinggi butuh supervised training.
4. **Fine-tune pada CleanCoNLL menjembatani gap** — dari [finetune table](../results/finetune/finetune_table.md),
   F1 naik ke 93% (dari 61% zero-shot) dengan ~2000 training steps. Ini cara paling
   efektif mengangkat semua label sekaligus.

Dengan kata lain: **open-type zero-shot ≠ supervised replacement**. Strength-nya adalah
flexibilitas label baru tanpa re-training; weakness-nya adalah ketergantungan pada
semantik prompt yang well-defined.
