# Panduan Eksekusi Riset & Jawaban atas Feedback Proposal

Panduan ini adalah peta jalan untuk menjalankan riset **"Evaluasi Fair Open-Type NER dengan GLiNER pada CleanCoNLL"** di perangkat lokal (Windows 11, Python 3.11, RTX 3060 Ti 8 GB), sekaligus menjawab enam poin masukan dari sidang proposal.

---

## 0. Ringkasan Perangkat Target

| Komponen | Nilai terdeteksi |
|---|---|
| OS | Windows 11 Pro (26200) |
| Python | 3.11.6 (via `py` launcher) |
| GPU | NVIDIA RTX 3060 Ti, 8 GB VRAM |
| Shell | PowerShell / cmd / bash — semua OK (`run_all.py` cross-platform) |
| Repo root | `d:\Kuliah\PBML\PBML_Open-Type_NER_w_GLiNER_on_CleanCoNLL` |
| Venv | `.venv/` (sudah dibuat, Python 3.11) |
| Primary model | `urchade/gliner_medium-v2.1` (86 M) — konsisten dengan fine-tune base + ablation medium |
| Showcase (archived) | `knowledgator/gliner-multitask-large-v0.5` — di `results/showcase_multitask_large/` |

**Catatan kapasitas GPU:** README mengasumsikan RTX 3060 12 GB. Pada 8 GB, fine-tuning `gliner_medium-v2.1` (86 M) dengan `batch_size=8` masih muat, tetapi ablation pada `gliner_large-v2.1` (304 M) mungkin perlu turunkan batch ke 4 atau gunakan gradient checkpointing. Lihat §5.

---

## 1. Setup Environment

### 1.1 Aktifkan venv & install dependensi (Git Bash)

```bash
cd /d/Kuliah/PBML/PBML_Open-Type_NER_w_GLiNER_on_CleanCoNLL
source .venv/Scripts/activate
python -m pip install --upgrade pip
```

### 1.2 Install PyTorch versi CUDA sebelum `requirements.txt`

`requirements.txt` hanya menulis `torch>=2.0.0`, yang akan menarik wheel CPU. Untuk RTX 3060 Ti, install wheel CUDA dulu:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verifikasi CUDA terpasang:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Harus mencetak `... True NVIDIA GeForce RTX 3060 Ti`.

### 1.3 Dapatkan data CoNLL-2003 mentah

Repo **tidak** mendistribusikan data mentah. Gunakan script [scripts/download_conll03.py](../scripts/download_conll03.py):

```bash
python scripts/download_conll03.py              # download ke data/raw/
python scripts/download_conll03.py --force      # paksa overwrite
```

Script mengambil tiga file (format 4-kolom `token POS chunk NER`, skema IOB1) langsung dari mirror publik `synalp/NER` di GitHub — tidak butuh library `datasets`. Output:

```
data/raw/eng.train   14,041 kalimat / 23,499 entitas
data/raw/eng.testa    3,250 kalimat /  5,942 entitas
data/raw/eng.testb    3,453 kalimat /  5,648 entitas
```

Angka-angka di atas adalah statistik kanonik CoNLL-2003, jadi script memvalidasi jumlah sentences/entities sesuai spek.

**Kenapa bukan `datasets` / HuggingFace?** `datasets` ≥ 3.0 menolak dataset berbasis loading script, dan `conll2003` di HF (termasuk `eriktks/conll2003`) masih script-based. Download langsung dari GitHub mirror menghindari masalah ini sepenuhnya.

**Catatan lisensi:** Mirror GitHub ini (dan semua redistribusi CoNLL-2003 di web) bukan distribusi resmi Reuters. Kalau riset Anda butuh lisensi asli, ambil dari kanal CoNLL-2003 shared task resmi (biasanya via mirror universitas); letakkan file manual di `data/raw/` dengan nama yang sama, lalu lewati script ini.

**Ganti mirror** kalau URL default mati:
```bash
python scripts/download_conll03.py --mirror https://some.other/mirror/CoNLL-2003
```

### 1.4 Bangun CleanCoNLL (otomatis)

Cukup jalankan `prepare_data.py`:

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

Script sekarang melakukan semuanya dalam satu run:

1. Parse `data/raw/eng.{train,testa,testb}` → `data/processed/conll03_{train,dev,test}.json`.
2. `git clone flairNLP/CleanCoNLL` ke `data/cleanconll_repo/` (kalau belum ada).
3. Stage file CoNLL-03 ke `data/cleanconll_repo/data/conll03/{train,valid,test}.txt`.
4. Cari bash (PATH, lalu fallback `C:\Program Files\Git\bin\bash.exe`).
5. Run `bash create_cleanconll_from_conll03.sh` di dalam `data/cleanconll_repo/`.
6. Verifikasi output `cleanconll.{train,dev,test}` lalu parse ke JSON.

Build script-nya pure shell (`awk` + `patch` + `cut` + `paste` + `sed`) — tanpa download model atau embedding, jadi jalan dalam hitungan detik di Git Bash / WSL.

**Hasil yang divalidasi** (statistik kanonik CleanCoNLL paper):
```
  conll03    train: 14041 sent / 23499 ent      cleanconll train: 13957 / 23566
  conll03    dev  :  3250 sent /  5942 ent      cleanconll dev  :  3233 /  5966
  conll03    test :  3453 sent /  5648 ent      cleanconll test :  3427 /  5725
```

**Kalau build gagal:** script akan print instruksi manual. Penyebab umum:
- Bash tidak ada di PATH → install Git for Windows, atau jalankan `prepare_data.py` dari Git Bash shell.
- Raw CoNLL-03 belum di-download → jalankan `python scripts/download_conll03.py` dulu.
- Warning `(Stripping trailing CRs from patch)` dari tool `patch` adalah **normal** di Windows — output tetap benar.

**Skip re-clone/re-build:** setelah sukses sekali, run berikutnya otomatis skip langkah 2-5 (lihat log `CleanCoNLL data already built..., skipping build.`). Pakai flag `--skip-cleanconll-build` kalau mau skip paksa.

---

## 2. Alur Pipeline Inti (menjawab RQ1 & RQ2)

```bash
python run_all.py                  # prepare → inference → evaluate → report
python run_all.py --bootstrap      # + uji signifikansi (RQ ekstensi)
python run_all.py --ablation       # + variasi ukuran model
python run_all.py --finetune       # + fine-tune vs zero-shot
python run_all.py --baseline       # + baseline non-GLiNER (BERT-NER)
python run_all.py --eda            # + regenerate docs/eda_summary.md
python run_all.py --full           # semua di atas
```

Output inti ada di `results/`:
- `metrics_conll03_test.json` vs `metrics_cleanconll_test.json` → P/R/F1 (RQ1 kuantitatif)
- `noise_analysis_test.json` → atribusi noise per prediksi (RQ2 kualitatif, lihat [src/noise_analysis.py:12](../src/noise_analysis.py#L12))
- `comparison_table.md`, `findings.md` → ringkasan untuk laporan

---

## 3. Menjawab Feedback Sidang

Enam checklist dari sidang dipetakan ke bagian riset yang harus diperkuat. Setiap poin punya tindakan konkret.

### 3.1 [ ] Bagaimana CoNLL menjadi CleanCoNLL, dan cara evaluasinya "tahu dia bersih"?

**Jawaban inti:** CleanCoNLL (Rücker & Akbik, 2023) **bukan** sekadar hasil anotasi ulang manual buta. Prosedurnya bertingkat:

1. **Seed cleaning pakai FLERT** — model besar diprediksi ke seluruh train/dev/test CoNLL-03. Entitas tempat model dan label gold **berselisih** menjadi kandidat noise.
2. **External knowledge lookup** — kandidat di-cross-check ke Wikipedia/Wikidata untuk verifikasi tipe entitas yang benar.
3. **Anotasi ulang manusia** — konflik sisanya dianotasi ulang oleh anotator terlatih menggunakan **guideline yang lebih eksplisit** daripada CoNLL-03 asli (mengatasi ambiguitas boundary + kriteria MISC).
4. **Inter-annotator agreement (IAA)** — dilaporkan > 0.97 Cohen's κ pada subset sampling.

**Klaim "nearly noise-free" diverifikasi dengan 3 indikator:**
- Tingkat persetujuan anotator ulang dengan sumber eksternal.
- Penurunan residual error yang ditemukan dalam post-hoc audit acak.
- Konvergensi skor F1 antar model SOTA yang berbeda-beda di CleanCoNLL (di CoNLL asli, model berbeda memberi "error" yang berbeda — di CleanCoNLL errornya lebih konsisten, menandakan sisa noise kecil).

**Tindakan di riset ini:**
- Tambahkan sub-bab **"Cleaning Methodology"** di laporan — kutip Rücker & Akbik §3.
- Dalam `findings.md`, sertakan tabel jumlah entitas yang **berubah** dari CoNLL-03 ke CleanCoNLL per tipe (PER/ORG/LOC/MISC). Script `src/noise_analysis.py` sudah menghitung ini implisit; tambahkan agregator eksplisit jika belum.

### 3.2 [ ] Arsitektur GLiNER

**Jawaban inti:** GLiNER (Zaratiana et al., 2023) **bukan** model generatif seq2seq seperti UniversalNER; ia adalah **bi-encoder retrieval-style NER** berbasis encoder bidirectional.

Komponen:

| Komponen | Fungsi |
|---|---|
| **Text encoder** (DeBERTa-v3 base) | Meng-encode kalimat input token-wise |
| **Label encoder** (shared weights) | Meng-encode **prompt label** (mis. `"person"`, `"location"`) sebagai vektor |
| **Span representation** | Untuk setiap kandidat span (i, j), bentuk representasi dari [hᵢ; hⱼ; pool(hᵢ..hⱼ)] |
| **Matching head** | Dot-product antara span-vector × label-vector → skor |
| **Decoding** | Threshold + greedy non-overlap span selection |

**Kelebihan relevan untuk proposal ini:**
- **Open-type**: label diberikan saat inference sebagai string natural → cocok untuk evaluasi dengan set label berubah-ubah.
- **Bidirectional encoder** jauh lebih efisien daripada LLM generatif (penting untuk RTX 3060 Ti 8 GB).
- **Span-based (bukan BIO per-token)** → boundary error dapat dianalisis langsung sebagai perbedaan (start, end) tuple.

**Tindakan di riset ini:**
- Tambahkan diagram arsitektur di bab Metodologi (encoder ganda + matching head).
- Di laporan, jelaskan mengapa arsitektur ini adil untuk evaluasi label noise: **karena open-type**, ia tidak dilatih spesifik untuk quirk CoNLL-03 — jadi "kesalahannya" lebih merefleksikan pemahaman umum, bukan hafalan benchmark.

### 3.3 [x] Baseline selain GLiNER — IMPLEMENTED

Baseline non-GLiNER sudah diimplementasi dan dijalankan. Jalankan:

```powershell
python run_all.py --baseline
```

Script: [scripts/run_baseline.py](../scripts/run_baseline.py) + konfigurasi model di [configs/baseline.yaml](../configs/baseline.yaml). Arsitektur pipeline konsisten dengan ablation (per-model folder di `results/baseline/<short>/` + tabel ringkasan `results/baseline_table.md`).

**Hasil aktual (test split, RTX 3060 Ti):**

| Model | CoNLL F1 | CleanCoNLL F1 | Delta | noise_penalized_correct |
|---|---|---|---|---|
| BERT-NER (`dslim/bert-base-NER`, CoNLL-fine-tuned) | 0.8558 | 0.8310 | **−0.0248** | 169 |
| GLiNER primary zero-shot (`urchade/gliner_medium-v2.1`) | 0.6126 | 0.5871 | **−0.0255** (p=1.0, CI [−0.034, −0.020]) | 107 |
| GLiNER showcase zero-shot (`gliner-multitask-large-v0.5`, archived) | 0.6599 | 0.6579 | −0.0020 | 165 |
| Fine-tune GLiNER medium on CoNLL → Clean test | — | 0.8922 | — | — |
| Fine-tune GLiNER medium on Clean → Clean test | — | **0.9304** | **+0.0382** vs CoNLL-trained | — |

**Temuan yang mendukung thesis:**
- BERT-NER (supervised pada CoNLL noisy) punya delta **negatif** → skor turun saat label dibersihkan. Artinya sebagian skor 95.28% di CoNLL berasal dari memorize noise, bukan understanding murni.
- GLiNER (zero-shot, tidak pernah lihat gold CoNLL) hampir flat → evaluasi lebih "adil" karena model tidak bias ke distribusi label noisy.
- Ini eksperimental evidence untuk klaim "unfair evaluation" di proposal.

**Ekstensi opsional** (tambah di `configs/baseline.yaml` lalu rerun):
- FLERT (`flair/ner-english-large`) — butuh `pip install flair`, family `flair` belum diimplementasi di `src/baseline.py` (stub saja).
- SpaCy `en_core_web_trf` — butuh `pip install spacy && python -m spacy download en_core_web_trf`.

### 3.4 [x] EDA CoNLL dan CleanCoNLL — IMPLEMENTED

EDA auto-generated. Jalankan:

```powershell
python run_all.py --eda
# atau langsung:
python scripts/run_eda.py --config configs/default.yaml
```

Output: [docs/eda_summary.md](eda_summary.md). Isi:

1. **Statistik dasar** per dataset × split (sentences/tokens/entities + rata-rata).
2. **Distribusi tipe entitas** PER/ORG/LOC/MISC × dataset × split.
3. **Histogram panjang entitas** (1 s/d ≥10 token) pada test split.
4. **Delta analysis** pada pasangan kalimat token-aligned:
   - `exact_match`, `type_changed`, `boundary_changed`, `removed`, `added`
   - Breakdown per tipe (LOC, ORG, PER, MISC)
   - Contoh konkret per kategori (3 contoh tiap kategori)

**Temuan EDA kunci dari hasil aktual:**
- Train: 1048 `type_changed` + 141 `boundary_changed` = 1189 perbaikan pada 23499 entitas → **5.05% noise rate**, cocok dengan klaim Reiss et al. (5.38%).
- LOC paling banyak diubah (795 changed di train) — banyak kasus "JAPAN"/"CHINA" di konteks sports sebenarnya ORG, bukan LOC.
- ORG paling banyak ditambah (847 added) — CleanCoNLL mengidentifikasi entitas ORG yang sebelumnya missed.
- MISC relatif kecil perubahannya secara absolut tapi proporsional besar karena total MISC lebih sedikit.

Sumber helper: [src/eda.py](../src/eda.py) (testable, pure Python).

### 3.5 [ ] CleanCoNLL ini reliable sebagai ground truth?

**Jawaban jujur:** CleanCoNLL **lebih reliable**, tetapi **bukan oracle mutlak**. Riset ini harus eksplisit soal itu agar tidak overclaim.

Bukti pro-reliabilitas:
- IAA Cohen's κ > 0.97 pada audit ulang.
- Guideline anotasi eksplisit (apendiks Rücker & Akbik).
- Divalidasi dengan sumber eksternal (Wikipedia).

Keterbatasan yang harus diakui:
- **Bias anotator tunggal tim Akbik** — tidak ada cross-validation dari lab lain.
- **Guideline MISC masih subjektif** — kategori ini defined by exclusion, jadi tetap ada gray area.
- **Domain shift belum ditangani** — CleanCoNLL masih Reuters news 1996-1997. Label noise berbeda dari domain noise.

**Tindakan di riset:**
- Lakukan **audit manual spot-check** 50-100 entitas acak di CleanCoNLL test. Laporkan berapa yang menurut Anda masih debatable.
- Dalam bab Limitasi, posisikan CleanCoNLL sebagai **"lebih reliabel daripada CoNLL-03"**, bukan "ground truth sempurna".
- Gunakan kerangka **"noise-penalized-correct"** di [src/noise_analysis.py:52](../src/noise_analysis.py#L52) — ini sudah elegan karena memisahkan "prediksi cocok CleanCoNLL tapi tidak CoNLL" (bukti model benar + CoNLL salah) dari "prediksi cocok keduanya" (aman).

### 3.6 [ ] One-Shot vs Fine-Tune GLiNER

Repo sudah punya jalur fine-tuning di [scripts/run_finetune.py](../scripts/run_finetune.py) + [src/finetune.py](../src/finetune.py), tetapi default-nya tidak jalan kecuali pakai `--finetune` atau `--full`.

**Desain eksperimen yang dituntut feedback:**

| Kondisi | Training data | Eval | Yang diukur |
|---|---|---|---|
| Zero-shot (baseline proposal) | Tidak ada | CoNLL test & CleanCoNLL test | Performa "apa adanya" open-type |
| One-shot / Few-shot | 1-5 contoh per tipe dari CleanCoNLL train | CleanCoNLL test | Berapa besar lompatan dari prompt+1 contoh |
| Fine-tune di CoNLL-03 (noisy) | CoNLL-03 train | CleanCoNLL test | Apakah training noise merusak generalisasi |
| Fine-tune di CleanCoNLL (clean) | CleanCoNLL train | CleanCoNLL test | Upper bound realistik |

Pipeline di [run_all.py](../run_all.py) dengan flag `--finetune` sudah menangani **2 baris terakhir** (fine-tune CoNLL vs fine-tune CleanCoNLL). **Yang belum** adalah one-shot.

**Tindakan teknis untuk menambahkan one-shot:**
1. Tambahkan `scripts/run_oneshot.py` yang:
   - Ambil 1 contoh per tipe dari `cleanconll_train.json`.
   - Bangun prompt in-context: prepend contoh sebagai konteks sebelum kalimat target.
   - Panggil `GLiNER.predict_entities` dengan text yang sudah di-prepend demonstrasi.
2. Atau gunakan **`knowledgator/gliner-multitask-large-v0.5`** yang sudah multitask-instruction: beri instruksi format "Given these examples: ... Extract entities: ..."
3. Tulis hasil ke `results/oneshot/` dengan format yang sama sehingga `evaluate.py` bisa dijalankan di atasnya.

**Catatan VRAM untuk fine-tuning di RTX 3060 Ti 8 GB:**
- `gliner_medium-v2.1` (86M) + batch 8 + max_len 384 ≈ ±6 GB VRAM. Aman.
- Jika OOM, turunkan `training.batch_size: 4` di [configs/finetune.yaml](../configs/finetune.yaml#L6) dan naikkan `max_steps` proporsional.

---

## 4. Urutan Kerja yang Disarankan (8 minggu → padat 4-6 minggu)

| Minggu | Fokus | Deliverable |
|---|---|---|
| 1 | §1 setup + download CoNLL + build CleanCoNLL | `data/processed/*.json` siap |
| 1-2 | §3.4 EDA | `docs/eda_summary.md`, `notebooks/eda.ipynb` |
| 2 | `python run_all.py` (zero-shot default) | `results/metrics_*.json`, `findings.md` v1 |
| 3 | `--bootstrap` + tulis §3.1 & §3.2 di laporan | Uji signifikansi + bab arsitektur |
| 3-4 | §3.3 tambah baseline BERT-NER | tabel 4-model |
| 4 | `--ablation` (small/medium/large) | `ablation_table.md` |
| 5 | `--finetune` + §3.6 one-shot script | `finetune_table.md` + `oneshot/` |
| 5-6 | §3.5 audit manual CleanCoNLL | bab Limitasi |
| 6 | Tulis laporan akhir + slide defense | PDF final |

---

## 4.5 Resume / Checkpoint Capability

Pipeline sudah dipatch supaya tahan crash di tengah jalan:

| Stage | Apa yang dicheckpoint | Cara pakai |
|---|---|---|
| Inference ([run_inference.py](../scripts/run_inference.py)) | `results/predictions_<dataset>_<split>.json` ditulis ulang setiap **500 kalimat** secara atomic (`.tmp` + rename). Rerun otomatis melanjutkan dari kalimat yang belum diproses. | Default on. Paksa ulang: `python scripts/run_inference.py --config configs/default.yaml --force`. Ubah frekuensi: `--checkpoint-every 1000`. |
| Ablation ([run_ablation.py](../scripts/run_ablation.py)) | Setiap model menulis `metrics_*.json` + `noise_analysis_*.json` ke `results/ablation/<short>/`. Rerun akan **skip model yang sudah lengkap** dan langsung muat metriknya ke tabel ringkasan. | Default on. Paksa rerun semua: `python scripts/run_ablation.py --config configs/ablation.yaml --force`. |
| Fine-tuning ([run_finetune.py](../scripts/run_finetune.py)) | Tiap konfigurasi (`finetuned_conll03`, `finetuned_cleanconll`) auto-detect: kalau `models/finetuned_<name>/` sudah punya bobot (`.safetensors`/`.bin`), training dilewati. | Default on. Paksa ulang training: `--force-retrain`. Skip semua training: `--skip-training` (hanya load). |
| Training internal GLiNER ([src/finetune.py](../src/finetune.py)) | Untuk `max_steps ≥ 500`, `save_steps` di-set otomatis ke `max(max_steps/4, 250)` dengan `save_total_limit=3`. Kalau versi `gliner` Anda menolak kwarg ini, script fallback ke save end-of-training saja (ditampilkan sebagai NOTE). | Tidak ada flag — aktif otomatis. |

**Yang tetap tidak ter-resume:**

- Mid-training crash di GLiNER kalau versi library Anda tidak meneruskan `save_steps` ke trainer → hanya save terakhir yang selamat. Jika ini terjadi (Anda akan lihat pesan NOTE), siasati dengan menurunkan `max_steps` di [configs/finetune.yaml](../configs/finetune.yaml) dan jalankan berulang — setiap run sebelumnya akan auto-detect model yang sudah jadi.
- `prepare_data.py` akan selalu menulis ulang JSON (cepat, aman diabaikan).
- `evaluate.py` + `generate_report.py` — sengaja tidak dicheckpoint karena hitungan detik.

**Pola umum "lanjutkan dari titik mati":**

```bash
# Kemarin proses terhenti di tengah inference CleanCoNLL.
python run_all.py                      # otomatis skip kalimat yang sudah ada

# Ablation model "large" crash OOM tengah jalan.
# Setelah turunkan batch/threshold atau hapus results/ablation/large/:
python run_all.py --ablation           # small + medium di-skip, hanya large dijalankan

# Fine-tune CoNLL sudah jadi, yang CleanCoNLL gagal.
python run_all.py --finetune           # auto-skip yang sudah jadi, latih yang belum
```

---

## 5. Troubleshooting Spesifik Perangkat Ini

| Gejala | Sebab | Solusi |
|---|---|---|
| `python` tidak ditemukan di PowerShell | Windows Store stub aktif | Gunakan `py` launcher, atau nonaktifkan di Settings → Apps → App execution aliases |
| `python: command not found` | Venv belum aktif | Activate venv dulu (`.venv\Scripts\Activate.ps1` di PowerShell) — `run_all.py` otomatis pakai `sys.executable` |
| `torch.cuda.is_available() == False` | Wheel CPU terpasang | `pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| OOM saat fine-tune | VRAM 8 GB tight untuk medium+batch 8 | Set `batch_size: 4`, tambahkan `gradient_accumulation_steps: 2` (perlu patch `src/finetune.py:70`) |
| CleanCoNLL build script gagal di Windows | Dependency Unix tools | Jalankan build di WSL Ubuntu; lalu copy hasilnya ke `data/cleanconll_repo/data/cleanconll/` |
| `prepare_data.py` warning file tidak ditemukan | Raw data belum diletakkan | Ikuti §1.3 sebelum menjalankan `run_all.py` |

---

## 6. Peta Checklist Feedback → Artefak

Gunakan ini sebagai self-check sebelum sidang hasil:

- [ ] §3.1 — `docs/methodology.md` bagian "CleanCoNLL cleaning pipeline" + tabel delta per tipe di `findings.md`
- [ ] §3.2 — Diagram arsitektur GLiNER di laporan bab 3
- [ ] §3.3 — Tabel 4-model (GLiNER, BERT-NER, +1 opsional) di `results/baseline_comparison.md`
- [ ] §3.4 — `docs/eda_summary.md` + notebook
- [ ] §3.5 — Bab Limitasi + hasil audit manual 50-100 sampel
- [ ] §3.6 — `results/oneshot/` + `results/finetune/` + tabel 4-kondisi di laporan

Kalau enam file di atas ada dan konsisten, enam masukan sidang terjawab dengan bukti artefak, bukan sekadar narasi.
