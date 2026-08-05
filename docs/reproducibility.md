# Reproducibility

Clone-to-result instructions for every workstream in this repo.

Sections are ordered by cost. The first two need nothing but a clone. The rest
need downloads measured in gigabytes, and one needs a corpus this repo cannot
distribute.

**Conventions:** commands assume the repo root as the working directory.
PowerShell is shown because the work was done on Windows; the bash equivalents
differ only in path separators and venv activation.

---

## 0. What you get for free from a clone

Committed on purpose, so these work immediately:

| | |
|---|---|
| The Android app | builds and runs — the ONNX encoders are committed |
| Per-tag LVIS thresholds | `research/lvis_calibration/results/` |
| The RAM++ tag mapping and benchmark table | `research/ram_comparison/results/` |
| The 4,585-tag LLM descriptions | `research/ram_comparison/assets/` |

Verify the clone is intact:

```bash
python research/common/verify_assets.py
```

Expected: `All 6 assets match, and every copy agrees.`

---

## 1. Android app

Requirements: Android Studio Hedgehog+, JDK 17+, SDK 35, device/emulator API 26+.

`local.properties` is gitignored (the SDK path is machine-specific), so a fresh
clone needs it before Gradle will run. Android Studio writes it when you open
the project; otherwise:

```bash
echo "sdk.dir=/path/to/Android/Sdk" > local.properties
```

```bash
./gradlew installDebug
```

That is the only setup step — `app/src/main/assets/*.onnx` is committed, so
there is no model export to run.

---

## 2. Prerequisites for the research workstreams

Python **3.11**. Two virtualenvs are required — see
[`../research/README.md`](../research/README.md) for why they cannot be merged.

```powershell
cd research/vitb32_benchmark; ./setup_venv.ps1
```

```powershell
cd research/ram_comparison; ./setup_venv.ps1
```

`venv_clip` also serves `lvis_calibration`.

**Corporate TLS:** if your network intercepts certificates, every HuggingFace
fetch fails with `CERTIFICATE_VERIFY_FAILED`. `research/common/ssl_bypass.py`
handles it and is imported automatically. If you would rather not disable
verification, set `HF_HUB_OFFLINE=1` and pre-populate the cache.

---

## 3. LVIS threshold calibration

**Cost:** ~1 GB annotations + ~100K COCO images (~20 GB), several hours of CPU.
**Skip it** if you only want the thresholds — they are already committed.

```powershell
cd research/lvis_calibration
..\vitb32_benchmark\venv_clip\Scripts\Activate.ps1
```

| Step | Command | Output |
|---|---|---|
| — | `python lvis_meta.py` | sanity: 1203 tags, buckets weak 337 / medium 461 / high 405 |
| 0 | `python 00_download_lvis.py` | `data/annotations/`, `data/images/train2017/` |
| 1 | `python 01_build_text_classifiers.py` | `data/text/` classifiers + `strings.jsonl` |
| 2 | `python 02_precompute_image_embeddings.py` | `data/image_embeds/img_matrix.npy` (205 MB) |
| 3 | `python 03_calibrate_thresholds.py` | `thresholds.npy`, `calibration_table.json` |
| 4 | `python 04_report.py` | printed Fβ-by-bucket summary |
| 5 | `python 05_ablation.py` | per-source vs combined Fβ |
| 6 | `python 06_per_tag_metrics.py` | naive-regime per-tag CSV |
| 7 | `python 07_balanced_calibration.py` | `balanced_thresholds.npy` + balanced CSV |

Most scripts accept `--limit N` for a fast subset dry run.

**Expected results** (β = 0.5, set by `config.FBETA`): balanced regime mean
precision **0.825**, recall **0.767**, weighted-F1 **0.800**. If you reproduce
and get materially different numbers, check `config.FBETA` first — the v1 run
used 0.816 and its numbers are not comparable.

Steps 3–7 read from `data/`, and write to `data/calibration/`. The committed
`results/` folder is a copy of those outputs; re-running does not update it
automatically.

---

## 4. RAM++ vs TinyCLIP comparison

**Cost:** 2.8 GB checkpoint, ~29 min to build the template classifiers.

```powershell
cd research/ram_comparison
```

```powershell
python 00_download_ram.py
```

Downloads the checkpoint (resumable) and clones `recognize-anything` into
`vendor/`. Then:

| Step | Command | Notes |
|---|---|---|
| 1 | `python 01_build_text_classifiers.py --source templates` | ~29 min, 403K strings |
| 1 | `python 01_build_text_classifiers.py --source llm` | fast — 20 strings/tag |
| 2 | `python 02_build_tag_mapping.py` | LVIS→RAM mapping + `thresholds_4585.npy` |
| — | `./venv_ramclip/Scripts/python.exe verify_env.py --full` | preflight; `--full` loads the checkpoint |
| — | `./venv_ramclip/Scripts/python.exe app.py` | UI at http://127.0.0.1:7863 |

Step 2 reads the committed `../lvis_calibration/results/`, so it does **not**
require reproducing section 3.

The quantitative A/B:

```bash
./venv_ramclip/Scripts/python.exe 03_benchmark_classifiers.py --combined --regime balanced
```

This one **does** need `../lvis_calibration/data/image_embeds/` — it scores both
classifier sets against real LVIS ground truth, so section 3 steps 0–2 are a
prerequisite.

**Expected:** templates macro Fβ 0.7955, llm 0.8098, combined 0.8111. The
templates figure reproducing the LVIS balanced calibration (0.800) is the
cross-check that the harness is wired correctly.

### Adding new descriptions

The committed `assets/clip_descriptions_4585.json` is the canonical set. To
extend it, drop new JSON into `data/llm_desc/` and run:

```bash
python ingest_descriptions.py
```

It reports coverage across the committed asset *plus* your drops. Expect
`4585/4585 tags, 10.0 descriptions per tag` from the committed file alone —
if you see 20.0, you have a duplicate of the committed file in the drop zone.

---

## 5. TinyCLIP vs CLIP ViT-B/32 benchmark

**This is the one section a fresh clone cannot complete unaided.** It needs a
Visual Genome corpus (~15 GB) that this repo cannot redistribute.

You need a directory containing:

```
resized_224_x_224/      108,079 images, pre-resized to 224x224
vg_clip_embeddings.pkl  CLIP ViT-B/32 embeddings, keyed "<id>.jpg"
```

Build both from the [Visual Genome release](https://homes.cs.washington.edu/~ranjay/visualgenome/):
resize each image to 224×224, then embed with `sentence-transformers`
`clip-ViT-B-32` and pickle a `{filename: vector}` dict.

Then point the scripts at it:

```powershell
$env:VG_DATA_ROOT = "D:\data\visual_genome"
```

```bash
export VG_DATA_ROOT=/data/visual_genome
```

Overrides `VG_RESIZED_DIR`, `VG_VIT_PKL`, `VG_TINY_PKL` take precedence.
If unset and the default is absent, the scripts fail with these instructions
rather than a bare traceback.

Run order is in [`../research/vitb32_benchmark/README.md`](../research/vitb32_benchmark/README.md).
`02_cosim_verification.py` is the gate that proves your pkl really came from
CLIP ViT-B/32 (expect cosine ≈ 1.00 ± 0.02).

---

## 6. Regenerating the ONNX encoders

Rarely needed — they are committed. If you do:

```bash
cd research/vitb32_benchmark && python build_tinyclip_assets.py
```

```bash
python research/common/parity_assets.py
```

```bash
python research/common/verify_assets.py --write
```

**Run parity before accepting the new hashes.** ONNX export is not
byte-deterministic, so `verify_assets.py` will fail after any rebuild even when
the model is identical — that is expected. What is *not* acceptable is parity
failing: that means the new export is a different model, and every threshold in
`research/lvis_calibration/results/` was calibrated against the old one.

The known-good result is text bit-identical (max elementwise diff exactly 0.0)
and vision within 4.2e-07 (~3 float32 ulp).

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: tinyclip_encoder` / `ssl_bypass` / `templates` | You imported before `config`. `config` is what puts `research/common` on `sys.path` — import it first. |
| `CERTIFICATE_VERIFY_FAILED` during model load | `ssl_bypass` was imported too late. It must precede transformers/gradio, which connect at import time. |
| `ImportError: apply_chunking_to_forward` | You are running RAM in `venv_clip`. It needs `venv_ramclip` (transformers 4.44). |
| `AttributeError: scipy.interpolate.interp2d` | SciPy ≥ 1.14 in `venv_ramclip`. Pin 1.13.1. |
| `verify_assets.py` reports MISMATCH | Someone regenerated the ONNX. Run `parity_assets.py` before trusting or re-writing the manifest. |
| Models re-download every run | Stale `HF_HOME`. It is pinned per workstream in `config.py`; an exported `HF_HOME` overrides it. |
| `Position interpolate ... 23x23 to 13x13` on RAM load | Expected. The checkpoint is a 384 model run at 224; its per-tag thresholds were tuned at 384, so they are approximate. Switch to 384 in the UI to remove the confound. |
