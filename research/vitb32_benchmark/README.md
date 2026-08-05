# TinyCLIP vs CLIP ViT-B/32 — side-by-side retrieval

How much retrieval quality does the 8M-parameter TinyCLIP give up against full
CLIP ViT-B/32? Both lanes index the same **108,079 Visual Genome images**; you
type one query and see the two ranked result sets next to each other.

The point is that TinyCLIP here is not a reference implementation — it is the
**Android-exact** pipeline, the same preprocessing, the same tokenizer quirk,
the same ONNX files the APK ships. So the comparison measures what the phone
will actually do, not what a Python CLIP would do.

## The one rule

Each query is embedded by **both** text encoders and matched **only** against
its own lane's image embeddings. Both are 512-d but they live in different
spaces — cross-comparing them produces confident nonsense.

| lane | image embeddings | text encoder |
|---|---|---|
| CLIP ViT-B/32 | `vg_clip_embeddings.pkl` | `sentence-transformers` `clip-ViT-B-32` (the exact model that produced them) |
| TinyCLIP | `vg_tinyclip_embeddings.pkl` | the Android-exact ONNX pipeline in `research/common/tinyclip_encoder.py` |

## Prerequisites

This workstream needs a Visual Genome corpus that is **not** in this repo
(~15 GB). Point `VG_DATA_ROOT` at a directory containing:

```
resized_224_x_224/      108,079 pre-resized .jpg
vg_clip_embeddings.pkl  CLIP ViT-B/32 embeddings, keyed "<id>.jpg"
```

```powershell
$env:VG_DATA_ROOT = "D:\data\visual_genome"
```

Per-artifact overrides `VG_RESIZED_DIR`, `VG_VIT_PKL`, `VG_TINY_PKL` also work.
Without it the scripts fail with setup instructions rather than a bare
`FileNotFoundError`. See [`../../docs/reproducibility.md`](../../docs/reproducibility.md).

## Setup

```powershell
./setup_venv.ps1
```

## Run order

| Stage | Command | Produces / checks |
|---|---|---|
| 0 | `python 01_verify_pkl_coverage.py` | every image has an embedding, and vice versa |
| 0 | `python 02_cosim_verification.py` | re-embeds 10 random images and confirms cosine ≈ 1.00 against the stored pkl — proves it really came from CLIP ViT-B/32 |
| 1 | `python build_tinyclip_assets.py` | regenerates the ONNX encoders into `research/common/assets/` |
| 2 | `python parity_check.py` | **the gate**: ONNX vs PyTorch reference cosine ≥ 0.99, plus the hand-computed tokenizer check |
| 3 | `python precompute_tinyclip.py` | `vg_tinyclip_embeddings.pkl` — multiprocessing, resumable, checkpointed |
| 3 | `python verify_tinyclip_pkl.py` | exactly 108,079 keys, all `(512,)` float32, finite, unit-norm, key set identical to the ViT-B/32 pkl |
| 4 | `python app.py` | the Gradio UI at http://127.0.0.1:7862 |

`GRADIO_SHARE`, `GRADIO_HOST`, `GRADIO_PORT`, `GRADIO_ROOT_PATH` are respected.

## Regenerating the ONNX assets — read first

`build_tinyclip_assets.py` **overwrites committed files**. ONNX export is not
byte-deterministic, so even an identical model produces different bytes and
`ASSETS.sha256` will then fail. That is the guard working, not a bug. After a
rebuild:

```bash
python ../common/parity_assets.py
```

```bash
python ../common/verify_assets.py --write
```

Check parity **before** accepting the new hashes. If parity fails, the new
export is a different model and every threshold in `../lvis_calibration/results/`
was calibrated against the old one.

## The tokenizer quirk

`tinyclip_encoder.tokenize()` prepends BOS and appends EOT to ids that
*already* contain BOS/EOT — a double wrap. This is deliberate: it mirrors
`Tokenizer.kt` exactly. "Fixing" it silently invalidates the LVIS calibration
and the RAM comparison, both of which were computed with it in place.

## Files

| file | role |
|---|---|
| `config.py` | paths, `sys.path` wiring to `research/common`, `HF_HOME` |
| `build_tinyclip_assets.py` | regenerates the three ONNX encoders |
| `parity_check.py` | ONNX vs PyTorch reference + tokenizer gate |
| `precompute_tinyclip.py` | embeds all 108K images, resumable |
| `verify_tinyclip_pkl.py` | completeness/alignment gate on the output pkl |
| `01_verify_pkl_coverage.py` | image-vs-pkl coverage in both directions |
| `02_cosim_verification.py` | proves the ViT-B/32 pkl's provenance |
| `app.py` | the dual-lane Gradio UI |

## Known limitations

1. **No quantitative retrieval metric.** This is a qualitative side-by-side —
   there is no ground-truth relevance set for Visual Genome captions here, so it
   answers "does it look worse?" not "how much worse, in nDCG". The quantitative
   work lives in `../lvis_calibration/` and `../ram_comparison/`.
2. **The corpus is not distributable**, so this is the one workstream a fresh
   clone cannot run end-to-end without sourcing data separately.
3. **CPU only.** Embedding 108K images takes hours; the script is resumable for
   exactly that reason.
