# TinyCLIP LVIS calibration

Text-side classifier construction (prompt ensembling) + per-tag cosine-threshold
calibration for **TinyCLIP-ViT-8M-16-Text-3M-YFCC15M** on **LVIS-v1 train**.

Goal: for each of the 1,203 LVIS tags, build one averaged text-classifier
embedding from many prompts/descriptions/questions, then pick the per-tag cosine
threshold on TinyCLIP image embeddings that maximizes a **precision-leaning
Fbeta** (`beta = 0.816`, ~60% precision / 40% recall).

Encoders are reused verbatim from the sibling Gradio app
(`../tinyClip_vs_ClipVit32/tinyclip_encoder.py` + `assets/*.onnx`) so results are
Android-exact: **int8** text encoder, **fp32** vision encoder, CLIP BPE tokenizer.

## Environment

Reuse the existing venv (Python 3.11):

```powershell
..\tinyClip_vs_ClipVit32\venv_clip\Scripts\Activate.ps1
pip install -r requirements.txt      # most deps already present in venv_clip
```

## Run order

| Step | Script | Output |
|---|---|---|
| 0 | `python 00_download_lvis.py` | `data/annotations/lvis_v1_train.json`, `data/images/train2017/*.jpg` |
| — | `python lvis_meta.py` | sanity: tag count + bucket histogram |
| 1 | `python 01_build_text_classifiers.py` | `data/text/` classifier `.npy/.npz` + `strings.jsonl` |
| 2 | `python 02_precompute_image_embeddings.py` | `data/image_embeds/img_matrix.npy`, `img_ids.npy` |
| 3 | `python 03_calibrate_thresholds.py` | `data/calibration/thresholds.npy`, `calibration_table.*` |
| 4 | `python 04_report.py` | printed summary of Fbeta by bucket |
| 5 | `python 05_ablation.py` | `data/calibration/ablation.json` — per-source vs combined Fbeta (optional) |

Most scripts accept `--limit N` for a fast subset dry run.

## Key decisions

- **Ground truth: positives only.** A tag's positives are the images annotated
  with it. No explicit negatives are stored and no LVIS federated neg-category
  logic is used — the false positives the precision term needs come from
  *implicit negatives by absence* (an image is negative for tag X only because X
  is not among its positive tags).
- **Threshold grid:** cosine in `[0.20, 0.65]`, step `0.01`.
- **Confidence buckets** (per-tag positive count): `weak` 1-10, `medium` 11-100,
  `high` >100. Weak tags use a bucket-level fallback threshold; all tags also
  store their own per-tag threshold plus the confidence flag.

## Artifacts

All under `data/` (gitignored). Row order is fixed by `data/text/tag_order.json`
and image row order by `data/image_embeds/img_ids.npy`. See `config.py` for the
canonical paths.
