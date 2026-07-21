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
| 6 | `python 06_per_tag_metrics.py` | `data/calibration/per_tag_metrics.csv` — per-tag confusion + metrics at the naive effective threshold |
| 7 | `python 07_balanced_calibration.py` | `data/calibration/balanced_thresholds.npy` + `balanced_per_tag_metrics.csv` — balanced-subset calibration (v2) |

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

## Results (v1 full run, LVIS train)

First end-to-end run: 100,169 image embeddings (vision fp32) × 1,203 tag
classifiers (int8 text, prompt/description/question ensemble), Fβ=0.816 sweep.

- **Common concrete tags calibrate well**: zebra Fβ 0.89 (P 0.96 / R 0.81),
  giraffe 0.89, elephant 0.83, pizza 0.72, airplane 0.74.
- **Aggregate is low**: mean per-tag Fβ ≈ 0.083 (mean P 0.10 / R 0.31). Bucket
  thresholds: high 0.40, medium 0.42, weak 0.44; global 0.41.
- **Prompt-ensembling ablation is flat**: prompts 0.0815 / descriptions 0.0893 /
  questions 0.0835 / combined 0.0825 (macro Fβ) — differences are noise.

### Why the aggregate is low (both effects are real, not bugs)

1. **Naive-negative base-rate ceiling.** With positives-only GT and implicit
   negatives by absence, a rare tag has ~1–50 positives against ~100K implicit
   negatives, so precision is structurally floored near 0 for rare tags. This
   drags the mean down and masks any text-side signal. (zebra P 0.96 confirms
   false-positive counting is correct — the effect is the metric, not a bug.)
2. **TinyCLIP-8M is weak on fine-grained LVIS.** It nails COCO-scale objects but
   cannot separate e.g. `halter_top` vs `tank_top`. Prompt ensembling (which
   helped full CLIP on ImageNet top-1) gives ~nothing here: the bottleneck is the
   tiny vision encoder plus a precision-floored metric, and per-source text
   vectors are near-duplicates so averaging barely moves them.

The per-tag `thresholds.npy` + `calibration_table.json` are usable as-is,
especially for the `high`-confidence bucket.

### Balanced calibration (v2 — `07_balanced_calibration.py`)

To remove the base-rate floor, v2 calibrates each tag on a **balanced per-tag
subset**: all P positives + P random (seeded) negatives = 2P images, sweep Fβ on
that. This makes precision meaningful and lifts mean weighted-F1 from 0.083 to
**~0.80** (high 0.77 / medium 0.79 / weak 0.86). Common tags reach ~0.95+
(zebra 0.968, giraffe 0.975).

Caveats: (1) rare-tag subsets are tiny — 222 tags have ≤10 images, so their
thresholds/metrics are noise and inflate the `weak` mean; trust results in
proportion to `subset_total` (median 74). (2) Balanced metrics are optimistic vs
real base-rate deployment, and the balanced thresholds run lower (zebra 0.36 vs
naive 0.41), so production precision will be lower than the balanced CSV shows —
this is an intrinsic-separability measure, not a deployment estimate.

## Future work (deferred)

- **Federated-correct negatives** — for tag X, only count an image as a negative
  if X is in its `neg_category_ids` or the image exhaustively annotates X
  (`not_exhaustive_category_ids`). Shrinks the negative pool for rare tags so
  precision/Fβ become meaningful. Highest-value next change; embeddings are
  already computed so only `03`/`04`/`05` re-run.
- **Per-tag ranking metric (AP / top-k)** to measure ordering quality independent
  of the threshold precision floor.
- **LLM-generated descriptions/questions** — `strings.jsonl` + the source-grouped
  design already accept a `source="llm"` batch.
