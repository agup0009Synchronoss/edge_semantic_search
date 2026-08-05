# RAM++ vs TinyCLIP — qualitative tag comparison

Upload an image, get the tags each model predicts, side by side, over the
**same 4585-tag vocabulary**. Because both models score an identical tag set,
disagreements are attributable to the models rather than to vocabulary coverage.

## Why this is not a trivially fair fight

The two models are calibrated on different scales and the UI never pretends
otherwise:

| | RAM++ | TinyCLIP |
|---|---|---|
| score | sigmoid probability | cosine similarity |
| typical cutoffs | 0.45 – 1.00 (its own per-tag file) | 0.20 – 0.40 (LVIS-calibrated) |
| tags with a real per-tag cutoff | 4585 / 4585 | **1037 / 4585** |
| everything else | — | one global knob in the UI |

Raw scores are **not comparable across models**. The margin over each tag's own
threshold is, so every panel prints it in bold.

**Ranking is by raw score, not by margin** — and that distinction was load-bearing.
Ranking by margin produced garbage: LVIS weak-bucket tags calibrate down to 0.20
(the grid floor), so a mediocre 0.35 cosine scored margin +0.15 and outranked
`train` at cosine 0.43 over a 0.36 threshold. On a photo of a train, TinyCLIP's
margin-ranked top hits were `file`, `masher`, `oar`, `pegleg`, `sawbuck`, and the
`Both` panel was empty. Switching to score ranking put `bullet train`, `railcar`,
`train car` on top and `Both` went 0 → 5. Margin is for reading, score is for ordering.

## Setup

```bash
./setup_venv.ps1
```

A dedicated `venv_ramclip` is required, not a convenience. RAM's vendored
BLIP-era `ram/models/bert.py` imports `transformers.modeling_utils.apply_chunking_to_forward`,
which was removed in transformers 5.x — the sibling `venv_clip` runs 5.12.1 and
cannot be reused without breaking the existing `tinyClip_vs_ClipVit32` app.

Two further pins are load-bearing and were found the hard way:
- **gradio 5.x, not 6.x** — gradio 6 requires `huggingface-hub>=1.2`, transformers 4.44 requires `<1.0`. Unresolvable together.
- **scipy 1.13.1** — RAM's Swin position-bias interpolation calls `scipy.interpolate.interp2d`, removed in SciPy 1.14.

## Build order

```bash
python 00_download_ram.py                         # 2.8 GB checkpoint + tag lists
python 01_build_text_classifiers.py --source templates   # ~29 min, 403k strings
python 02_build_tag_mapping.py                    # LVIS -> RAM thresholds
./venv_ramclip/Scripts/python.exe verify_env.py --full   # preflight
./venv_ramclip/Scripts/python.exe app.py          # http://127.0.0.1:7863
```

The `recognize-anything` source is vendored under `vendor/` (git clone, gitignored).

### When the ChatGPT descriptions arrive

The `llm` classifier set is the second half of the A/B: identical pipeline, but
the per-tag text is 10 generated visual descriptions instead of 88 templates.

```bash
# drop the returned .json files into data/llm_desc/ then:
python ingest_descriptions.py                     # validate, report gaps
python ingest_descriptions.py --missing-out redo.json   # re-request just the gaps
python 01_build_text_classifiers.py --source llm  # builds classifiers_llm.npy
```

Both sets are then selectable in the UI, so you can compare template-driven vs
description-driven TinyCLIP on the same image against the same RAM++ output.

## Benchmark: templates vs LLM descriptions

`03_benchmark_classifiers.py` answers "which text source makes a better classifier"
quantitatively, by reusing the existing calibration infrastructure. 1037 of the
4585 RAM tags map onto an LVIS category, so we have real ground truth: the 100,169
precomputed LVIS-train image embeddings and their per-category positives. Both
classifier sets are scored over the same images, the same tags, and the same
seeded negative samples — the only variable is the text that produced the embedding.

Balanced regime, macro Fβ (β=0.5), best-achievable threshold per tag:

| set | strings/tag | macro Fβ | precision | recall | high | medium | weak |
|---|---|---|---|---|---|---|---|
| templates | 88 | 0.7955 | 0.8193 | 0.7683 | 0.7580 | 0.7850 | 0.8560 |
| **llm** | **20** | **0.8098** | **0.8360** | 0.7718 | 0.7712 | 0.8024 | 0.8673 |
| combined | 108 | 0.8111 | 0.8385 | 0.7693 | 0.7734 | 0.8033 | 0.8683 |

Head-to-head on the 1037 tags: **llm wins 482, templates wins 387, 168 ties**,
mean delta **+0.0143 (+1.8%)**.

Read it honestly — this is a modest win, not a transformation:

- The gain is **+1.8% macro Fβ, almost entirely from precision** (+0.017); recall
  barely moves. So descriptions mostly buy fewer false positives.
- It comes from **4.4x fewer strings** (20 vs 88), which is the more interesting
  result: 10 descriptions × 2 templates beats 88 generic prompts. The earlier LVIS
  ablation hinted at this (2 gloss descriptions beat 85 prompts per-string) and it
  reproduces at scale.
- **`combined` is within noise of `llm` alone** (+0.0013). Adding the 88 templates
  back on top buys essentially nothing, so there is no reason to ship the union.
- In the **naive** full-split regime templates are marginally ahead
  (0.0884 vs 0.0848), but both numbers sit on the base-rate floor and should only
  be read set-vs-set, never in isolation.

The practical difference shows up in the app: on the same image at the same knob,
`templates` fires **266** tags above threshold while `llm` fires **87** — the
description embeddings are meaningfully more discriminative.

Sanity check: the templates balanced number (0.7955) reproduces the original LVIS
balanced calibration (0.800 in `tinyClipCaliberation/README.md`), which is
independent evidence the benchmark harness is wired correctly.

```bash
./venv_ramclip/Scripts/python.exe 03_benchmark_classifiers.py
./venv_ramclip/Scripts/python.exe 03_benchmark_classifiers.py --combined --regime balanced
```

## Scripts

| file | role |
|---|---|
| `config.py` | all paths/constants; wires `tinyclip_encoder` and `templates` onto `sys.path` |
| `00_download_ram.py` | resumable 2.8 GB checkpoint fetch with SSL bypass |
| `01_build_text_classifiers.py` | `--source templates\|llm` → `(4585, 512)` super-embeddings |
| `02_build_tag_mapping.py` | LVIS 1203 → RAM 4585 matching, emits audit CSV + threshold array |
| `03_benchmark_classifiers.py` | quantitative templates-vs-llm A/B on LVIS ground truth |
| `ingest_descriptions.py` | validates ChatGPT JSON drops, reports exactly what's missing |
| `ssl_bypass.py` | corporate-TLS workaround; must import before transformers |
| `ram_tagger.py` / `tinyclip_tagger.py` | the two engines, same result shape |
| `verify_env.py` | preflight; `--full` loads RAM++ and runs one inference |
| `smoke_compare.py` | both models in one process, no Gradio |
| `app.py` | the Gradio UI |

## Reuse

Nothing about the encoder or the prompt templates is reimplemented here:

- `tinyClip_vs_ClipVit32/tinyclip_encoder.py` — Android-exact TinyCLIP. Its
  deliberate double BOS/EOT tokenizer quirk must be preserved, otherwise
  embeddings stop matching the LVIS calibration that produced the thresholds.
- `tinyClipCaliberation/templates.py` — the 85 prompt + 3 question templates.
  RAM tags are bare strings with no synonyms and no gloss, so each yields
  exactly 88 strings (verified across all 4585).
- `tinyClipCaliberation/data/calibration/balanced_thresholds.npy` — the
  authoritative per-tag cutoffs, chosen over the naive full-split thresholds
  because the latter are crushed by the 100k-image negative base rate.

## Known limitations

1. **RAM++ runs at 224 by default, but the checkpoint is a 384 model.** On load
   you will see `Position interpolate ... from 23x23 to 13x13` — that is the
   position bias being resampled. `ram_tag_list_threshold.txt` was tuned at 384,
   so at 224 those cutoffs are approximate and RAM's scores shift. The status
   line warns whenever size != 384; switch to 384 in the UI to remove the confound.
2. **Only 22.6% of RAM tags have a real TinyCLIP threshold.** The vocabularies
   genuinely diverge — LVIS has `airplane`/`trash_can`, RAM has `plane` and no
   trash can at all. The rest ride the global knob, so knob choice dominates
   TinyCLIP's apparent behavior.
3. **7 RAM tags were dropped as ambiguous** (`bow`, `fish`, `glasses`, `mailbox`,
   `octopus`, `pan`, `salmon`) because two LVIS senses claimed them. They use the
   knob rather than an arbitrarily-picked sense threshold. See the `note` column
   in `lvis_to_ram_mapping.csv`.
4. **No GPU.** RAM++ is ~1.5 s/image at 224 on CPU, several seconds at 384, plus
   a ~6 s first load. Single-image interactive use only.
