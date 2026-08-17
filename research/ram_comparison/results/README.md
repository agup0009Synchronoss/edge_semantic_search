# RAM++ vs TinyCLIP — committed results

Small outputs of the mapping and benchmark steps, committed so the app can
start and the README's numbers can be checked without rebuilding the classifier
sets (~29 min for templates). The classifier `.npy` files themselves and the
`strings_*.jsonl` are regenerable and stay out of git.

## Files

| File | Produced by | What it is |
|---|---|---|
| `tag_order_4585.json` | `01_build_text_classifiers.py` | Row order for the 4585 RAM tags. **Required to interpret `thresholds_4585.npy`** and both classifier matrices. |
| `thresholds_4585.npy` | `02_build_tag_mapping.py` | `(4585,)` float32 per-tag TinyCLIP cosine cutoffs, `NaN` where uncalibrated. **1037 of 4585 are real** (22.6%); range 0.20–0.40. The rest ride the UI's global knob. |
| `lvis_to_ram_mapping.csv` | `02_build_tag_mapping.py` | 1051 rows — the LVIS-1203 → RAM-4585 audit trail: which surface form matched, by which rule, via synonym or not. |
| `benchmark_templates_vs_llm.csv` | `03_benchmark_classifiers.py` | 1037 rows — per-tag templates vs LLM vs combined, in the balanced regime, with a `winner` column. |
| `precision_thresholds_p80.npy`<br>`_p85.npy` / `_p90.npy` | `04_precision_calibration.py` | `(1203,)` float32 **indexed by LVIS row**, `NaN` = not calibrated. Thresholds that *guarantee* ≥80/85/90% precision on a balanced subset. 539 / 497 / 452 real values. |
| `precision_calibration.csv` | `04_precision_calibration.py` | 3609 rows (1203 tags × 3 targets) — threshold, confusion matrix, achieved precision/recall, and a `reason` column explaining every NA. |

## The precision sets are not interchangeable with the Fβ ones

Two traps worth naming explicitly.

**They are LVIS-indexed but built from RAM embeddings.** `precision_thresholds_*.npy`
is `(1203,)`, the same shape as
`../../lvis_calibration/results/balanced_thresholds.npy`, and the two are *not*
comparable. The Fβ set was calibrated on that project's own
`classifiers_combined.npy`; these were calibrated on **this** project's
`classifiers_templates.npy`. Same shape, different text embeddings, different
meaning. `02_build_tag_mapping.py` is what projects them onto the 4585.

**`p90` does not mean 90% precise.** It means the tag cleared 90% precision on a
forced 50/50 subset. With `f/r` pinned by that operating point, precision at real
prevalence π is `π / (π + (1−π)·f/r)` — so at a realistic ~1% prevalence, p80
lands near **4%** and p90 near **8%**. These are intrinsic-separability operating
points, not deployment estimates.

## Precision-target calibration

Every other threshold here comes from `argmax(Fβ)`, which *optimizes* for
precision without promising any. These *guarantee* a floor or report NA.

Eligibility is strict: the tag must map to a RAM tag **and** have >20 ground-truth
images (1037 mapped − 401 too rare = **636 eligible**; the other 567 are NA before
any target applies). A grid point also needs ≥5 predicted positives to be trusted,
so a spurious 2/2 = 1.0 cannot win.

| target | calibrated / 636 eligible | threshold range (median) | median recall | mean achieved precision |
|---|---|---|---|---|
| ≥80% | 539 (85%) | 0.280–0.415 (0.340) | 0.717 | 0.823 |
| ≥85% | 497 (78%) | 0.295–0.400 (0.350) | 0.629 | 0.874 |
| ≥90% | 452 (71%) | 0.300–0.415 (0.360) | 0.531 | 0.923 |

The sets are properly nested — p90 ⊆ p85 ⊆ p80 — and no `weak`-bucket tag appears
in any of them, which is the expected consequence of the >20 GT filter.

Note the threshold range: **nothing exceeds 0.415**, even though the search grid
runs to 0.95. Raising the old 0.65 cap rescued zero tags. What limits the higher
targets is score separability, not the grid — 97 eligible tags cannot reach even
80% anywhere on `[0.20, 0.95]`.

## Read the mapping CSV before trusting a threshold

The `note` column records where the mapping was not clean:

- **1036 rows** map cleanly.
- **14 rows** are `ambiguous_sense_dropped` — that is **7 RAM tags × 2 competing
  LVIS senses each** (`bow`, `fish`, `glasses`, `mailbox`, `octopus`, `pan`,
  `salmon`). Rather than pick a sense arbitrarily, these get no threshold and
  use the knob.
- **1 row** is `review_ram_variant`.

(1051 rows for 1037 calibrated tags: the 14 dropped rows are the difference.)

## Benchmark result

Balanced regime, macro Fβ (β = 0.5), best-achievable threshold per tag:

| set | strings/tag | macro Fβ | precision | recall |
|---|---|---|---|---|
| templates | 88 | 0.7955 | 0.8193 | 0.7683 |
| **llm** | **20** | **0.8098** | **0.8360** | 0.7718 |
| combined | 108 | 0.8111 | 0.8385 | 0.7693 |

Head-to-head on the 1037 mapped tags: llm wins 482, templates wins 387,
168 ties; mean delta +0.0143 (+1.8%).

Read honestly — this is a modest win, not a transformation. The gain is almost
entirely precision, and it comes from **4.4x fewer strings**, which is the more
interesting result. `combined` is within noise of `llm` alone, so there is no
reason to ship the union.

The templates figure (0.7955) independently reproduces the LVIS balanced
calibration (0.800 in `../../results/`), which is evidence the benchmark
harness is wired correctly.

## Reproducing

```bash
python 02_build_tag_mapping.py
./venv_ramclip/Scripts/python.exe 03_benchmark_classifiers.py --combined --regime balanced
./venv_ramclip/Scripts/python.exe 04_precision_calibration.py
```

`03_` and `04_` require the LVIS image embeddings from the sibling calibration
project.

To run the app on a precision set instead of the Fβ default, re-project the
thresholds onto the 4585 and restart:

```bash
./venv_ramclip/Scripts/python.exe 02_build_tag_mapping.py --thresholds p85
```

That overwrites `thresholds_4585.npy`, so `--thresholds balanced` puts it back.
Expect far fewer `cal` tags — 497 rather than 1037 — because tags that cannot
clear the floor correctly fall through to the UI knob.
