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
```

`03_` requires the LVIS image embeddings from the sibling calibration project.
