# Edge Semantic Search

On-device semantic photo search for Android, plus the model-evaluation work that
justified the model choice.

Type **"dinner with friends"** or **"sunset at the beach"** and the app ranks
your photos by semantic similarity in milliseconds. Everything runs on-device:
the ML models, the tokenizer, the index, and the search. No network calls are
made at any point.

---

## What is in this repo

Four workstreams. The Android app is the deliverable; the three under
`research/` are the measurement work behind it.

| # | Where | What it answers | Status |
|---|---|---|---|
| 1 | [`app/`](app/) | Can CLIP-class semantic search run entirely on an Android device? | **Working POC**, tested on a Galaxy Z Flip5 |
| 2 | [`research/vitb32_benchmark/`](research/vitb32_benchmark/) | How much retrieval quality does TinyCLIP-8M lose against full CLIP ViT-B/32? | Complete — side-by-side Gradio search over 108K images |
| 3 | [`research/lvis_calibration/`](research/lvis_calibration/) | What cosine threshold means "this tag is present" for each of 1,203 LVIS tags? | Complete — per-tag thresholds committed |
| 4 | [`research/ram_comparison/`](research/ram_comparison/) | How does TinyCLIP tagging compare to production RAM++ over the same 4,585 tags? | Complete — Gradio A/B app + quantitative benchmark |

They chain in that order:

```
vitb32_benchmark ──┐
                   │ establishes the Android-exact encoder
                   ▼
         research/common/  (tinyclip_encoder.py + the ONNX assets)
                   │
                   ├──► lvis_calibration ──► per-tag thresholds
                   │                              │
                   └──────────────────────────────┴──► ram_comparison
```

`research/common/` is the shared floor: one encoder, one copy of the ONNX
models, one prompt-template set. Nothing is reimplemented downstream.

---

## Start here

- **Just want the app?** → [Build & install](#build--install) below.
- **Want to reproduce the research?** → [`docs/reproducibility.md`](docs/reproducibility.md)
- **Want the full POC write-up?** → [`docs/android_poc.md`](docs/android_poc.md)
- **Want the numbers?** → [`research/README.md`](research/README.md)

---

## How the Android app works

| Stage | What happens |
|---|---|
| **Background indexing** | WorkManager job (charging + idle) scans new photos via MediaStore, runs each through the vision encoder, stores the 512-d embedding in a Room database |
| **Live query** | You type a query; the text encoder embeds it in real time |
| **Ranking** | Brute-force cosine similarity across all stored embeddings; top results shown in a ranked grid |

```
MediaStore
    └─► IndexWorker (WorkManager, charging+idle)
            └─► ClipEngine.encodeImage()   vision_model_fp32.onnx
                    └─► FloatArray[512] L2-normalized
                            └─► Room DB (mediaId, embedding blob)

Query text
    └─► ClipEngine.encodeText()   text_model_int8.onnx + custom_op_cliptok.onnx
            └─► FloatArray[512] L2-normalized
                    └─► dot product with all stored embeddings
                            └─► ranked URIs → Coil image grid
```

Key files:

| File | Role |
|---|---|
| `app/.../EdgeSearchApp.kt` | Application class; copies assets, builds ORT sessions, exposes `engineReady` |
| `app/.../ml/ClipEngine.kt` | Two ORT sessions (vision fp32, text int8) + tokenizer; `encodeImage()` / `encodeText()` |
| `app/.../ml/ImagePreprocess.kt` | Bitmap → resize/crop 224×224 → normalize → NCHW float32 |
| `app/.../ml/Tokenizer.kt` | Wraps `custom_op_cliptok.onnx`; pads/truncates to 77; BOS=49406, EOT=49407 |
| `app/.../index/IndexWorker.kt` | CoroutineWorker; incremental (skips already-indexed media IDs) |
| `app/.../search/SearchScreen.kt` | Compose UI — picker, embed log, search bar, result grid |
| `tools/export_split.py` | Dev-machine script that produced the ONNX assets |

The app also ships a **benchmark screen**: pick N images, watch per-image decode
and inference timings, then search and see text-embed latency plus cosine scores
to 3 decimals.

---

## The model

**TinyCLIP-ViT-8M-16-Text-3M-YFCC15M** ([paper](https://arxiv.org/abs/2309.04504)),
projection dimension 512.

| Asset | Size | Notes |
|---|---|---|
| `vision_model_fp32.onnx` | ~33 MB | fp32. The int8 variant is unusable on ORT Android — no `ConvInteger` CPU kernel |
| `text_model_int8.onnx` | ~15 MB | int8, extracted from the HF pre-quantized combined model |
| `custom_op_cliptok.onnx` | ~1.4 MB | CLIP BPE vocab + merges, via `onnxruntime-extensions` |

**These are committed**, in two places: `app/src/main/assets/` (Gradle packages
the APK from there) and `research/common/assets/` (the Python side imports from
there). Both copies are byte-identical and held that way by
[`ASSETS.sha256`](ASSETS.sha256):

```bash
python research/common/verify_assets.py
```

Sources: [`onnx-community/TinyCLIP-...-ONNX`](https://huggingface.co/onnx-community/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX),
original [`wkcn/TinyCLIP-...`](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M).
See [`THIRD_PARTY.md`](THIRD_PARTY.md) for licensing of every upstream model and dataset.

### Runtime

- ONNX Runtime Android `1.20.0`, ONNX Runtime Extensions Android `0.13.0`
- NNAPI EP enabled with CPU fallback

---

## Build & install

Requirements: Android Studio Hedgehog+, JDK 17+, Android SDK 35, device/emulator API 26+.

A fresh clone needs `local.properties` pointing at your SDK — it is gitignored
because the path is machine-specific. Opening the project in Android Studio
creates it for you; otherwise write it yourself, or export `ANDROID_HOME`:

```bash
echo "sdk.dir=/path/to/Android/Sdk" > local.properties
```

Then:

```bash
./gradlew installDebug
```

```bash
adb shell am start -n com.edgesearch.app/.MainActivity
```

```bash
adb logcat -s EdgeSearchApp ClipEngine Tokenizer
```

On Windows, use `gradlew.bat` and point `JAVA_HOME` at the Android Studio JBR:

```bat
set JAVA_HOME=C:\Program Files\Android\Android Studio\jbr
```

Because the ONNX assets are committed, there is no model-generation step — once
the SDK path is set, a fresh clone builds and runs as-is.

---

## Permissions

| Permission | Why |
|---|---|
| `READ_MEDIA_IMAGES` | Photo library access (API 33+) |
| `READ_MEDIA_VISUAL_USER_SELECTED` | Selective photo access (API 34+) |
| `READ_EXTERNAL_STORAGE` | Photo access on API ≤ 32 |
| `FOREGROUND_SERVICE` + `..._DATA_SYNC` | Long-running background indexer |
| `POST_NOTIFICATIONS` | Indexer progress notification (API 33+) |
| `RECEIVE_BOOT_COMPLETED` | WorkManager reschedule after reboot |

## Key dependencies

| Library | Version | Purpose |
|---|---|---|
| ONNX Runtime Android | 1.20.0 | On-device inference |
| ONNX Runtime Extensions | 0.13.0 | CLIPTokenizer custom op |
| Jetpack Compose BOM | 2024.10.01 | UI |
| Coil | 2.7.0 | Async image loading |
| WorkManager | 2.10.0 | Background indexing |
| Room | 2.6.1 | Embedding persistence |
| Kotlin Coroutines | 1.9.0 | Async inference + IO |

**Device tested:** Samsung Galaxy Z Flip5 (SM-F731U1), Android 16 (API 36).

---

## Out of scope

No server, no cloud sync, no account, no ANN index, no hand-written quantizer.
Everything wires together published components.
