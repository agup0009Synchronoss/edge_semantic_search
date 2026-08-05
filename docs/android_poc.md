# Semantic Search on Edge Device — Android POC

**Status:** POC Complete  
**Platform:** Samsung Galaxy Z Flip5 (SM-F731U1), Android 16 (API 36)  
**GitHub:** https://github.com/agup0009Synchronoss/edge_semantic_search  
**Author:** agup0009  
**Date:** June 2026  

---

## Table of Contents

1. [Objective](#objective)
2. [What We Proved](#what-we-proved)
3. [System Architecture](#system-architecture)
4. [Model Selection & Rationale](#model-selection--rationale)
5. [The ONNX Asset Pipeline](#the-onnx-asset-pipeline)
6. [On-Device Inference Stack](#on-device-inference-stack)
7. [Image Preprocessing](#image-preprocessing)
8. [Tokenization](#tokenization)
9. [Embedding Storage & Retrieval](#embedding-storage--retrieval)
10. [Benchmark UI — What It Measures](#benchmark-ui--what-it-measures)
11. [Technical Blockers Encountered & Resolutions](#technical-blockers-encountered--resolutions)
12. [Future Improvements — Prioritised](#future-improvements--prioritised)
13. [References & External Links](#references--external-links)

---

## Objective

Prove that **semantic (embedding-based) photo search can run entirely on an Android device** — no server, no cloud API, no network call — using a compressed CLIP model served through ONNX Runtime.

The hypothesis being tested: *can a sub-10M parameter vision-language model, running on a mobile CPU/NNAPI, produce embeddings that are semantically meaningful enough to return relevant photos from a natural-language query?*

This is not a product. It is a measurement exercise: build the minimal viable end-to-end system, instrument it fully, and surface the real costs (model size, inference latency, indexing time) so that production viability can be assessed.

---

## What We Proved

| Claim | Result |
|---|---|
| CLIP-class model runs on Android CPU without server | ✅ Confirmed — ORT 1.20.0 on device |
| Text and vision towers can be separated from a combined ONNX | ✅ `onnx.utils.extract_model` |
| CLIP BPE tokenizer can run on-device without writing one by hand | ✅ `onnxruntime-extensions` CLIPTokenizer custom op |
| Embeddings are stable across sessions (persisted in Room) | ✅ Background WorkManager indexer |
| NNAPI EP loads and partially accelerates inference | ✅ Enabled, partial coverage (see blockers) |
| Full benchmark — per-image decode time, inference time, text embed time, cosine scores | ✅ Benchmark UI delivers all of this live |

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  INDEXING PATH (background, charging + idle, WorkManager)      │
│                                                                │
│  MediaStore ──► IndexRepository ──► ClipEngine.encodeImage()  │
│                      │                      │                  │
│            diff vs Room DB          vision_model_fp32.onnx     │
│            (skip known IDs)         (ONNX Runtime, NNAPI EP)   │
│                      │                      │                  │
│                      └──────── Room DB ◄────┘                  │
│                         (mediaId, dateAdded,                   │
│                          embedding BLOB, 512 × f32)            │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│  QUERY PATH (live, on keystroke submission)                     │
│                                                                │
│  query string                                                  │
│      └──► Tokenizer (custom_op_cliptok.onnx)                  │
│                └──► input_ids [1,77] + attention_mask [1,77]   │
│                          └──► ClipEngine.encodeText()          │
│                                    │                           │
│                           text_model_int8.onnx                 │
│                           (ONNX Runtime, CPU EP)               │
│                                    │                           │
│                          text_embed [512] L2-normalized        │
│                                    │                           │
│                     dot product vs all stored image_embeds     │
│                     (brute force, in-memory, O(N))             │
│                                    │                           │
│                        ranked URIs → Coil image grid           │
└────────────────────────────────────────────────────────────────┘
```

Both embedding vectors are **L2-normalized before storage and comparison**. This collapses cosine similarity to a plain dot product — no division at query time, and the stored BLOBs are already search-ready.

---

## Model Selection & Rationale

### TinyCLIP-ViT-8M-16-Text-3M-YFCC15M

**Paper:** [TinyCLIP: CLIP Distillation via Weight Inheritance for Large-Scale Language-Image Pre-training](https://arxiv.org/abs/2309.04504) (ICCV 2023)  
**HuggingFace (ONNX):** https://huggingface.co/onnx-community/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX  
**HuggingFace (original):** https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M  
**GitHub (TinyCLIP):** https://github.com/wkcn/TinyCLIP  

**Why this model:**

| Criterion | Decision |
|---|---|
| Parameter count | ~11M total (8M vision + 3M text) — fits in device RAM without swapping |
| Projection dimension | 512-d — same as OpenAI ViT-B/32; tokenizer vocab is directly compatible |
| Architecture | ViT-8M (vision), transformer (text) — standard ONNX export with clean input/output names |
| Quantized variant available | HF provides pre-tested `model_int8.onnx` — no quantization work needed for text tower |
| Licensing | MIT |

**What the model number means:** `ViT-8M-16` = Vision Transformer with 8M params, patch size 16. `Text-3M` = 3M param text encoder. `YFCC15M` = trained on Yahoo Flickr Creative Commons 15M image-text pairs (a subset of the full CLIP training set).

### Why not a larger CLIP variant?

| Model | Vision params | Projection dim | Practical issue for edge |
|---|---|---|---|
| OpenAI ViT-B/32 | 87M | 512 | ~350MB on device — too large for POC |
| OpenAI ViT-L/14 | 307M | 768 | Impractical |
| TinyCLIP 39M | 39M | 512 | Possible, but 5× larger than 8M for marginal quality gain at POC stage |
| **TinyCLIP 8M (this POC)** | **8M** | **512** | **Chosen: smallest valid CLIP model** |
| MobileCLIP-S0 (Apple) | ~6M | 512 | Good candidate; not explored in this POC |
| SigLIP (Google) | varies | 768/1152 | Different loss function; better zero-shot but heavier |

---

## The ONNX Asset Pipeline

### The challenge

The HuggingFace ONNX community repo for TinyCLIP only ships a **combined** model (`model_int8.onnx`, 23MB) that runs both towers in a single ONNX graph. Running it as-is means executing both the vision encoder AND the text encoder on every call — wasted compute. We need two independent sessions: one for bulk image indexing, one for live text queries.

### Solution: `onnx.utils.extract_model`

`onnx.utils.extract_model` takes a combined ONNX and a set of input/output boundary names and produces a valid subgraph ONNX. No PyTorch required, no custom quantization — just graph surgery on the pre-tested HF artifact.

```
model_int8.onnx (23MB, HF pre-quantized)
    │
    ├─► extract [pixel_values] → [image_embeds]  ──► vision_model_int8.onnx (~8.7MB)
    └─► extract [input_ids, attention_mask] → [text_embeds]  ──► text_model_int8.onnx (~14MB)
```

**Script:** `tools/export_split.py`  
**Dependencies:** `pip install onnx onnxruntime onnxruntime-extensions==0.13.0 huggingface_hub`

### The vision model int8 problem (key blocker — see §Blockers)

The extracted `vision_model_int8.onnx` **cannot run on ORT Android CPU EP** because dynamically-quantized Conv layers emit `ConvInteger` ops, and this kernel is absent from the Android CPU execution provider. The workaround is to export the vision tower as **fp32** directly from PyTorch:

```python
from transformers import CLIPVisionModelWithProjection
import torch

model = CLIPVisionModelWithProjection.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(model, dummy, "vision_model_fp32.onnx",
                  input_names=["pixel_values"], output_names=["image_embeds"],
                  opset_version=14)
```

The text model int8 extracted subgraph works fine because text transformer layers use `MatMulInteger` / `QLinearMatMul`, which *are* in the ORT Android kernel set.

### Final assets committed to the repo

| File | Precision | Size | How produced |
|---|---|---|---|
| `vision_model_fp32.onnx` | fp32 | ~33MB | PyTorch `torch.onnx.export` of `CLIPVisionModelWithProjection` |
| `text_model_int8.onnx` | int8 | ~15MB | `onnx.utils.extract_model` from HF `model_int8.onnx` |
| `custom_op_cliptok.onnx` | n/a | ~1.4MB | `onnxruntime_extensions.SingleOpGraph.build_graph(CLIPTokenizer)` with vocab+merges embedded |

---

## On-Device Inference Stack

### ONNX Runtime Android 1.20.0

**GitHub:** https://github.com/microsoft/onnxruntime  
**Maven:** `com.microsoft.onnxruntime:onnxruntime-android:1.20.0`  

ORT is the inference engine. It supports multiple **Execution Providers (EPs)** that delegate computation to hardware accelerators.

### Execution Provider strategy

```kotlin
// Vision session — try NNAPI first, fall back to CPU
val vOpts = OrtSession.SessionOptions().apply {
    try {
        addNnapi()   // delegates to Android Neural Networks API
    } catch (e: Exception) {
        // CPU EP handles everything
    }
}
```

**NNAPI EP status in this POC:**  
NNAPI loaded successfully, but only **63 out of 855 nodes** in the fp32 vision model are NNAPI-delegatable. The remaining 792 nodes fall back to the CPU EP. This is because the fp32 ViT model contains op types (e.g. `Gather`, `Reshape`, `Transpose` with dynamic shapes) that the NNAPI EP partition algorithm conservatively leaves on CPU to avoid shape inference issues.

**Text session:** runs on CPU EP only (no NNAPI). The text model is small (~3M params) and query-time, so inference is fast enough on CPU.

### ORT Extensions — CLIPTokenizer custom op

**GitHub:** https://github.com/microsoft/onnxruntime-extensions  
**Maven:** `com.microsoft.onnxruntime:onnxruntime-extensions-android:0.13.0`  

The extensions library ships a `CLIPTokenizer` as a registered custom op. The tokenizer vocab and BPE merge rules are embedded directly into the ONNX model as string attributes — the device needs no external files. The custom op library is loaded at session creation time via `OrtxPackage.getLibraryPath()`.

**Token validation:** `"a photo of a cat"` → `[49406, 320, 1125, 539, 320, 2368, 49407, 0, 0, ...]`  
(49406 = BOS `<|startoftext|>`, 49407 = EOT `<|endoftext|>`)

---

## Image Preprocessing

CLIP has a fixed preprocessing contract that must be reproduced exactly on-device, or embeddings will be in the wrong space and similarity scores will be garbage.

**Pipeline: Bitmap → NCHW float32 tensor [1, 3, 224, 224]**

1. **Resize** — scale so the shorter side equals 224 (maintains aspect ratio)
2. **Center crop** — take the central 224×224 patch
3. **Normalize** — divide by 255, subtract per-channel mean, divide by per-channel std

```
Mean: [0.48145466, 0.4578275,  0.40821073]   (CLIP training distribution)
Std:  [0.26862954, 0.26130258, 0.27577711]
```

4. **Channel order** — R, G, B packed in NCHW order (all R pixels, then all G, then all B)

The implementation uses a **reusable `FloatBuffer`** (pre-allocated, 1×3×224×224 = 150,528 floats) to avoid GC pressure during batch indexing.

---

## Tokenization

- Vocabulary: 49,408 tokens (standard CLIP BPE)
- Sequence length: 77 tokens (fixed, same as OpenAI CLIP)
- BOS token: `49406` (`<|startoftext|>`)
- EOT token: `49407` (`<|endoftext|>`)
- Padding: `0` with corresponding `attention_mask = 0`
- Truncation: if text encodes to > 75 tokens, the suffix is dropped and EOT is placed at position 76

The `Tokenizer.kt` wrapper normalises the custom op output (which may be shape `[1,N]` or `[N]`) to a fixed `[1,77]` int64 tensor, inserting BOS/EOT and building the attention mask.

---

## Embedding Storage & Retrieval

**Room database** stores one row per photo:

| Column | Type | Notes |
|---|---|---|
| `mediaId` | `Long` (PK) | MediaStore content URI ID |
| `dateAdded` | `Long` | Unix timestamp from MediaStore |
| `embedding` | `ByteArray` | 512 × 4 bytes = 2048 bytes per image (raw float32 little-endian) |

**Indexing is incremental:** `IndexRepository` queries `existingIds()` from Room, fetches new IDs from MediaStore, and only encodes photos not already in the DB. Re-running indexing on an unmodified library is a no-op.

**Retrieval is brute-force O(N):** at query time, all 512-d embeddings are loaded into a flat `FloatArray`, the text embedding is computed, and cosine similarity (= dot product, since both sides are L2-normalized) is computed against every stored image. For a 1,000-photo library this is ~512,000 multiply-adds — negligible on a modern CPU.

---

## Benchmark UI — What It Measures

The app ships a benchmark screen designed to surface the real cost of each operation:

### Image embedding benchmark

Select N images → tap **Embed** → live log appears per image:

| Column | What it measures |
|---|---|
| `Decode ms` | `BitmapFactory.decodeStream()` + `ImagePreprocess.process()` |
| `Infer ms` | `OrtSession.run()` for vision model only |
| `Total ms` | Decode + Infer (wall time) |

A wall-clock total and per-image average is shown when all images finish.

### Text query benchmark

Type a query → tap **Search**:

- **Text embed ms** — time for tokenizer session + text ORT session + L2 normalize
- **Ranked results** — all N images sorted by cosine similarity, scores shown to 3 decimal places (e.g. `cos: 0.284`)

Score interpretation for TinyCLIP 8M (approximate, will vary by content):

| Score range | Interpretation |
|---|---|
| `≥ 0.28` | Strong semantic match |
| `0.20 – 0.28` | Moderate / partial match |
| `< 0.20` | Weak / coincidental similarity |

---

## Technical Blockers Encountered & Resolutions

These are documented because each one represents a real edge-ML deployment risk that will recur in production.

---

### B1 — `ConvInteger` op missing from ORT Android CPU EP

**Symptom:** App crashed at session load with `ORT_NOT_IMPLEMENTED: Could not find an implementation for ConvInteger(10)`.  
**Root cause:** ORT's post-training dynamic quantization converts `Conv` to `ConvInteger`. The Android CPU EP does not ship a `ConvInteger` kernel (as of ORT 1.20.0). The NNAPI EP covers only a fraction of nodes, so it cannot route around it.  
**Resolution:** Export the vision tower from PyTorch as fp32 (`torch.onnx.export`). fp32 `Conv` is fully supported. Model size increases from ~8MB (int8) to ~33MB (fp32).  
**Future fix:** See §Future Improvements — QDQ quantization or NNAPI full-model delegation.

---

### B2 — NNAPI EP covers only 63/855 nodes

**Symptom:** NNAPI loaded but inference speed was not significantly faster than CPU-only baseline.  
**Root cause:** NNAPI partitions the graph conservatively. Dynamic-shape ops (`Gather`, `Reshape` with computed axes, etc.) cannot be delegated. The fp32 ViT model has many such ops.  
**Resolution:** Accepted for POC. Both paths (NNAPI partial + CPU remainder) produce correct output.  
**Future fix:** See §Future Improvements — QNN EP, or model restructuring to maximise NNAPI coverage.

---

### B3 — Tokenizer ONNX was corrupted (Git LFS pointer served as HTML)

**Symptom:** `ORT_INVALID_PROTOBUF` when loading `custom_op_cliptok.onnx`. First 16 bytes of file were `<!DOCTYPE`.  
**Root cause:** Downloaded from `oracle/sd4j` GitHub repository; the file was stored in Git LFS and the LFS pointer was served as raw HTML rather than the binary.  
**Resolution:** Generated the tokenizer ONNX from scratch using `onnxruntime_extensions.SingleOpGraph.build_graph(CLIPTokenizer, vocab=..., merges=...)` with the vocab and merges files from HuggingFace. This is actually a better approach — the tokenizer is purpose-built, validated, and reproducible.

---

### B4 — Stale asset persisted across installs

**Symptom:** After the tokenizer was regenerated and the SharedPreferences key was bumped, the old corrupt tokenizer was still on-device.  
**Root cause:** Asset copy code had `if (!dest.exists()) { copy() }` — the file existed (even though corrupt), so the new version was never written.  
**Resolution:** Removed the existence check. The SharedPreferences version key now controls copying unconditionally: if the key is absent (first install or version bump), all assets are always overwritten.

---

### B5 — fp16 text model conversion hung for 3+ hours

**Symptom:** `convert_float_to_float16` with `op_block_list=["Cast", "Gather"]` on the 59MB combined model ran indefinitely.  
**Root cause:** Blocking `Gather` (the vocabulary embedding lookup op) is catastrophic for the converter — it forces the converter into a degenerate state where it cannot propagate types through the token embedding layer, causing a pathological loop.  
**Resolution:** Abandoned fp16 conversion entirely. Extracted text model directly from HF's pre-quantized `model_int8.onnx`. The HF int8 text model is well-tested and loads cleanly on ORT Android.

---

## Future Improvements — Prioritised

These are ordered by expected effort vs. impact from an edge-ML standpoint.

---

### F1 — Vision model: fp32 → int8 via QDQ quantization *(High impact, medium effort)*

**Current state:** fp32 vision model, 33MB, standard `Conv` ops.  
**Problem:** Dynamic quantization (ORT's default `quantize_dynamic`) produces `ConvInteger` ops that ORT Android CPU EP cannot run.  
**Fix:** Use **QDQ (Quantize-Dequantize) format** quantization instead. QDQ inserts `QuantizeLinear` / `DequantizeLinear` nodes around ops rather than replacing `Conv` with `ConvInteger`. These ops ARE in the ORT Android kernel set.

```python
from onnxruntime.quantization import quantize_static, QuantType, QuantFormat

quantize_static(
    model_input="vision_model_fp32.onnx",
    model_output="vision_model_qdq.onnx",
    calibration_data_reader=my_calibration_reader,
    quant_format=QuantFormat.QDQ,   # <-- key change
    weight_type=QuantType.QInt8,
)
```

QDQ static quantization requires a calibration dataset (~100 representative images). **Expected gain:** model size 33MB → ~9MB, inference latency reduction ~40-60% on CPU.  

**Reference:** https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

---

### F2 — QNN Execution Provider (Snapdragon Hexagon NPU) *(Very high impact, higher effort)*

**Background:** Qualcomm Snapdragon SoCs (including the one in the Z Flip5) contain a dedicated **Hexagon DSP/NPU** that runs quantized neural networks at very low power. ORT ships a **QNN EP** that can delegate entire subgraphs to this hardware.  
**Expected gain:** 10–50× lower latency for vision inference vs CPU; significant battery saving during background indexing.  
**Requirement:** QDQ int8 model (see F1 above); QNN EP library bundled or sideloaded.  
**Reference:** https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html  
**Qualcomm AI Hub (pre-compiled models):** https://aihub.qualcomm.com/

---

### F3 — Replace NNAPI with full-graph QNN delegation *(High impact, medium effort once F1+F2 done)*

**Current state:** NNAPI covers 63/855 nodes. Most of the fp32 ViT graph is not NNAPI-delegatable.  
**Fix:** Once the model is in QDQ int8 format, try full-graph QNN EP delegation. QNN is far more capable than NNAPI for transformer-class workloads.

---

### F4 — Approximate Nearest Neighbour (ANN) index for large libraries *(Medium impact, low effort)*

**Current state:** Brute-force O(N) dot product at query time. Fine for ≤ 10k photos (< 5ms on device).  
**Problem:** At 50k–100k photos (large libraries), brute force becomes 25–50ms and scales linearly.  
**Fix:** HNSW index via `hnswlib` (produces a file, reload on startup) or FAISS IVFPQ for very large libraries. Alternatively, use Product Quantization to compress embeddings from 2KB to ~64 bytes each — 32× memory reduction.  
**Note:** At TinyCLIP 8M quality, ANN recall@1 is likely the bottleneck before index speed is. Profile first.

---

### F5 — FP16 vision model for NNAPI acceleration *(Medium impact, low effort)*

**Background:** NNAPI natively supports fp16 compute. The fp32 vision model can be converted to fp16 for free (no calibration data needed), halving model size to ~16MB and potentially increasing NNAPI node coverage since NNAPI handles fp16 better than fp32 for some op types.

```python
from onnxconverter_common import convert_float_to_float16
# block only embedding lookups
model_fp16 = convert_float_to_float16(model_fp32, op_block_list=["Gather"])
```

**Risk:** Some ORT Android ops don't have fp16 kernels on CPU EP — test fallback carefully.

---

### F6 — MobileCLIP or SigLIP as a drop-in replacement *(Medium impact, medium effort)*

Better zero-shot accuracy than TinyCLIP 8M at similar parameter counts:

| Model | Params | Notes |
|---|---|---|
| [MobileCLIP-S0 (Apple)](https://github.com/apple/ml-mobileclip) | ~6M | 4× faster than ViT-B/32; ONNX export available |
| [SigLIP-So400m/patch14-384 (Google)](https://huggingface.co/google/siglip-so400m-patch14-384) | 400M | Better accuracy but too large for this form factor |
| [SigLIP2 base/16 (Google)](https://huggingface.co/google/siglip2-base-patch16-224) | ~86M | Better than CLIP B/32 at same size |
| [NanoCLIP / OpenCLIP variants](https://github.com/mlfoundations/open_clip) | various | Community fine-tunes on LAION |

MobileCLIP-S0 is the most promising near-drop-in: similar projection dimension, ONNX export pipeline documented, trained on DataCompDR.

---

### F7 — Streaming / progressive photo indexing *(Low impact, low effort)*

**Current state:** WorkManager runs periodically under charging + idle constraints. New photos added between runs are not indexed until the next scheduled window.  
**Fix:** Register a `ContentObserver` on `MediaStore.Images.Media.EXTERNAL_CONTENT_URI` and trigger incremental indexing within seconds of a new photo being added.

---

### F8 — On-device fine-tuning / domain adaptation *(High impact, high effort — research territory)*

**Background:** TinyCLIP was trained on YFCC15M (general web images). For specialized domains (medical, industrial inspection, personal photo libraries), the embedding space may not align well with domain-specific queries.  
**Options:**
- **Linear probe:** train a lightweight linear projection on top of frozen CLIP embeddings using a small labeled set
- **LoRA fine-tuning on-device:** possible with ORT Training API (experimental on Android)
- **Federated learning:** fine-tune across user devices without centralising personal photos

---

### F9 — Cross-modal search (image query → ranked images) *(Low effort once pipeline exists)*

**Current state:** text → images only.  
**Extension:** Feed a photo as the query into `encodeImage()` instead of `encodeText()`. The embedding space is shared — an image query and text query produce comparable 512-d vectors. No model changes needed; UI change only.

---

### F10 — Embedding compression via Product Quantization *(Medium impact, medium effort)*

**Current state:** each embedding stored as 512 × float32 = 2048 bytes.  
**Fix:** Product Quantization (PQ) compresses 512-d float32 to ~64 bytes (32× compression) with minimal recall loss. This reduces Room DB size and speeds up dot product (integer arithmetic). `faiss` provides PQ training; the codebook can be shipped as an asset.

---

## References & External Links

### Papers

| Paper | Link |
|---|---|
| TinyCLIP: CLIP Distillation via Weight Inheritance (ICCV 2023) | https://arxiv.org/abs/2309.04504 |
| Learning Transferable Visual Models From Natural Language Supervision (CLIP, OpenAI 2021) | https://arxiv.org/abs/2103.00020 |
| MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training (Apple 2023) | https://arxiv.org/abs/2311.17049 |
| Sigmoid Loss for Language Image Pre-Training (SigLIP, Google 2023) | https://arxiv.org/abs/2303.15343 |
| Efficient Transformers: A Survey | https://arxiv.org/abs/2009.06732 |

### Models & Repos

| Resource | Link |
|---|---|
| TinyCLIP (original PyTorch) | https://github.com/wkcn/TinyCLIP |
| TinyCLIP HuggingFace ONNX community model | https://huggingface.co/onnx-community/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX |
| TinyCLIP HuggingFace original weights | https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M |
| MobileCLIP (Apple) | https://github.com/apple/ml-mobileclip |
| OpenCLIP (community CLIP variants) | https://github.com/mlfoundations/open_clip |

### ONNX Runtime

| Resource | Link |
|---|---|
| ONNX Runtime GitHub | https://github.com/microsoft/onnxruntime |
| ONNX Runtime Android docs | https://onnxruntime.ai/docs/build/android.html |
| ONNX Runtime Extensions GitHub | https://github.com/microsoft/onnxruntime-extensions |
| ORT Quantization guide | https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html |
| QNN Execution Provider | https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html |
| NNAPI Execution Provider | https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html |
| ORT issue — graph not pruned to requested outputs | https://github.com/microsoft/onnxruntime/issues/16483 |

### Android / Jetpack

| Resource | Link |
|---|---|
| WorkManager | https://developer.android.com/topic/libraries/architecture/workmanager |
| Room | https://developer.android.com/training/data-storage/room |
| Jetpack Compose | https://developer.android.com/jetpack/compose |
| Coil (async image loader) | https://github.com/coil-kt/coil |

### Qualcomm / NPU

| Resource | Link |
|---|---|
| Qualcomm AI Hub (pre-compiled edge models) | https://aihub.qualcomm.com/ |
| QNN SDK | https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk |

### This project

| Resource | Link |
|---|---|
| GitHub repo | https://github.com/agup0009Synchronoss/edge_semantic_search |
| Asset export script | `tools/export_split.py` |
| Inference engine | `app/src/main/java/com/edgesearch/app/ml/ClipEngine.kt` |
| Benchmark UI | `app/src/main/java/com/edgesearch/app/search/SearchScreen.kt` |

---

## Appendix: Key Numbers

| Metric | Value |
|---|---|
| Vision model size (fp32) | 33 MB |
| Text model size (int8) | 15 MB |
| Tokenizer model size | 1.4 MB |
| Total assets on device | ~49 MB |
| Embedding dimension | 512 |
| Sequence length (text) | 77 tokens |
| Vocabulary size | 49,408 tokens |
| Image resolution | 224 × 224 px (center-cropped) |
| Embedding storage per photo | 2,048 bytes (512 × float32) |
| Storage for 10k photos | ~20 MB (Room DB) |
| ORT version | 1.20.0 |
| ORT Extensions version | 0.13.0 |
| Android min SDK | 26 (NNAPI available) |
| Test device | Samsung Galaxy Z Flip5, Android 16 |
