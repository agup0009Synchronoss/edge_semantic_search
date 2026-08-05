# Third-party models, datasets and libraries

This repo redistributes some upstream artifacts and downloads others at build
time. This file records what comes from where, and under what terms.

**This is a good-faith engineering summary, not legal advice.** Verify each
license against its upstream source before any redistribution or commercial use.

---

## Redistributed in this repo

These are **committed**, so cloning this repo copies them.

| Artifact | Upstream | License |
|---|---|---|
| `app/src/main/assets/*.onnx`, `research/common/assets/*.onnx` | Derived from [`onnx-community/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX`](https://huggingface.co/onnx-community/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX), original weights [`wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M`](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M) (Microsoft, [TinyCLIP paper](https://arxiv.org/abs/2309.04504)) | MIT (per the upstream model repos) |
| `custom_op_cliptok.onnx` | Built by `onnxruntime-extensions` `SingleOpGraph.build_graph(CLIPTokenizer)`; embeds the OpenAI CLIP BPE vocab + merges | MIT (onnxruntime-extensions); CLIP vocab from OpenAI CLIP, MIT |
| `research/ram_comparison/assets/clip_descriptions_4585.json` | **Generated for this project** via ChatGPT, over the RAM++ tag vocabulary | Generated content; the underlying tag list is RAM++'s (Apache-2.0) |
| `research/lvis_calibration/results/*` | **Produced by this project**, derived from LVIS-v1 annotations | Derived work — see LVIS terms below |
| `research/ram_comparison/results/*` | **Produced by this project** | Derived work |

Note on the derived results: `tag_order.json` and `lvis_to_ram_mapping.csv`
contain LVIS and RAM++ category *names*. The threshold arrays are numeric
outputs of our own calibration.

---

## Downloaded at run time (not redistributed)

Fetched by the `00_*` scripts into gitignored directories.

| Artifact | Upstream | License |
|---|---|---|
| `ram_plus_swin_large_14m.pth` (2.8 GB) | [`xinyu1205/recognize-anything-plus-model`](https://huggingface.co/xinyu1205/recognize-anything-plus-model) | Apache-2.0 |
| `recognize-anything` source (vendored into `vendor/`) | [github.com/xinyu1205/recognize-anything](https://github.com/xinyu1205/recognize-anything) | Apache-2.0 |
| `ram_tag_list.txt`, `ram_tag_list_threshold.txt` | same | Apache-2.0 |
| LVIS-v1 annotations | [lvisdataset.org](https://www.lvisdataset.org/) | **CC BY 4.0**, non-commercial research intent — check before commercial use |
| COCO train2017 images | [cocodataset.org](https://cocodataset.org/) | CC BY 4.0 for annotations; **individual images retain their original Flickr licenses** |
| Visual Genome | [visualgenome.org](https://homes.cs.washington.edu/~ranjay/visualgenome/) | CC BY 4.0 |
| `bert-base-uncased` (pulled by RAM's `init_tokenizer()`) | HuggingFace | Apache-2.0 |
| `clip-ViT-B-32` via sentence-transformers | HuggingFace | MIT (OpenAI CLIP) |

**COCO/Visual Genome images are the reason no image corpus is committed here.**
The images are third-party works under their own licenses; only pointers and
derived embeddings are produced.

---

## Runtime libraries

Declared in `app/build.gradle.kts`, `gradle/libs.versions.toml`, and the
per-workstream `requirements.txt`. Principal ones:

| Library | License |
|---|---|
| ONNX Runtime / ONNX Runtime Extensions | MIT |
| PyTorch, torchvision | BSD-3-Clause |
| transformers, huggingface_hub, timm | Apache-2.0 |
| sentence-transformers | Apache-2.0 |
| Gradio | Apache-2.0 |
| NumPy, SciPy | BSD-3-Clause |
| Pillow | MIT-CMU |
| Jetpack Compose, WorkManager, Room | Apache-2.0 |
| Coil | Apache-2.0 |

---

## Excluded internal material

`research/ram_comparison/mldev_asset/` contains internal Verizon production tag
frequencies and S3/Spark pipeline code. It is **gitignored and must never be
committed to this remote.** It is retained on local disk only as a reference for
how production RAM++ tagging is run.

---

## This project's own code

No license has been declared for this repository. Until one is added, default
copyright applies and others have no right to reuse it. If this is meant to be
shared, add a `LICENSE` file — the upstream components above are permissively
licensed and would not obstruct MIT or Apache-2.0.
