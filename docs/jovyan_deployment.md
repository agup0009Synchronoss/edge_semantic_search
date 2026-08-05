# Running the Gradio apps on a jovyan box

Getting **RAM++ vs TinyCLIP** (`:7863`) and **TinyCLIP vs CLIP ViT-B/32**
(`:7862`) running from the command line on a Linux image — JupyterHub
`jovyan`-style, but nothing here is jovyan-specific beyond the default paths.

---

## TL;DR

```bash
git clone https://github.com/agup0009Synchronoss/edge_semantic_search.git
```

```bash
cd edge_semantic_search && ./scripts/bootstrap_jovyan.sh --ram --regenerate
```

The bootstrap creates the venvs, seeds what is committed, downloads what is
downloadable, builds what is derivable, and then **tells you exactly what is
still missing**.

With `--regenerate`, the RAM++ app needs **nothing copied at all**. The CLIP app
needs exactly one thing: the Visual Genome images. See
[What you must copy](#what-you-must-copy).

---

## Using the GPU

Worth knowing before you plan a rebuild, because the answer is not uniform:

| workload | uses GPU? |
|---|---|
| RAM++ inference (`ram_tagger.py`) | **Yes, automatically.** Already does `torch.device("cuda" if available)` — ~1.5 s/image on CPU, well under a second on GPU. |
| CLIP ViT-B/32 embedding (`build_vit_embeddings.py`) | **Yes, automatically.** torch + sentence-transformers. |
| TinyCLIP encoding — classifiers, image embeds | **No, by default.** These run on onnxruntime, which is pinned to `CPUExecutionProvider` and scales across CPU cores via a multiprocessing pool. |

That last row is deliberate. Every committed threshold in
`research/lvis_calibration/results/` was calibrated on the CPU EP, and the
Android app runs NNAPI+CPU. A different execution provider is a different
kernel implementation, not merely a different device, so switching it is a
correctness question rather than a performance one.

On a many-core box the CPU path is perfectly workable — the pool defaults to
`cpu_count()-1`. If you want the GPU anyway:

```bash
./scripts/bootstrap_jovyan.sh --ram --regenerate --ort-gpu
```

```bash
export ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider
```

**Then verify before trusting anything built that way:**

```bash
python research/common/parity_assets.py --check-providers
```

That compares CPU against your provider on the same weights over real images.
Expect `EQUIVALENT`. If it reports otherwise, rebuild on CPU — GPU-built
artifacts would be inconsistent with the committed thresholds.

---

## What ships in the clone

Committed, so it arrives with `git clone` and needs no transfer:

| | size |
|---|---|
| The ONNX encoders (both copies) | ~50 MB of distinct content |
| LVIS per-tag thresholds + tag order | ~1 MB |
| RAM tag mapping, `thresholds_4585.npy`, `tag_order_4585.json` | ~0.4 MB |
| The 4,585-tag LLM descriptions | 7.5 MB |
| Benchmark results | ~0.2 MB |

The bootstrap copies the third row into `data/text/`, which is where the app
reads them from. You do **not** need to re-run `02_build_tag_mapping.py`.

## What the box downloads for itself

| | size | fetched by |
|---|---|---|
| `ram_plus_swin_large_14m.pth` | 2.87 GB | `00_download_ram.py` (resumable) |
| `ram_tag_list.txt`, `..._threshold.txt` | 70 KB | same |
| `recognize-anything` source | 33 MB | `git clone --depth 1` |
| torch / gradio / transformers wheels | ~1 GB | `setup_venv.sh` |

## What you must copy

### RAM++ app — nothing

Everything it needs is committed, downloadable, or derivable. `--regenerate`
builds the two classifier matrices from the committed descriptions asset and
the committed templates module:

| built here | from | cost |
|---|---|---|
| `classifiers_llm.npy` (9 MB) | `assets/clip_descriptions_4585.json` | ~92k strings |
| `classifiers_templates.npy` (9 MB) | `research/common/templates.py` | ~403k strings, the slow one |

Reference timing: ~29 min for `templates` on a modest laptop pool. A server with
many cores will be substantially faster, since the work is `Pool`-parallel
across `cpu_count()-1`.

If you would rather not spend the CPU time, copy the two `.npy` (17.9 MB total)
into `research/ram_comparison/data/text/` and drop `--regenerate`.

### CLIP app — the image corpus, 2.72 GB

| file | size | destination |
|---|---|---|
| `224.tar` (the resized image corpus) | 2.72 GB | untar into `$VG_DATA_ROOT/` |

**This is the one irreducible transfer.** Visual Genome cannot be redistributed
through this repo, and both embedding files are computed *from* these images —
so if they are absent there is nothing to regenerate and the app has nothing to
display.

Given the images, `--regenerate` builds the rest on the box:

| built here | by | GPU? |
|---|---|---|
| `vg_clip_embeddings.pkl` (215 MB) | `build_vit_embeddings.py` | yes |
| `vg_tinyclip_embeddings.pkl` (216 MB) | `precompute_tinyclip.py` | no — CPU pool |

Copying those two `.pkl` (431 MB) instead is also fine and skips the compute.

> The alternative to copying `224.tar` is downloading Visual Genome from
> [visualgenome.org](https://homes.cs.washington.edu/~ranjay/visualgenome/)
> (~15 GB of originals) and resizing to 224×224 yourself. Copying 2.72 GB is
> cheaper in both bandwidth and time.

**If you only want the RAM++ app, skip this section entirely** and run `--ram`.
It stands up with zero transfers.

---

## Step by step

### 1. Clone

```bash
cd ~ && git clone https://github.com/agup0009Synchronoss/edge_semantic_search.git
```

### 2. Pick an interpreter

The venvs are built with `python3` unless told otherwise. On images with
several interpreters:

```bash
export PYTHON=python3.11
```

Python **3.10–3.12** is the tested range. If the image already carries torch
and you would rather not download it again, pass `--system-site` so the venv
can see it.

### 3. Corporate TLS

If the network intercepts certificates:

```bash
export PIP_TRUSTED=1 GIT_SSL_NO_VERIFY=1
```

`PIP_TRUSTED=1` adds the `--trusted-host` flags to every pip call. At runtime
the apps import `research/common/ssl_bypass.py` automatically, which is what
lets RAM's `init_tokenizer()` reach `bert-base-uncased`.

Prefer not to disable verification? Pre-populate the HF cache and set
`HF_HUB_OFFLINE=1`.

### 4. Bootstrap

RAM++ only, nothing to copy:

```bash
./scripts/bootstrap_jovyan.sh --ram --regenerate
```

Both apps, once `224.tar` is untarred into `$VG_DATA_ROOT`:

```bash
VG_DATA_ROOT=/home/jovyan/data/visual_genome ./scripts/bootstrap_jovyan.sh --all --regenerate
```

Idempotent — re-run it as often as you like. `--skip-venv` reuses existing
environments and only fixes up assets, which is handy when a long rebuild dies
partway: both embedding scripts checkpoint and resume.

A sensible order on a fresh box is to run **without** `--regenerate` first. That
installs the environments and reports what is missing without committing to any
long compute, so you find out the venvs are healthy before starting an hour of
embedding.

### 5. Launch

```bash
cd research/ram_comparison && ./venv_ramclip/bin/python app.py
```

```bash
cd research/vitb32_benchmark && ./venv_clip/bin/python app.py
```

Both bind `127.0.0.1` by default. To reach them from elsewhere:

```bash
GRADIO_HOST=0.0.0.0 GRADIO_PORT=7863 ./venv_ramclip/bin/python app.py
```

**On JupyterHub**, do not open a port — go through the proxy. With
`jupyter-server-proxy` installed:

```bash
GRADIO_ROOT_PATH=/user/jovyan/proxy/7863 ./venv_ramclip/bin/python app.py
```

Then browse to `https://<hub>/user/jovyan/proxy/7863/`. `GRADIO_ROOT_PATH` is
what makes Gradio emit correct asset URLs under a path prefix; without it the
page loads but the JavaScript 404s.

Keeping it alive past your terminal:

```bash
nohup ./venv_ramclip/bin/python app.py > ~/ram_app.log 2>&1 &
```

---

## Resource notes

**RAM++ is heavy and CPU-only here.** The checkpoint is ~2.8 GB on disk and
loads to roughly **4–5 GB RSS**; a single 224px inference takes ~1.5 s, several
seconds at 384, plus a ~6 s first load. Give the container **at least 8 GB**.
If it is being OOM-killed, that is why.

The CLIP app holds both embedding matrices in RAM (~430 MB) plus
sentence-transformers, so budget ~2–3 GB.

Disk, if you run both: ~3 GB checkpoint + ~3 GB VG corpus + ~2 GB of wheels
across the two venvs ≈ **9 GB**, before the clone's own ~165 MB.

---

## Verifying

```bash
./research/ram_comparison/venv_ramclip/bin/python research/common/verify_assets.py
```

```bash
cd research/ram_comparison && ./venv_ramclip/bin/python verify_env.py --full
```

`--full` loads the checkpoint and runs one real inference — the honest check
that the box can actually serve the app. Expect every line to read `PASS`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ImportError: apply_chunking_to_forward` | transformers 5.x got in. The RAM venv needs `<5` — check `pip show transformers` inside `venv_ramclip`. |
| `AttributeError: scipy.interpolate.interp2d` | SciPy ≥ 1.14. The bound is `scipy<1.14`; a `--system-site` venv may be shadowing it. |
| pip backtracks for ages on `gradio` | gradio 6 vs `huggingface_hub<1.0` are unsatisfiable together. The `gradio>=5,<6` bound prevents it — do not relax it. |
| `ModuleNotFoundError: ram` | `vendor/recognize-anything` was not cloned. Re-run the bootstrap. |
| `CERTIFICATE_VERIFY_FAILED` at model load | `ssl_bypass` imported too late, or the venv lacks it. It must precede transformers; `config` is what puts it on `sys.path`. |
| `FileNotFoundError: classifiers_templates.npy` | The one thing you must copy. See [What you must copy](#what-you-must-copy). |
| App loads, thumbnails broken | `resized_224_x_224/` missing. Untar `224.tar` into `$VG_DATA_ROOT`. |
| Page loads under JupyterHub but is blank | `GRADIO_ROOT_PATH` not set to the proxy prefix. |
| `Permission denied: ./scripts/bootstrap_jovyan.sh` | `chmod +x scripts/*.sh research/*/setup_venv.sh` — some transfer paths drop the mode bit. |
| Killed with no traceback while loading RAM | OOM. See [Resource notes](#resource-notes). |
