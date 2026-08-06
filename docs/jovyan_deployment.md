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
| RAM++ inference (`ram_tagger.py`) | **Yes, automatically.** Resolves `cuda if torch.cuda.is_available()` and reports the device in the UI status line. Measured ~1.7 s/image at 224px on CPU here; well under a second on GPU. |
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

The RAM++ app **shares by default** — it prints a public `*.gradio.live` URL you
can hand to other people, which is usually what you want on a remote box where
opening a port is awkward.

> **That link is public and unauthenticated for its ~72 h life.** Anyone holding
> it can upload images and consume your GPU. Two ways to tighten it:
> ```bash
> GRADIO_SHARE=0 ./venv_ramclip/bin/python app.py            # local only
> ```
> ```bash
> GRADIO_AUTH=team:somepassword ./venv_ramclip/bin/python app.py
> ```

The app also **warms both models up before serving**, so the first visitor does
not sit through the ~6 s checkpoint load with no feedback. `GRADIO_WARMUP=0`
disables it.

The CLIP app is unchanged and stays local unless you ask otherwise:

```bash
cd research/vitb32_benchmark && GRADIO_SHARE=1 ./venv_clip/bin/python app.py
```

To bind a port directly instead of sharing:

```bash
GRADIO_SHARE=0 GRADIO_HOST=0.0.0.0 GRADIO_PORT=7863 ./venv_ramclip/bin/python app.py
```

### Serving several people at once

RAM++ runs on the GPU automatically — `ram_tagger.py` resolves
`cuda if torch.cuda.is_available() else cpu` at construction, and the device
appears in the UI status line after every run, so you can confirm it says
`cuda:0` rather than guessing.

Requests are queued, three concurrent by default:

```bash
GRADIO_CONCURRENCY=6 GRADIO_QUEUE_SIZE=64 ./venv_ramclip/bin/python app.py
```

Raising `GRADIO_CONCURRENCY` is not free throughput. The GPU serialises the
RAM++ forward pass regardless, so the win comes from overlapping it with the
CPU-side TinyCLIP encode; past a small number you are buying queueing latency
and OOM risk. Each in-flight request holds its own activations — at 384px on a
smaller card, 3 is already reasonable.

`GRADIO_QUEUE_SIZE` bounds the backlog so a burst gets a clear "queue is full"
rather than an unbounded wait that looks like a hang.

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
| `ImportError: apply_chunking_to_forward` | transformers `>=4.56` got in — `modeling_utils.py` stopped re-exporting it from `pytorch_utils.py` between 4.55.0 and 4.56.0, well inside the 4.x line, nowhere near 5.0. `requirements.txt` pins `<4.56`. Check `venv_ramclip/bin/pip show transformers`; fix in place with `venv_ramclip/bin/pip install "transformers>=4.40,<4.56"` — no need to rebuild the venv. |
| `AttributeError: scipy.interpolate.interp2d` | SciPy ≥ 1.14. The bound is `scipy<1.14`; a `--system-site` venv may be shadowing it. |
| pip backtracks for ages on `gradio` | gradio 6 vs `huggingface_hub<1.0` are unsatisfiable together. The `gradio>=5,<6` bound prevents it — do not relax it. |
| `ModuleNotFoundError: ram` | `vendor/recognize-anything` was not cloned. Re-run the bootstrap. |
| `CERTIFICATE_VERIFY_FAILED` at model load | `ssl_bypass` imported too late, or the venv lacks it. It must precede transformers; `config` is what puts it on `sys.path`. |
| `FileNotFoundError: classifiers_templates.npy` | The one thing you must copy. See [What you must copy](#what-you-must-copy). |
| App loads, thumbnails broken | `resized_224_x_224/` missing. Untar `224.tar` into `$VG_DATA_ROOT`. |
| Page loads under JupyterHub but is blank | `GRADIO_ROOT_PATH` not set to the proxy prefix. |
| No share link printed | The tunnel binary could not be fetched, or egress is blocked. Fall back to `GRADIO_SHARE=0 GRADIO_HOST=0.0.0.0`. |
| Status line says `cpu` on a GPU box | torch has no CUDA build in that venv. `pip show torch` — a `+cpu` version means `setup_venv.sh` did not see `nvidia-smi`. Re-run it with `--gpu`. |
| Second user's request hangs behind the first | Expected up to `GRADIO_CONCURRENCY` (default 3), then queued. Raise it if you have VRAM headroom. |
| CUDA OOM under load | `GRADIO_CONCURRENCY` too high for the card, especially at 384px. Lower it. |
| `Permission denied: ./scripts/bootstrap_jovyan.sh` | `chmod +x scripts/*.sh research/*/setup_venv.sh` — some transfer paths drop the mode bit. |
| Killed with no traceback while loading RAM | OOM. See [Resource notes](#resource-notes). |
