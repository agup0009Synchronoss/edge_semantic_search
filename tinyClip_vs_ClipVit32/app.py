"""
app.py  (Stage 4)

Gradio dual-lane text-to-image semantic search over the 108K Visual Genome
images, comparing:

  - CLIP ViT-B/32 : precomputed vg_clip_embeddings.pkl  (text via sentence-
                    transformers clip-ViT-B-32, the exact model that produced
                    the image embeddings)
  - TinyCLIP      : precomputed vg_tinyclip_embeddings.pkl  (text via the
                    Android-exact ONNX pipeline in tinyclip_encoder.py)

Each query is embedded by BOTH text encoders and matched ONLY against its own
lane's precomputed image embeddings (both are 512-d but live in different
spaces - never cross-compare). Top-N results are shown side by side with
cosine scores.
"""

# ── SSL bypass (sentence-transformers / HF model load) ────────────────────────
import os, ssl, urllib3, warnings
os.environ['PYTHONHTTPSVERIFY']               = '0'
os.environ['REQUESTS_CA_BUNDLE']              = ''
os.environ['CURL_CA_BUNDLE']                  = ''
os.environ['SSL_CERT_FILE']                   = ''
os.environ['HF_HUB_DISABLE_SSL_VERIFICATION'] = '1'
os.environ['HF_HOME']                         = './hf_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_VERIFY_SSL']         = '0'
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context
import requests as _rq
_OrigSession = _rq.Session
class _NoSSLSession(_OrigSession):
    def __init__(self):
        super().__init__(); self.verify = False
_rq.Session = _NoSSLSession
try:
    import httpx as _httpx
    _OC = _httpx.Client
    class _NC(_OC):
        def __init__(self, *a, **k): k['verify']=False; super().__init__(*a, **k)
    _httpx.Client = _NC
    _OA = _httpx.AsyncClient
    class _NA(_OA):
        def __init__(self, *a, **k): k['verify']=False; super().__init__(*a, **k)
    _httpx.AsyncClient = _NA
except ImportError:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import time
import pickle
import pathlib
import numpy as np
import gradio as gr

HERE        = pathlib.Path(__file__).parent

# Paths can be overridden with environment variables so the same code runs on
# any machine without editing this file:
#   export VG_RESIZED_DIR=/path/to/resized_224_x_224
#   export VG_VIT_PKL=/path/to/vg_clip_embeddings.pkl
#   export VG_TINY_PKL=/path/to/vg_tinyclip_embeddings.pkl
RESIZED_DIR = pathlib.Path(os.environ.get(
    "VG_RESIZED_DIR",
    r"C:\Users\agup0009\code\edge_object_detection\data\visual_genome\resized_224_x_224"
))
VIT_PKL = pathlib.Path(os.environ.get(
    "VG_VIT_PKL",
    r"C:\Users\agup0009\code\edge_object_detection\data\visual_genome\vg_clip_embeddings.pkl"
))
TINY_PKL = pathlib.Path(os.environ.get(
    "VG_TINY_PKL",
    str(HERE / "vg_tinyclip_embeddings.pkl")
))


def _load_lane(pkl_path: pathlib.Path):
    """Load a pkl dict {name: vec} into (filenames, L2-normalized (N,512) matrix)."""
    with open(pkl_path, "rb") as fh:
        data = pickle.load(fh)
    names = list(data.keys())
    mat = np.stack([np.asarray(data[n], dtype=np.float32).reshape(-1) for n in names])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    mat = mat / norms
    return names, mat.astype(np.float32)


print("Loading embedding lanes ...")
VIT_NAMES, VIT_MAT = _load_lane(VIT_PKL)
print(f"  ViT-B/32 : {len(VIT_NAMES):,} x {VIT_MAT.shape[1]}")
TINY_NAMES, TINY_MAT = _load_lane(TINY_PKL)
print(f"  TinyCLIP : {len(TINY_NAMES):,} x {TINY_MAT.shape[1]}")

# Build name→index lookups and a shared sorted name list for correct subset sampling.
# Sampling by filename (not by raw index) guarantees both models consider the
# exact same image files regardless of insertion order in their respective pkls.
VIT_NAME_TO_IDX  = {n: i for i, n in enumerate(VIT_NAMES)}
TINY_NAME_TO_IDX = {n: i for i, n in enumerate(TINY_NAMES)}
COMMON_NAMES = sorted(set(VIT_NAMES) & set(TINY_NAMES))
print(f"  Common images available for subset sampling: {len(COMMON_NAMES):,}")

# ── Text encoders ──────────────────────────────────────────────────────────────
print("Loading text encoders ...")
from sentence_transformers import SentenceTransformer
VIT_TEXT = SentenceTransformer("clip-ViT-B-32")

from tinyclip_encoder import TinyClipEncoder
TINY_ENC = TinyClipEncoder()
print("Ready.")


def _vit_text_vec(query: str) -> np.ndarray:
    v = VIT_TEXT.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else v


def _tiny_text_vec(query: str) -> np.ndarray:
    return TINY_ENC.encode_text(query)  # already L2-normalized


def _search(text_vec, names, mat, top_n, min_cosim):
    scores = mat @ text_vec
    # Filter by cosine threshold first
    mask = scores >= min_cosim
    filtered_idx = np.where(mask)[0]
    if len(filtered_idx) == 0:
        return []
    filtered_scores = scores[filtered_idx]
    # Sort and take top N
    sorted_order = np.argsort(-filtered_scores)
    top_k = min(top_n, len(sorted_order))
    final_idx = filtered_idx[sorted_order[:top_k]]

    out = []
    for i in final_idx:
        path = str(RESIZED_DIR / names[i])
        out.append((path, f"{names[i]}\ncos={scores[i]:.3f}"))
    return out


def _split_results(vit_res, tiny_res):
    """
    Splits two ranked result lists into three *disjoint* sets keyed by image path:
      common    – in both; caption shows both scores; sorted by avg-score desc
      vit_only  – exclusive to ViT; original rank order preserved
      tiny_only – exclusive to TinyCLIP; original rank order preserved

    Because the three sets are disjoint, each image appears in at most one
    comparison gallery.  Images that also appear in the per-model galleries above
    are cached by the browser after the first load, so there is no duplicate I/O.
    """
    vit_dict  = {path: cap for path, cap in vit_res}
    tiny_dict = {path: cap for path, cap in tiny_res}

    common_paths    = set(vit_dict) & set(tiny_dict)
    vit_only_paths  = set(vit_dict) - common_paths
    tiny_only_paths = set(tiny_dict) - common_paths

    def _score(cap):
        try:
            return float(cap.rsplit("cos=", 1)[-1])
        except ValueError:
            return 0.0

    # Intersection: merged caption, sorted by avg cosine desc
    common = []
    for path in common_paths:
        v = _score(vit_dict[path])
        t = _score(tiny_dict[path])
        fname = pathlib.Path(path).name
        common.append((path, f"{fname}\nViT={v:.3f} | Tiny={t:.3f}", (v + t) / 2))
    common.sort(key=lambda x: -x[2])
    common = [(p, cap) for p, cap, _ in common]

    # Exclusives: preserve the rank order already set by _search
    vit_only  = [(p, vit_dict[p])  for p, _ in vit_res  if p in vit_only_paths]
    tiny_only = [(p, tiny_dict[p]) for p, _ in tiny_res if p in tiny_only_paths]

    return common, vit_only, tiny_only


def _sample_common_names(n: int, seed: int):
    """
    Sample n filenames from COMMON_NAMES using a stable seed.
    Returns (sampled_names, vit_row_indices, tiny_row_indices), or
    (None, None, None) if n >= len(COMMON_NAMES) (use full dataset).
    Both index arrays point into the same image files, just at their
    respective positions in each model's embedding matrix.
    """
    total = len(COMMON_NAMES)
    n = int(n)
    if n <= 0 or n >= total:
        return None, None, None
    rng = np.random.RandomState(seed)
    chosen_names = rng.choice(COMMON_NAMES, size=n, replace=False).tolist()
    vit_idx  = np.array([VIT_NAME_TO_IDX[name]  for name in chosen_names], dtype=np.intp)
    tiny_idx = np.array([TINY_NAME_TO_IDX[name] for name in chosen_names], dtype=np.intp)
    return chosen_names, vit_idx, tiny_idx


def run_query(query: str, top_n: int, total_images: int, vit_min_cos: float, tiny_min_cos: float):
    query = (query or "").strip()
    if not query:
        return [], [], "Enter a query.", [], [], []
    top_n = int(top_n)
    total_images = int(total_images)
    vit_min_cos = float(vit_min_cos)
    tiny_min_cos = float(tiny_min_cos)

    # Stable seed from query text → same query always draws the same subset
    seed = abs(hash(query)) % (2 ** 31)

    chosen_names, vit_idx, tiny_idx = _sample_common_names(total_images, seed)

    if chosen_names is not None:
        # Subset: both models search the exact same image files
        vit_names_use  = chosen_names
        tiny_names_use = chosen_names
        vit_mat_use    = VIT_MAT[vit_idx]
        tiny_mat_use   = TINY_MAT[tiny_idx]
    else:
        vit_names_use  = VIT_NAMES
        tiny_names_use = TINY_NAMES
        vit_mat_use    = VIT_MAT
        tiny_mat_use   = TINY_MAT

    pool_size = len(vit_names_use)

    t0 = time.perf_counter()
    vit_vec = _vit_text_vec(query)
    t1 = time.perf_counter()
    tiny_vec = _tiny_text_vec(query)
    t2 = time.perf_counter()

    vit_res  = _search(vit_vec,  vit_names_use,  vit_mat_use,  top_n, vit_min_cos)
    tiny_res = _search(tiny_vec, tiny_names_use, tiny_mat_use, top_n, tiny_min_cos)
    t3 = time.perf_counter()

    common_res, vit_only_res, tiny_only_res = _split_results(vit_res, tiny_res)

    pool_label = (
        f"searched {pool_size:,} / {len(COMMON_NAMES):,} images (random seed {seed})"
        if chosen_names is not None
        else f"searched all {pool_size:,} images"
    )
    status = (
        f"Query: '{query}'  |  top {top_n}  |  {pool_label}  |  "
        f"ViT-B/32: min_cos={vit_min_cos:.2f} → {len(vit_res)} results  |  "
        f"TinyCLIP: min_cos={tiny_min_cos:.2f} → {len(tiny_res)} results  |  "
        f"common={len(common_res)}  vit_only={len(vit_only_res)}  tiny_only={len(tiny_only_res)}  |  "
        f"ViT-B/32 text {1000*(t1-t0):.0f} ms, TinyCLIP text {1000*(t2-t1):.0f} ms, "
        f"rank {1000*(t3-t2):.0f} ms"
    )
    return vit_res, tiny_res, status, common_res, vit_only_res, tiny_only_res


with gr.Blocks(title="TinyCLIP vs CLIP ViT-B/32 - Visual Genome search") as demo:
    gr.Markdown(
        "# TinyCLIP vs CLIP ViT-B/32 - Visual Genome semantic search\n"
        "Type a natural-language query. Each lane embeds the text with its own "
        "encoder and ranks its own precomputed image embeddings by cosine similarity.\n\n"
        "**Left lane**: sentence-transformers `clip-ViT-B-32` (server-side production)  |  "
        "**Right lane**: Android-exact TinyCLIP ONNX (fp32 vision + int8 text + double BOS/EOT tokenizer wrap)"
    )
    with gr.Row():
        query_box = gr.Textbox(
            label="Query", placeholder="me and my dog playing in the snow",
            scale=3, autofocus=True,
        )
        top_n = gr.Number(value=20, label="Top N (max results)", precision=0, scale=1)
        total_images = gr.Number(
            value=108_000, label="Total images to consider (random subset)",
            precision=0, minimum=1, scale=1,
            info="Reduce to search a random sample. Same subset used for both models."
        )
        go = gr.Button("Search", variant="primary", scale=1)
    with gr.Row():
        vit_min_cos = gr.Slider(0.0, 1.0, value=0.20, step=0.01,
                                label="ViT-B/32 min cosine", scale=1)
        tiny_min_cos = gr.Slider(0.0, 1.0, value=0.30, step=0.01,
                                 label="TinyCLIP min cosine", scale=1)
    status = gr.Markdown()

    gr.Markdown("## Per-model ranked results")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### CLIP ViT-B/32 (server-side production)")
            vit_gallery = gr.Gallery(label="ViT-B/32 — all top-N", columns=4, height=800, object_fit="contain")
        with gr.Column():
            gr.Markdown("### TinyCLIP ViT-8M (on-device / Android-exact)")
            tiny_gallery = gr.Gallery(label="TinyCLIP — all top-N", columns=4, height=800, object_fit="contain")

    gr.Markdown(
        "## Agreement / disagreement analysis\n"
        "The three galleries below are **disjoint** — each image appears in exactly one."
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🤝 Common to both models")
            common_gallery = gr.Gallery(
                label="Common (both models agreed)",
                columns=4, height=700, object_fit="contain",
            )
        with gr.Column():
            gr.Markdown("### 🔵 Exclusive to ViT-B/32")
            vit_only_gallery = gr.Gallery(
                label="ViT-B/32 exclusive",
                columns=4, height=700, object_fit="contain",
            )
        with gr.Column():
            gr.Markdown("### 🟢 Exclusive to TinyCLIP")
            tiny_only_gallery = gr.Gallery(
                label="TinyCLIP exclusive",
                columns=4, height=700, object_fit="contain",
            )

    inputs  = [query_box, top_n, total_images, vit_min_cos, tiny_min_cos]
    outputs = [vit_gallery, tiny_gallery, status, common_gallery, vit_only_gallery, tiny_only_gallery]
    go.click(run_query, inputs, outputs)
    query_box.submit(run_query, inputs, outputs)


if __name__ == "__main__":
    # Set GRADIO_SHARE=1  to get a public tunnelled URL (e.g. on Jovyan/k8s).
    # Set GRADIO_HOST=0.0.0.0 to bind to all interfaces (needed behind k8s ingress).
    # Set GRADIO_PORT=XXXX  to override the default port.
    _share = os.environ.get("GRADIO_SHARE", "0") == "1"
    _host  = os.environ.get("GRADIO_HOST", "127.0.0.1")
    _port  = int(os.environ.get("GRADIO_PORT", "7862"))

    demo.launch(
        server_name=_host,
        server_port=_port,
        share=_share,
        show_error=True,
        allowed_paths=[str(RESIZED_DIR)],
    )
