"""
Produces vision_model_int8.onnx, text_model_int8.onnx, and custom_op_cliptok.onnx
for the Android app.

Strategy
--------
Both models are extracted from the HF pre-quantized combined model_int8.onnx
using onnx.utils.extract_model.  No PyTorch, no fp16 conversion, no type issues.

  model_int8.onnx (23 MB, HF-tested, both towers)
      |-- extract pixel_values -> image_embeds  => vision_model_int8.onnx  (~8.7 MB)
      |-- extract input_ids + attention_mask -> text_embeds => text_model_int8.onnx  (~14 MB)

Both subgraphs are valid int8 ONNX; ORT loads them cleanly on Android and desktop.

Usage
-----
    pip install onnx onnxruntime onnxruntime-extensions==0.15.0 huggingface_hub
    python tools/export_split.py

Outputs (auto-copied to app/src/main/assets/)
-------
    tools/out/vision_model_int8.onnx
    tools/out/text_model_int8.onnx
    tools/out/custom_op_cliptok.onnx  (CLIP BPE tokenizer, vocab embedded)
"""

import sys
from pathlib import Path
import shutil

OUT   = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
ASSETS = Path(__file__).parent.parent / "app" / "src" / "main" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

HF_REPO   = "onnx-community/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX"
COMBINED  = OUT / "model_int8_combined.onnx"


# ---------------------------------------------------------------------------
# Step 1: download combined int8 model (skip if cached)
# ---------------------------------------------------------------------------
def ensure_combined():
    if COMBINED.exists():
        print(f"Using cached {COMBINED.name}  ({COMBINED.stat().st_size // 1024} KB)")
        return
    print("Downloading model_int8.onnx from HuggingFace ...")
    from huggingface_hub import hf_hub_download
    dl = hf_hub_download(repo_id=HF_REPO, filename="onnx/model_int8.onnx", local_dir=str(OUT))
    src = Path(dl)
    if src != COMBINED:
        src.rename(COMBINED)
    print(f"  -> {COMBINED}  ({COMBINED.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Step 2: extract both subgraphs
# ---------------------------------------------------------------------------
def extract(label, input_names, output_names, out_path):
    import onnx
    from onnx.utils import extract_model

    print(f"Extracting {label} subgraph ...")
    extract_model(
        input_path=str(COMBINED),
        output_path=str(out_path),
        input_names=input_names,
        output_names=output_names,
    )
    m = onnx.load(str(out_path))
    ins  = [i.name for i in m.graph.input]
    outs = [o.name for o in m.graph.output]
    size = out_path.stat().st_size // 1024
    print(f"  inputs : {ins}")
    print(f"  outputs: {outs}")
    print(f"  size   : {size} KB")
    return out_path


# ---------------------------------------------------------------------------
# Step 3: smoke-test both models with ORT
# ---------------------------------------------------------------------------
def validate(vision_path, text_path):
    import onnxruntime as ort
    import numpy as np

    print("\n--- Validation ---")

    # Vision: blank image -> non-zero embedding
    vsess   = ort.InferenceSession(str(vision_path))
    v_out   = vsess.run(["image_embeds"], {"pixel_values": np.zeros((1,3,224,224), dtype=np.float32)})[0]
    v_norm  = float(np.linalg.norm(v_out[0]))
    v_ok    = v_norm > 0
    print(f"  vision  norm={v_norm:.4f}  {'PASS' if v_ok else 'FAIL'}")

    # Text: "a photo of a cat" hardcoded ids
    tsess   = ort.InferenceSession(str(text_path))
    ids     = np.array([[49406, 320, 1125, 539, 320, 2368, 49407] + [0]*70], dtype=np.int64)
    mask    = (ids > 0).astype(np.int64)
    t_ins   = {i.name for i in tsess.get_inputs()}
    feed    = {"input_ids": ids}
    if "attention_mask" in t_ins:
        feed["attention_mask"] = mask
    t_out   = tsess.run(["text_embeds"], feed)[0]
    t_norm  = float(np.linalg.norm(t_out[0]))
    t_ok    = t_norm > 0
    print(f"  text    norm={t_norm:.4f}  {'PASS' if t_ok else 'FAIL'}")

    # Cross-modal cosine (blank image vs text — just needs to be finite)
    v_unit  = v_out[0] / (v_norm + 1e-8)
    t_unit  = t_out[0].astype(np.float32) / (t_norm + 1e-8)
    cosine  = float(np.dot(v_unit, t_unit))
    c_ok    = -1.0 <= cosine <= 1.0
    print(f"  cosine  {cosine:.4f}  {'PASS' if c_ok else 'FAIL'}")

    if not (v_ok and t_ok and c_ok):
        sys.exit("Validation FAILED — check the models.")
    print("  All checks passed.")


# ---------------------------------------------------------------------------
# Step 4: copy to assets
# ---------------------------------------------------------------------------
def copy_to_assets(*paths):
    print("\nCopying to app/src/main/assets/ ...")
    for src in paths:
        dst = ASSETS / src.name
        shutil.copy(src, dst)
        print(f"  {dst.name}  ({dst.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------

def build_tokenizer() -> Path:
    import onnxruntime_extensions as ortx
    import onnxruntime as ort
    import numpy as np

    print("\nBuilding CLIPTokenizer ONNX (vocab+merges embedded) ...")

    # Download vocab.json and merges.txt if not cached
    for fname, hf_name in [("vocab.json", "vocab.json"), ("merges.txt", "merges.txt")]:
        p = OUT / fname
        if not p.exists():
            from huggingface_hub import hf_hub_download
            dl = hf_hub_download(repo_id=HF_REPO, filename=hf_name, local_dir=str(OUT))
            src = Path(dl)
            if src != p:
                src.rename(p)

    vocab_content  = (OUT / "vocab.json").read_text(encoding="utf-8")
    merges_content = (OUT / "merges.txt").read_text(encoding="utf-8")

    graph = ortx.SingleOpGraph.build_graph(
        ortx.CLIPTokenizer,
        vocab=vocab_content,
        merges=merges_content,
    )
    import onnx as onnx_mod
    model = ortx.make_onnx_model(graph)
    out_path = OUT / "custom_op_cliptok.onnx"
    onnx_mod.save(model, str(out_path))

    # Smoke-test: "a photo of a cat" -> [49406, 320, 1125, 539, 320, 2368, 49407, ...]
    so = ort.SessionOptions()
    so.register_custom_ops_library(ortx.get_library_path())
    sess = ort.InferenceSession(str(out_path), so)
    ids = sess.run(None, {"input_text": np.array(["a photo of a cat"])})[0][0].tolist()
    expected_start = [49406, 320, 1125, 539, 320, 2368, 49407]
    ok = ids[:7] == expected_start
    print(f"  ids[:7] = {ids[:7]}  {'PASS' if ok else 'FAIL - expected ' + str(expected_start)}")
    print(f"  -> {out_path}  ({out_path.stat().st_size // 1024} KB)")
    if not ok:
        sys.exit("Tokenizer validation FAILED.")
    return out_path


if __name__ == "__main__":
    ensure_combined()

    v_path = extract(
        "vision",
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        out_path=OUT / "vision_model_int8.onnx",
    )
    t_path = extract(
        "text",
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        out_path=OUT / "text_model_int8.onnx",
    )

    tok_path = build_tokenizer()
    validate(v_path, t_path)
    copy_to_assets(v_path, t_path, tok_path)

    print("\nDone. Assets:")
    for f in sorted(ASSETS.glob("*.onnx")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
