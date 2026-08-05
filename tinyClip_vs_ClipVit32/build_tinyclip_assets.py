"""
build_tinyclip_assets.py  (Stage 1)

Regenerates the exact TinyCLIP ONNX encoders used by the Android app, into a
local ./assets folder (Android app assets are left untouched).

Produces:
  assets/text_model_int8.onnx     - extracted from HF model_int8.onnx (text tower)
  assets/vision_model_fp32.onnx   - PyTorch export of CLIPVisionModelWithProjection
  assets/custom_op_cliptok.onnx   - copied from the Android app assets (reused as-is)

This mirrors:
  - tools/export_split.py            (text extraction + HF source repo)
  - CONFLUENCE_POC.md lines 150-159  (vision fp32 PyTorch export)

VERIFY gate at the end:
  - tokenizer "a photo of a cat" -> [49406, 320, 1125, 539, 320, 2368, 49407]
  - vision blank-image embedding finite, dim 512
  - text embedding dim 512
"""

# ── SSL bypass — ORDER IS LOAD-BEARING ────────────────────────────────────────
# config puts research/common on sys.path and pins HF_HOME; ssl_bypass applies
# the corporate-TLS workaround. Both must land before huggingface_hub or
# transformers, which this script uses to fetch the upstream models.
import config       # noqa: F401  (sys.path wiring + HF_HOME)
import ssl_bypass   # noqa: F401  (MUST precede huggingface_hub / transformers)

import sys
import shutil
import pathlib
import numpy as np

HERE          = pathlib.Path(__file__).parent

# Writes into the canonical asset location shared with the Android app.
#
# NOTE: this OVERWRITES committed files. ONNX export is not byte-deterministic,
# so a rerun will produce different bytes for the same model and ASSETS.sha256
# will then fail. That is intended: re-run
#     python research/common/parity_assets.py
# to confirm the new export is numerically equivalent, then
#     python research/common/verify_assets.py --write
# to accept it. Do not update the manifest without checking parity first.
ASSETS        = config.ASSETS_DIR
ASSETS.mkdir(parents=True, exist_ok=True)
OUT_CACHE     = HERE / "_build_cache"
OUT_CACHE.mkdir(exist_ok=True)

HF_ONNX_REPO  = "onnx-community/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M-ONNX"
HF_PT_REPO    = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
COMBINED      = OUT_CACHE / "model_int8_combined.onnx"

# Existing Android tokenizer asset (reused as-is)
ANDROID_TOK   = config.REPO_ROOT / "app" / "src" / "main" / "assets" / "custom_op_cliptok.onnx"

TEXT_OUT      = ASSETS / "text_model_int8.onnx"
VISION_OUT    = ASSETS / "vision_model_fp32.onnx"
TOK_OUT       = ASSETS / "custom_op_cliptok.onnx"


def step(msg):
    print(f"\n=== {msg} ===", flush=True)


def ensure_combined():
    step("Step 1: download HF combined model_int8.onnx")
    if COMBINED.exists():
        print(f"  cached: {COMBINED.name} ({COMBINED.stat().st_size // 1024} KB)")
        return
    from huggingface_hub import hf_hub_download
    dl = hf_hub_download(repo_id=HF_ONNX_REPO, filename="onnx/model_int8.onnx",
                         local_dir=str(OUT_CACHE))
    src = pathlib.Path(dl)
    if src != COMBINED:
        src.replace(COMBINED)
    print(f"  -> {COMBINED} ({COMBINED.stat().st_size // 1024} KB)")


def extract_text():
    step("Step 2: extract text subgraph -> text_model_int8.onnx")
    import onnx
    from onnx.utils import extract_model
    extract_model(
        input_path=str(COMBINED),
        output_path=str(TEXT_OUT),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
    )
    m = onnx.load(str(TEXT_OUT))
    print(f"  inputs : {[i.name for i in m.graph.input]}")
    print(f"  outputs: {[o.name for o in m.graph.output]}")
    print(f"  -> {TEXT_OUT} ({TEXT_OUT.stat().st_size // 1024} KB)")


def export_vision_fp32():
    step("Step 3: export vision_model_fp32.onnx from PyTorch")
    if VISION_OUT.exists():
        print(f"  exists: {VISION_OUT.name} ({VISION_OUT.stat().st_size // 1024} KB) - skipping")
        return
    import torch
    from transformers import CLIPVisionModelWithProjection
    model = CLIPVisionModelWithProjection.from_pretrained(HF_PT_REPO).eval()
    dummy = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(VISION_OUT),
            input_names=["pixel_values"], output_names=["image_embeds"],
            opset_version=14,
        )
    print(f"  -> {VISION_OUT} ({VISION_OUT.stat().st_size // 1024} KB)")


def copy_tokenizer():
    step("Step 4: copy existing Android tokenizer asset")
    if not ANDROID_TOK.exists():
        sys.exit(f"  ERROR: Android tokenizer not found at {ANDROID_TOK}")
    shutil.copy(ANDROID_TOK, TOK_OUT)
    print(f"  -> {TOK_OUT} ({TOK_OUT.stat().st_size // 1024} KB)")


def validate():
    step("Step 5: VALIDATION GATE")
    import onnxruntime as ort
    import onnxruntime_extensions as ortx

    # 5a. tokenizer ids
    so = ort.SessionOptions()
    so.register_custom_ops_library(ortx.get_library_path())
    tok = ort.InferenceSession(str(TOK_OUT), so, providers=["CPUExecutionProvider"])
    tok_in = tok.get_inputs()[0].name
    ids = tok.run(None, {tok_in: np.array(["a photo of a cat"])})[0][0].tolist()
    expected = [49406, 320, 1125, 539, 320, 2368, 49407]
    tok_ok = ids[:7] == expected
    print(f"  tokenizer ids[:7] = {ids[:7]}  {'PASS' if tok_ok else 'FAIL expected ' + str(expected)}")

    # 5b. vision blank-image embedding
    vsess = ort.InferenceSession(str(VISION_OUT), providers=["CPUExecutionProvider"])
    v_out = vsess.run(["image_embeds"], {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)})[0]
    v_dim = v_out.shape[-1]
    v_norm = float(np.linalg.norm(v_out[0]))
    v_ok = v_dim == 512 and np.isfinite(v_norm) and v_norm > 0
    print(f"  vision dim={v_dim} norm={v_norm:.4f}  {'PASS' if v_ok else 'FAIL'}")

    # 5c. text embedding (canonical ids from export_split.py)
    tsess = ort.InferenceSession(str(TEXT_OUT), providers=["CPUExecutionProvider"])
    t_ids = np.array([[49406, 320, 1125, 539, 320, 2368, 49407] + [0] * 70], dtype=np.int64)
    t_mask = (t_ids > 0).astype(np.int64)
    t_inputs = {i.name for i in tsess.get_inputs()}
    feed = {"input_ids": t_ids}
    if "attention_mask" in t_inputs:
        feed["attention_mask"] = t_mask
    t_out = tsess.run(["text_embeds"], feed)[0]
    t_dim = t_out.shape[-1]
    t_norm = float(np.linalg.norm(t_out[0]))
    t_ok = t_dim == 512 and np.isfinite(t_norm) and t_norm > 0
    print(f"  text   dim={t_dim} norm={t_norm:.4f}  {'PASS' if t_ok else 'FAIL'}")

    print("\n-- STAGE 1 VERDICT --------------------------------------------------")
    if tok_ok and v_ok and t_ok:
        print("PASS  All TinyCLIP assets regenerated and validated.")
    else:
        sys.exit("FAIL  Asset validation failed - see above.")


if __name__ == "__main__":
    ensure_combined()
    extract_text()
    export_vision_fp32()
    copy_tokenizer()
    validate()
    print("\nAssets in:", ASSETS)
    for f in sorted(ASSETS.glob("*.onnx")):
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
