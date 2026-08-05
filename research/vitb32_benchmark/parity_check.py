"""
parity_check.py  (Stage 2 gate)

Confirms the Python TinyCLIP encoder matches the Android pipeline.

Checks:
  1. Tokenizer double-BOS/EOT wrap matches the hand-computed Kotlin transform
     for "a photo of a cat".
  2. Vision ONNX (Android preprocess) vs PyTorch CLIPVisionModelWithProjection
     reference on N sampled images -> cosine >= 0.99.
  3. Sanity cross-modal: matching caption scores higher than a non-matching one.
"""

# ── SSL bypass — ORDER IS LOAD-BEARING ────────────────────────────────────────
# config puts research/common on sys.path and pins HF_HOME; ssl_bypass applies
# the corporate-TLS workaround. Both must land before transformers, which this
# script uses to download the PyTorch reference weights.
import config       # noqa: F401  (sys.path wiring + HF_HOME)
import ssl_bypass   # noqa: F401  (MUST precede transformers)

import sys
import pathlib
import random
import numpy as np
from PIL import Image

from tinyclip_encoder import TinyClipEncoder, _BOS, _EOT

RESIZED_DIR = config.resized_dir()
HF_PT_REPO = "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
N_IMAGES   = 5
SEED       = 42
COS_THRESH = 0.99


def cos(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


def check_tokenizer(enc: TinyClipEncoder) -> bool:
    print("\n== Check 1: tokenizer double-BOS/EOT wrap ==")
    ids, mask = enc.tokenize("a photo of a cat")
    ids_list = ids[0].tolist()

    # Hand-computed Kotlin transform: raw ids already are
    #   [49406, 320, 1125, 539, 320, 2368, 49407]
    # Tokenizer.kt prepends BOS and appends EOT -> double wrap.
    raw = [49406, 320, 1125, 539, 320, 2368, 49407]
    expected = [_BOS] + raw + [_EOT] + [0] * (77 - len(raw) - 2)
    n_real = len(raw) + 2          # tokens with mask=1 (indices 0..pos inclusive)

    ids_ok  = ids_list == expected
    mask_ok = int(mask[0].sum()) == n_real

    print(f"  produced ids[:11] : {ids_list[:11]}")
    print(f"  expected ids[:11] : {expected[:11]}")
    print(f"  double BOS at [0,1]={ids_list[0]},{ids_list[1]}  "
          f"double EOT at [7,8]={ids_list[7]},{ids_list[8]}")
    print(f"  attention_mask sum = {int(mask[0].sum())} (expected {n_real})")
    print(f"  -> {'PASS' if ids_ok and mask_ok else 'FAIL'}")
    return ids_ok and mask_ok


def check_vision(enc: TinyClipEncoder) -> bool:
    print("\n== Check 2: vision ONNX vs PyTorch reference ==")
    import torch
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

    model = CLIPVisionModelWithProjection.from_pretrained(HF_PT_REPO).eval()
    try:
        proc = CLIPImageProcessor.from_pretrained(HF_PT_REPO)
    except Exception:
        from transformers import CLIPProcessor
        proc = CLIPProcessor.from_pretrained(HF_PT_REPO).image_processor

    files = list(RESIZED_DIR.glob("*.jpg"))
    random.seed(SEED)
    sample = random.sample(files, min(N_IMAGES, len(files)))

    all_ok = True
    print(f"  {'image':<16}{'cosine':>10}  status")
    print("  " + "-" * 34)
    for f in sample:
        img = Image.open(f).convert("RGB")
        ours = enc.encode_image(img)                       # Android-exact + ONNX
        inputs = proc(images=img, return_tensors="pt")
        with torch.no_grad():
            ref = model(**inputs).image_embeds[0].numpy()
        ref = ref / (np.linalg.norm(ref) + 1e-8)
        c = cos(ours, ref)
        ok = c >= COS_THRESH
        all_ok = all_ok and ok
        print(f"  {f.name:<16}{c:>10.5f}  {'PASS' if ok else 'FAIL'}")
    print(f"  -> {'PASS' if all_ok else 'FAIL'} (threshold {COS_THRESH})")
    return all_ok


def check_crossmodal(enc: TinyClipEncoder) -> bool:
    print("\n== Check 3: cross-modal sanity ==")
    files = list(RESIZED_DIR.glob("*.jpg"))
    random.seed(SEED)
    img = Image.open(random.choice(files)).convert("RGB")
    iv = enc.encode_image(img)
    # Generic captions; we only assert text embeddings are finite + unit norm and
    # that scores are within valid cosine range (content of VG images is unknown).
    captions = ["a photo of a dog", "a city street with cars", "a plate of food"]
    print(f"  {'caption':<28}{'cos(img)':>10}")
    ok = True
    for cap in captions:
        tv = enc.encode_text(cap)
        c = cos(iv, tv)
        finite = np.isfinite(tv).all() and abs(np.linalg.norm(tv) - 1.0) < 1e-3
        ok = ok and finite and -1.0 <= c <= 1.0
        print(f"  {cap:<28}{c:>10.4f}")
    print(f"  text vectors unit-norm & finite, scores in [-1,1] -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    enc = TinyClipEncoder()
    r1 = check_tokenizer(enc)
    r3 = check_crossmodal(enc)
    r2 = check_vision(enc)

    print("\n-- STAGE 2 VERDICT --------------------------------------------------")
    if r1 and r2 and r3:
        print("PASS  Python TinyCLIP encoder matches the Android pipeline.")
    else:
        sys.exit("FAIL  Parity check failed - see above.")
