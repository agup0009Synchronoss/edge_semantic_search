"""
Optional parity check: compare on-device embedding values to Python reference.

Usage:
    python tools/reference_check.py --image path/to/image.jpg --device-embed path/to/embed.bin

The on-device app can dump an embedding as raw little-endian float32 bytes
(512 floats = 2048 bytes) via:
    File(filesDir, "debug_embed.bin").writeBytes(floatsToBytes(embeds))

Then pull it with:
    adb pull /data/data/com.edgesearch.app/files/debug_embed.bin tools/debug_embed.bin
    python tools/reference_check.py --image tools/test.jpg --device-embed tools/debug_embed.bin
"""

import argparse, sys
import numpy as np


def load_device_embed(path):
    raw = open(path, "rb").read()
    assert len(raw) == 512 * 4, f"Expected 2048 bytes, got {len(raw)}"
    v = np.frombuffer(raw, dtype="<f4")
    return v / (np.linalg.norm(v) + 1e-8)


def compute_reference_embed(image_path):
    try:
        import torch
        from transformers import CLIPVisionModelWithProjection, CLIPProcessor
        from PIL import Image
    except ImportError:
        sys.exit("pip install torch transformers Pillow")

    model = CLIPVisionModelWithProjection.from_pretrained(
        "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
    ).eval()
    proc = CLIPProcessor.from_pretrained("wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M")
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt")
    with torch.no_grad():
        emb = model(**inputs).image_embeds[0].numpy()
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    return emb


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",        required=True)
    p.add_argument("--device-embed", required=True)
    args = p.parse_args()

    device_emb = load_device_embed(args.device_embed)
    ref_emb    = compute_reference_embed(args.image)

    cosine = float(np.dot(device_emb, ref_emb))
    print(f"Cosine similarity (device vs reference): {cosine:.6f}")
    print("✓ PASS (≥0.99)" if cosine >= 0.99 else f"✗ FAIL — check preprocessing or quantization drift")
