"""
tinyclip_encoder.py  (Stage 2)

Android-exact TinyCLIP encoders, replicating the on-device Kotlin pipeline
byte-for-byte:

  - Image preprocess  : ImagePreprocess.kt  (resize shorter side 224, center
                        crop 224, /255, CLIP mean/std, NCHW float32)
  - Vision encoder    : vision_model_fp32.onnx  -> image_embeds -> L2 norm
  - Tokenizer         : custom_op_cliptok.onnx via onnxruntime-extensions, then
                        the EXACT Tokenizer.kt wrap (prepend BOS + append EOT to
                        the raw ids that ALREADY contain BOS/EOT -> double wrap),
                        pad/truncate to 77, build attention_mask
  - Text encoder      : text_model_int8.onnx -> text_embeds -> L2 norm

Both embeddings are L2-normalized to match ClipEngine.l2normalize().

This module is imported by parity_check.py, precompute_tinyclip.py and app.py.
"""

import pathlib
import numpy as np
from PIL import Image

import onnxruntime as ort
import onnxruntime_extensions as ortx

# ── Constants (verbatim from the Android sources) ─────────────────────────────
# ImagePreprocess.kt
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
_CROP = 224
# Tokenizer.kt
_SEQ_LEN = 77
_BOS = 49406
_EOT = 49407

_ASSETS = pathlib.Path(__file__).parent / "assets"


def _l2normalize(v: np.ndarray) -> np.ndarray:
    """Mirror ClipEngine.l2normalize: return v unchanged if norm < 1e-8."""
    norm = float(np.sqrt(np.sum(v * v)))
    if norm < 1e-8:
        return v
    return (v / norm).astype(np.float32)


class TinyClipEncoder:
    def __init__(self, assets_dir: pathlib.Path = _ASSETS, intra_op_threads: int | None = None):
        assets_dir = pathlib.Path(assets_dir)
        self.vision_path = assets_dir / "vision_model_fp32.onnx"
        self.text_path   = assets_dir / "text_model_int8.onnx"
        self.tok_path    = assets_dir / "custom_op_cliptok.onnx"

        def _opts(register_ortx: bool = False) -> ort.SessionOptions:
            o = ort.SessionOptions()
            if intra_op_threads is not None:
                o.intra_op_num_threads = intra_op_threads
            if register_ortx:
                o.register_custom_ops_library(ortx.get_library_path())
            return o

        # Vision session (CPU; Android uses NNAPI+CPU fallback, math identical)
        self.vision = ort.InferenceSession(
            str(self.vision_path), _opts(), providers=["CPUExecutionProvider"]
        )

        # Text session
        self.text = ort.InferenceSession(
            str(self.text_path), _opts(), providers=["CPUExecutionProvider"]
        )
        self.text_has_mask = "attention_mask" in {i.name for i in self.text.get_inputs()}

        # Tokenizer session (needs ORT-extensions custom op library)
        self.tok = ort.InferenceSession(
            str(self.tok_path), _opts(register_ortx=True), providers=["CPUExecutionProvider"]
        )
        self.tok_in = self.tok.get_inputs()[0].name

    # ── Image path ───────────────────────────────────────────────────────────
    @staticmethod
    def preprocess_image(img: Image.Image) -> np.ndarray:
        """Mirror ImagePreprocess.process(): NCHW float32 [1,3,224,224]."""
        img = img.convert("RGB")
        w, h = img.size
        # resize so the shorter side == 224 (preserve aspect ratio)
        if w < h:
            new_w, new_h = _CROP, (h * _CROP) // w
        else:
            new_w, new_h = (w * _CROP) // h, _CROP
        # Android uses Bitmap.createScaledBitmap(..., filter=true) == bilinear
        img = img.resize((new_w, new_h), Image.BILINEAR)
        # center crop 224x224
        left = (new_w - _CROP) // 2
        top  = (new_h - _CROP) // 2
        img = img.crop((left, top, left + _CROP, top + _CROP))

        arr = np.asarray(img, dtype=np.float32) / 255.0      # HWC, RGB, [0,1]
        arr = (arr - _MEAN) / _STD                           # per-channel norm
        arr = np.transpose(arr, (2, 0, 1))                   # CHW
        return arr[None, ...].astype(np.float32)             # NCHW [1,3,224,224]

    def encode_image(self, img: Image.Image) -> np.ndarray:
        x = self.preprocess_image(img)
        out = self.vision.run(["image_embeds"], {"pixel_values": x})[0]
        return _l2normalize(out[0].astype(np.float32))

    # ── Text path ──────────────────────────────────────────────────────────────
    def tokenize(self, text: str):
        """Replicate Tokenizer.kt EXACTLY (incl. double BOS/EOT wrap).

        Returns (input_ids[1,77] int64, attention_mask[1,77] int64).
        """
        # ClipEngine.encodeText lowercases + trims before tokenizing
        norm_text = text.lower().strip()
        raw = self.tok.run(None, {self.tok_in: np.array([norm_text])})[0]
        raw_ids = np.asarray(raw).reshape(-1).astype(np.int64).tolist()

        ids  = [0] * _SEQ_LEN
        mask = [0] * _SEQ_LEN
        ids[0] = _BOS
        pos = 1
        for tid in raw_ids:
            if pos >= _SEQ_LEN - 1:        # leave room for EOT
                break
            ids[pos] = tid
            pos += 1
        ids[pos] = _EOT
        for i in range(pos + 1):           # mask 1 for indices 0..pos inclusive
            mask[i] = 1

        ids_arr  = np.array([ids], dtype=np.int64)
        mask_arr = np.array([mask], dtype=np.int64)
        return ids_arr, mask_arr

    def encode_text(self, text: str) -> np.ndarray:
        ids, mask = self.tokenize(text)
        feed = {"input_ids": ids}
        if self.text_has_mask:
            feed["attention_mask"] = mask
        out = self.text.run(["text_embeds"], feed)[0]
        return _l2normalize(out[0].astype(np.float32))


# Convenience singleton for scripts that just want defaults
_default = None
def get_encoder() -> "TinyClipEncoder":
    global _default
    if _default is None:
        _default = TinyClipEncoder()
    return _default
