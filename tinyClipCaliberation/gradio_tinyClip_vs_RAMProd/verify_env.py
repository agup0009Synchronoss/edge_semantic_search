"""
verify_env.py

One-shot preflight for the RAM++ vs TinyCLIP app. Checks that every dependency,
artifact, and model actually loads in THIS interpreter before you try to launch
the Gradio app, and prints a per-item PASS/FAIL so a failure points at the fix.

Usage:
    ./venv_ramclip/Scripts/python.exe verify_env.py
    ./venv_ramclip/Scripts/python.exe verify_env.py --full   # also runs RAM++
"""

from __future__ import annotations

import argparse
import sys
import time

# Windows consoles default to cp1252, which cannot encode the box-drawing and
# arrow characters below. Fall back to replacement rather than dying on output.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
    pass

import ssl_bypass  # noqa: F401  (must precede transformers import)

OK, BAD = "  PASS", "  FAIL"
_failures: list[str] = []


def check(label: str, fn):
    try:
        detail = fn()
        print(f"{OK}  {label}" + (f" — {detail}" if detail else ""))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{BAD}  {label} — {type(e).__name__}: {str(e)[:160]}")
        _failures.append(label)
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also load RAM++ (2.8 GB) and run one inference")
    args = ap.parse_args()

    print(f"python {sys.version.split()[0]}\n")

    print("-- packages --")

    def _torch():
        import torch
        return f"{torch.__version__} (cuda={torch.cuda.is_available()})"
    check("torch", _torch)
    check("torchvision", lambda: __import__("torchvision").__version__)
    check("timm", lambda: __import__("timm").__version__)

    def _tf():
        import transformers
        from transformers.modeling_utils import apply_chunking_to_forward  # noqa: F401
        from transformers.file_utils import ModelOutput  # noqa: F401
        return f"{transformers.__version__} (RAM-compatible imports OK)"
    check("transformers", _tf)
    check("onnxruntime", lambda: __import__("onnxruntime").__version__)
    check("onnxruntime_extensions", lambda: __import__("onnxruntime_extensions").__version__)
    check("gradio", lambda: __import__("gradio").__version__)
    check("numpy", lambda: __import__("numpy").__version__)

    print("\n── artifacts ──")
    import config

    def _tags():
        t = config.load_ram_tags()
        assert len(t) == config.N_TAGS, f"{len(t)} tags, expected {config.N_TAGS}"
        return f"{len(t)} tags"
    check("ram_tag_list.txt", _tags)

    def _thr():
        t = config.load_ram_thresholds()
        assert len(t) == config.N_TAGS
        return f"{len(t)} thresholds, {min(t):.2f}-{max(t):.2f}"
    check("ram_tag_list_threshold.txt", _thr)

    def _ckpt():
        p = config.RAM_CHECKPOINT
        assert p.exists(), "not downloaded — run 00_download_ram.py"
        gb = p.stat().st_size / 1024**3
        assert gb > 0.5, f"suspiciously small ({gb:.2f} GB)"
        return f"{gb:.2f} GB"
    check("ram_plus_swin_large_14m.pth", _ckpt)

    def _vendor():
        p = config.VENDOR_DIR / "recognize-anything" / "ram" / "models" / "ram_plus.py"
        assert p.exists(), "vendor/recognize-anything missing — git clone it"
        return "vendored"
    check("recognize-anything source", _vendor)

    import numpy as np

    def _tc(name):
        def inner():
            p = config.CLASSIFIER_SETS[name]
            assert p.exists(), f"not built — run 01_build_text_classifiers.py --source {name}"
            m = np.load(p)
            assert m.shape == (config.N_TAGS, config.EMBED_DIM), f"shape {m.shape}"
            n = np.linalg.norm(m, axis=1)
            live = n[n > 1e-6]
            return f"{m.shape}, {len(live)} non-empty rows, norms ~{live.mean():.4f}"
        return inner
    check("classifiers_templates.npy", _tc("templates"))
    if config.CLASSIFIERS_LLM.exists():
        check("classifiers_llm.npy", _tc("llm"))
    else:
        print("  SKIP  classifiers_llm.npy — awaiting ChatGPT descriptions")

    def _map():
        t = np.load(config.THRESHOLDS_4585)
        n = int(np.isfinite(t).sum())
        fin = t[np.isfinite(t)]
        return f"{n}/{config.N_TAGS} calibrated, range {fin.min():.2f}-{fin.max():.2f}"
    check("thresholds_4585.npy", _map)

    print("\n── models ──")

    def _tiny():
        from tinyclip_tagger import TinyClipTagger
        from PIL import Image
        tg = TinyClipTagger()
        img = Image.new("RGB", (400, 300), (110, 140, 90))
        t0 = time.time()
        hits, info = tg.tag(img, knob=0.30, top_k=3)
        return (f"encode {info['encode_ms']:.0f} ms, "
                f"score {info['score_ms']:.1f} ms, {info['n_above']} hits")
    check("TinyCLIP end-to-end", _tiny)

    if args.full:
        def _ram():
            from ram_tagger import RamTagger
            from PIL import Image
            p = config.VENDOR_DIR / "recognize-anything" / "images" / "demo" / "demo1.jpg"
            img = Image.open(p) if p.exists() else Image.new("RGB", (400, 300), (110, 140, 90))
            tg = RamTagger()
            hits, info = tg.tag(img, top_k=5)
            top = ", ".join(h.tag for h in hits[:5])
            return (f"load {tg.load_seconds:.0f}s, infer {info['infer_ms']:.0f} ms "
                    f"@{info['image_size']}px, {info['n_above']} tags → {top}")
        check("RAM++ end-to-end", _ram)
    else:
        print("  SKIP  RAM++ end-to-end — pass --full to load the 2.8 GB checkpoint")

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
