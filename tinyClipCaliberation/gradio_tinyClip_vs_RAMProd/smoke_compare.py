"""
smoke_compare.py

End-to-end check that both taggers run in ONE process and produce a sensible
comparison — the exact thing app.py does, minus Gradio.

Usage:
    ./venv_ramclip/Scripts/python.exe smoke_compare.py
    ./venv_ramclip/Scripts/python.exe smoke_compare.py --image path.jpg --knob 0.36
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

import ssl_bypass  # noqa: F401  (must precede transformers import)

from PIL import Image

import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=None)
    ap.add_argument("--knob", type=float, default=config.DEFAULT_UNCALIBRATED_THRESHOLD)
    ap.add_argument("--ram-size", type=int, default=config.RAM_IMAGE_SIZE_DEFAULT)
    ap.add_argument("--top-k", type=int, default=15)
    ap.add_argument("--set", dest="cset", default="templates")
    args = ap.parse_args()

    path = (args.image or
            config.VENDOR_DIR / "recognize-anything" / "images" / "demo" / "demo4.jpg")
    img = Image.open(path)
    print(f"image: {path} {img.size}", flush=True)

    print("loading TinyCLIP...", flush=True)
    from tinyclip_tagger import TinyClipTagger
    tiny = TinyClipTagger(classifier_set=args.cset)
    t0 = time.time()
    th, ti = tiny.tag(img, knob=args.knob, top_k=args.top_k)
    print(f"  TinyCLIP done in {time.time()-t0:.1f}s", flush=True)

    print("loading RAM++ (2.8 GB, first call is slow)...", flush=True)
    from ram_tagger import RamTagger
    ram = RamTagger(image_size=args.ram_size)
    t0 = time.time()
    rh, ri = ram.tag(img, top_k=args.top_k)
    print(f"  RAM++ done in {time.time()-t0:.1f}s "
          f"(load {ram.load_seconds:.1f}s)", flush=True)

    print(f"\nRAM++    : {ri['n_above']} above thr, {ri['infer_ms']:.0f} ms "
          f"@{ri['image_size']}px, max p={ri['score_max']:.3f}")
    print(f"TinyCLIP : {ti['n_above']} above thr, {ti['encode_ms']:.0f} ms encode, "
          f"knob={ti['knob']}, max cos={ti['score_max']:.3f}, set={ti['classifier_set']}")

    rb = {h.tag: h for h in rh}
    tb = {h.tag: h for h in th}
    both = sorted(set(rb) & set(tb), key=lambda t: -(rb[t].margin + tb[t].margin))
    ro = sorted(set(rb) - set(tb), key=lambda t: -rb[t].margin)
    to = sorted(set(tb) - set(rb), key=lambda t: -tb[t].margin)

    print(f"\n-- BOTH ({len(both)}) --")
    for t in both:
        print(f"   {t:<24s} ram {rb[t].score:.3f} ({rb[t].margin:+.3f})   "
              f"tc {tb[t].score:.3f} ({tb[t].margin:+.3f}) [{tb[t].source}]")
    print(f"\n-- RAM++ ONLY ({len(ro)}) --")
    for t in ro:
        print(f"   {t:<24s} p={rb[t].score:.3f} ({rb[t].margin:+.3f})")
    print(f"\n-- TinyCLIP ONLY ({len(to)}) --")
    for t in to:
        print(f"   {t:<24s} cos={tb[t].score:.3f} ({tb[t].margin:+.3f}) [{tb[t].source}]")


if __name__ == "__main__":
    main()
