"""
parity_assets.py

Numerical parity between two directories of TinyCLIP ONNX encoders.

Hashes tell you the files differ; they cannot tell you whether that matters.
ONNX export is not byte-deterministic — a re-export can change producer
metadata, node ordering, and initializer layout while computing exactly the
same function. This script answers the question that actually matters: do the
two sets produce the same embeddings?

It exists because `app/src/main/assets/` and the Python copy (then at
`tinyClip_vs_ClipVit32/assets/`, before the research/ reorg) were found to have
different hashes for the text and vision models. Since every
calibrated threshold in this repo was produced with one of those sets and the
APK ships the other, "the difference is probably harmless" was not good enough:
if they were genuinely different models, every committed threshold would be
calibrated against an encoder the app does not run.

Result on the sets as of the repo-cleanup work:
    text   — bit-identical output (max elementwise diff exactly 0.0)
    vision — max elementwise diff 4.2e-07, min cosine 0.99999994
That is ~3 float32 ulp, i.e. inert. The two locations were then made
byte-identical and are held that way by verify_assets.py.

The same machinery answers a second question: does a GPU execution provider
compute the same thing as the CPU one? An EP is a different kernel
implementation, not just a different device, so this is not a given. Run
`--check-providers` before trusting any artifact rebuilt with $ORT_PROVIDERS
set — a divergence there would silently invalidate every committed threshold.

Usage:
    python research/common/parity_assets.py                     # default A/B dirs
    python research/common/parity_assets.py --a DIR --b DIR
    python research/common/parity_assets.py --images DIR --limit 25
    python research/common/parity_assets.py --check-providers   # CPU vs $ORT_PROVIDERS
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Captions chosen to span the score range the app actually sees: easy concrete
# objects, a fine-grained pair the model is known to struggle with, and a
# natural-language query of the kind the Android UI takes.
CAPTIONS = [
    "a photo of a cat",
    "a photo of a zebra",
    "a photo of a giraffe",
    "a blurry photo of a halter top",
    "bullet train at a station",
    "a person riding a skateboard",
    "an airplane on the runway",
    "a slice of pepperoni pizza",
    "a trash can on the sidewalk",
    "sunset at the beach with friends",
]

# Cosine below this between the two sets means "different model", not "rounding".
# float32 eps is ~1.2e-07, so a genuine re-export lands far above this.
COS_TOLERANCE = 0.9999


def _report(label: str, cosines: list[float], max_delta: float) -> bool:
    ok = min(cosines) >= COS_TOLERANCE
    print(
        f"  -> {label}: min cos={min(cosines):.8f}  mean={np.mean(cosines):.8f}  "
        f"max|delta|={max_delta:.3e}  {'EQUIVALENT' if ok else 'DIFFERENT'}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", type=pathlib.Path, default=REPO_ROOT / "app/src/main/assets")
    ap.add_argument("--b", type=pathlib.Path, default=REPO_ROOT / "research/common/assets")
    ap.add_argument(
        "--images",
        type=pathlib.Path,
        default=REPO_ROOT / "research/lvis_calibration/data/images/train2017",
        help="directory of .jpg to encode; falls back to seeded synthetic images",
    )
    ap.add_argument("--limit", type=int, default=12, help="how many images to compare")
    ap.add_argument(
        "--check-providers",
        action="store_true",
        help="compare CPU against $ORT_PROVIDERS on the SAME assets, instead of "
             "comparing two asset directories",
    )
    args = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from tinyclip_encoder import TinyClipEncoder  # noqa: E402
    import onnxruntime as ort  # noqa: E402

    if args.check_providers:
        import os
        wanted = os.environ.get("ORT_PROVIDERS", "").strip()
        if not wanted:
            print("ERROR: --check-providers needs $ORT_PROVIDERS set, e.g.\n"
                  "  export ORT_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider",
                  file=sys.stderr)
            return 1
        gpu = [p.strip() for p in wanted.split(",") if p.strip()]
        unavailable = [p for p in gpu if p not in set(ort.get_available_providers())]
        if unavailable:
            print(f"ERROR: provider(s) not available in this onnxruntime build: "
                  f"{', '.join(unavailable)}\n"
                  f"  available: {', '.join(ort.get_available_providers())}\n"
                  f"  (install onnxruntime-gpu in place of onnxruntime)", file=sys.stderr)
            return 1
        assets = args.b if args.b.is_dir() else args.a
        print(f"assets  = {assets}\nA = CPUExecutionProvider\nB = {', '.join(gpu)}\n"
              "loading encoders...")
        ea = TinyClipEncoder(assets_dir=assets, intra_op_threads=4,
                             providers=["CPUExecutionProvider"])
        eb = TinyClipEncoder(assets_dir=assets, intra_op_threads=4, providers=gpu)
    else:
        for d in (args.a, args.b):
            if not d.is_dir():
                print(f"ERROR: asset dir not found: {d}", file=sys.stderr)
                return 1

        print(f"A = {args.a}\nB = {args.b}\nloading encoders...")
        ea = TinyClipEncoder(assets_dir=args.a, intra_op_threads=4)
        eb = TinyClipEncoder(assets_dir=args.b, intra_op_threads=4)

    print(f"\n=== TEXT ({len(CAPTIONS)} captions) ===")
    tcos, tmax = [], 0.0
    for c in CAPTIONS:
        va, vb = ea.encode_text(c), eb.encode_text(c)
        tcos.append(float(np.dot(va, vb)))
        tmax = max(tmax, float(np.abs(va - vb).max()))
    ok_text = _report("text", tcos, tmax)

    if args.images.is_dir():
        paths = sorted(args.images.glob("*.jpg"))[: args.limit]
        samples = [(p.name, Image.open(p)) for p in paths]
    else:
        samples = []
    if not samples:
        print(f"\n(no images under {args.images} — using seeded synthetic patterns)")
        rng = np.random.default_rng(0)
        samples = [
            (f"synthetic_{i}", Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)))
            for i in range(args.limit)
        ]

    print(f"\n=== VISION ({len(samples)} images) ===")
    vcos, vmax = [], 0.0
    for _name, im in samples:
        va, vb = ea.encode_image(im), eb.encode_image(im)
        vcos.append(float(np.dot(va, vb)))
        vmax = max(vmax, float(np.abs(va - vb).max()))
    ok_vision = _report("vision", vcos, vmax)

    print("\n=== VERDICT ===")
    if ok_text and ok_vision:
        if args.check_providers:
            print("  EQUIVALENT — the GPU kernels agree with CPU on this model.")
            print("  Artifacts rebuilt with $ORT_PROVIDERS set are safe to trust.")
        else:
            print("  EQUIVALENT — any byte difference is inert export metadata.")
        return 0
    if args.check_providers:
        print("  *** NOT EQUIVALENT — the GPU kernels disagree with CPU. ***")
        print("  Do NOT rebuild artifacts with this provider: the thresholds in")
        print("  research/lvis_calibration/results/ were calibrated on CPU.")
    else:
        print("  *** NOT EQUIVALENT — these are different models. ***")
        print("  Thresholds calibrated with one set are not valid for the other.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
