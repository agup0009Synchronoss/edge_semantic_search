"""
verify_assets.py

Guards the committed ONNX encoders against silent drift.

The repo tracks the same three encoders in more than one place, because Gradle
can only package assets from `app/src/main/assets/` while the Python research
code needs them importable from `research/common/assets/`. Duplication is the
price; undetected divergence is not.

That divergence already happened once: `app/src/main/assets/` and
`tinyClip_vs_ClipVit32/assets/` drifted to different bytes and nobody noticed,
which quietly put the "Android-exact encoder" claim behind every calibrated
threshold in doubt. A parity check showed the two were numerically equivalent
(text bit-identical, vision within 4.2e-07 — float32 rounding), so nothing was
actually wrong, but it took a 100k-image investigation to establish that.
This script exists so the next drift is caught in one second instead.

Usage:
    python research/common/verify_assets.py            # check against ASSETS.sha256
    python research/common/verify_assets.py --write    # regenerate the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ASSETS.sha256"

# Every tracked location of the ONNX encoders, relative to the repo root.
ASSET_DIRS = [
    pathlib.Path("app/src/main/assets"),
    pathlib.Path("research/common/assets"),
]
ASSET_NAMES = [
    "custom_op_cliptok.onnx",
    "text_model_int8.onnx",
    "vision_model_fp32.onnx",
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_assets() -> list[pathlib.Path]:
    """Every expected asset path, repo-root-relative, in a stable order."""
    return [d / n for d in ASSET_DIRS for n in ASSET_NAMES]


def write_manifest() -> int:
    lines = []
    for rel in tracked_assets():
        p = REPO_ROOT / rel
        if not p.is_file():
            print(f"MISSING  {rel}", file=sys.stderr)
            return 1
        lines.append(f"{sha256(p)}  {rel.as_posix()}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)} ({len(lines)} entries)")
    return 0


def check_manifest() -> int:
    if not MANIFEST.is_file():
        print(f"ERROR: {MANIFEST} not found. Run with --write to create it.", file=sys.stderr)
        return 1

    expected: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        expected[rel.strip()] = digest

    failures = 0
    for rel in tracked_assets():
        key = rel.as_posix()
        p = REPO_ROOT / rel
        if key not in expected:
            print(f"UNLISTED {key} — present on disk but not in the manifest")
            failures += 1
            continue
        if not p.is_file():
            print(f"MISSING  {key}")
            failures += 1
            continue
        actual = sha256(p)
        if actual != expected[key]:
            print(f"MISMATCH {key}\n         expected {expected[key]}\n         actual   {actual}")
            failures += 1
        else:
            print(f"OK       {key}")

    # All copies of a given filename must agree — that is the whole point.
    for name in ASSET_NAMES:
        digests = {
            expected.get((d / name).as_posix())
            for d in ASSET_DIRS
            if (d / name).as_posix() in expected
        }
        if len(digests) > 1:
            print(f"DIVERGED {name} — copies do not share a hash")
            failures += 1

    if failures:
        print(f"\n{failures} problem(s). If the change was intentional, re-run with --write.")
        return 1
    print(f"\nAll {len(tracked_assets())} assets match, and every copy agrees.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="regenerate ASSETS.sha256 from what is on disk")
    args = ap.parse_args()
    return write_manifest() if args.write else check_manifest()


if __name__ == "__main__":
    sys.exit(main())
