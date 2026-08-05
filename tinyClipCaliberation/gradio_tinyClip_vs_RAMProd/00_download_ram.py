"""
00_download_ram.py

Fetch the RAM++ artifacts we do not vendor:

  - ram_plus_swin_large_14m.pth  (~3 GB) from HuggingFace
  - ram_tag_list.txt / ram_tag_list_threshold.txt (4585 each) from GitHub raw
    (already staged, re-fetched with --tags to refresh)

Corporate TLS interception means the default certificate chain fails, so this
mirrors the SSL-bypass approach used elsewhere in the repo
(tinyClip_vs_ClipVit32/app.py, build_tinyclip_assets.py).

Downloads are resumable: an interrupted transfer restarts from the byte offset
already on disk via an HTTP Range header, so a dropped 3 GB download does not
start over.

Usage:
    python 00_download_ram.py            # checkpoint (+ tags if missing)
    python 00_download_ram.py --tags     # only refresh the tag lists
"""

from __future__ import annotations

import argparse
import os
import pathlib
import ssl
import sys
import time
import urllib.request

import config

# ── SSL bypass (corporate TLS interception) ───────────────────────────────────
os.environ["PYTHONHTTPSVERIFY"] = "0"
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_UA = {"User-Agent": "Mozilla/5.0 (edge_semantic_search RAM++ fetcher)"}
CHUNK = 1 << 20  # 1 MiB


def _fmt(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def download(url: str, dest: pathlib.Path, expect_min_bytes: int = 0,
             retries: int = 5) -> None:
    """Resumable download with retry. Writes to dest.part then renames."""
    dest = pathlib.Path(dest)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url, headers=dict(_UA))
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, context=_SSL_CTX, timeout=120) as r:
                # If the server ignored our Range, start clean.
                if have and r.status == 200:
                    have = 0
                    part.unlink(missing_ok=True)
                total = int(r.headers.get("Content-Length", 0)) + have
                mode = "ab" if have else "wb"
                t0, last = time.time(), have
                with part.open(mode) as f:
                    while True:
                        buf = r.read(CHUNK)
                        if not buf:
                            break
                        f.write(buf)
                        have += len(buf)
                        now = time.time()
                        if now - t0 > 2:
                            rate = (have - last) / (now - t0)
                            pct = f"{100*have/total:.1f}%" if total else "?"
                            sys.stdout.write(
                                f"\r  {_fmt(have)} / {_fmt(total) if total else '?'}"
                                f"  {pct}  {_fmt(rate)}/s   "
                            )
                            sys.stdout.flush()
                            t0, last = now, have
            print()
            if expect_min_bytes and part.stat().st_size < expect_min_bytes:
                raise IOError(
                    f"short file: {part.stat().st_size} < {expect_min_bytes} expected"
                )
            part.replace(dest)
            print(f"  -> {dest.name} ({_fmt(dest.stat().st_size)})")
            return
        except Exception as e:  # noqa: BLE001
            print(f"\n  attempt {attempt}/{retries} failed: {type(e).__name__}: {e}")
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def fetch_tags() -> None:
    for name in ("ram_tag_list.txt", "ram_tag_list_threshold.txt"):
        dest = config.RAM_DIR / name
        print(f"[tags] {name}")
        download(config.RAM_RAW_BASE + f"ram/data/{name}", dest)
        n = len([l for l in dest.read_text(encoding="utf8").splitlines() if l.strip()])
        print(f"  {n} entries")
        if n != config.N_TAGS:
            print(f"  ! expected {config.N_TAGS}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", action="store_true", help="only refresh the tag lists")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    config.ensure_dirs()

    if args.tags:
        fetch_tags()
        return

    if not config.RAM_TAG_LIST.exists() or not config.RAM_TAG_THRESHOLDS.exists():
        fetch_tags()

    ckpt = config.RAM_CHECKPOINT
    if ckpt.exists() and not args.force:
        print(f"[ckpt] already present: {ckpt.name} ({_fmt(ckpt.stat().st_size)})")
        return

    print(f"[ckpt] {config.RAM_CKPT_URL}")
    print("  ~3 GB, resumable — safe to re-run if interrupted")
    # Guard against silently saving an HTML error page as a 'checkpoint'.
    download(config.RAM_CKPT_URL, ckpt, expect_min_bytes=500 * 1024 * 1024)


if __name__ == "__main__":
    main()
