"""
00_download_lvis.py  (Step 0)

One-time dataset prep for LVIS-v1 train calibration.

  1. LVIS train annotations : lvis_v1_train.json.zip  -> data/annotations/lvis_v1_train.json
  2. COCO train2017 images  : train2017.zip           -> data/images/train2017/*.jpg
     (only the ~100K images LVIS train actually references are kept)

Both downloads are resumable (HTTP Range). The image step supports two modes:

  --mode zip        (default) download train2017.zip once, then selectively
                    extract only the members LVIS references.
  --mode per-image  skip the 18 GB zip; download each needed image from its
                    coco_url individually (threaded, resumable, disk-lean).

Corporate SSL: pass --insecure to disable TLS verification (mirrors the
verify=False pattern used elsewhere in this repo) if a proxy MITMs the CDN.

Usage:
  python 00_download_lvis.py --annotations-only
  python 00_download_lvis.py                       # ann + images (zip mode)
  python 00_download_lvis.py --mode per-image --workers 16
  python 00_download_lvis.py --limit 500           # tiny subset for a dry run
"""

from __future__ import annotations

import os
import sys
import zipfile
import argparse
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

import config

LVIS_ANN_URL = "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip"
COCO_IMG_URL = "http://images.cocodataset.org/zips/train2017.zip"
COCO_IMG_BASE = "http://images.cocodataset.org/train2017/"   # + <012d>.jpg

CHUNK = 1 << 20  # 1 MiB


# ── Generic resumable streaming download ──────────────────────────────────────
# Transient network faults we retry on (flaky VPN/proxy drops mid-stream).
_RETRYABLE = (
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
)


def _stream_once(url: str, tmp: pathlib.Path, verify: bool, desc: str) -> bool:
    """One attempt. Returns True if the transfer completed, False if it should
    be retried (resuming from the current .part size)."""
    resume = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume}-"} if resume else {}
    with requests.get(url, stream=True, headers=headers, verify=verify, timeout=(30, 300)) as r:
        if resume and r.status_code == 200:
            # server ignored Range -> restart from scratch
            resume = 0
            tmp.unlink(missing_ok=True)
        elif r.status_code not in (200, 206):
            r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + resume
        mode = "ab" if resume else "wb"
        with open(tmp, mode) as f, tqdm(
            total=total or None, initial=resume, unit="B", unit_scale=True,
            desc=desc,
        ) as bar:
            for chunk in r.iter_content(CHUNK):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        # completed if we reached the advertised total (or size unknown)
        return (not total) or tmp.stat().st_size >= total


def stream_download(url: str, dest: pathlib.Path, verify: bool = True,
                    desc: str | None = None, retries: int = 10) -> None:
    import time as _time
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [skip] already downloaded: {dest.name}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    desc = desc or dest.name
    for attempt in range(1, retries + 1):
        try:
            if _stream_once(url, tmp, verify, desc):
                break
            print(f"  [retry {attempt}/{retries}] short read; resuming from "
                  f"{tmp.stat().st_size/1e6:.0f} MB ...")
        except _RETRYABLE as e:
            have = tmp.stat().st_size if tmp.exists() else 0
            if attempt == retries:
                raise
            print(f"  [retry {attempt}/{retries}] {type(e).__name__}; resuming from "
                  f"{have/1e6:.0f} MB ...")
            _time.sleep(min(30, 2 ** attempt))
    else:
        raise RuntimeError(f"download did not complete after {retries} attempts: {url}")
    tmp.replace(dest)


# ── Annotations ───────────────────────────────────────────────────────────────
def download_annotations(verify: bool) -> None:
    if config.ANN_JSON.exists():
        print(f"[ann] present: {config.ANN_JSON}")
        return
    zip_path = config.ANN_DIR / "lvis_v1_train.json.zip"
    print("[ann] downloading LVIS train annotations ...")
    stream_download(LVIS_ANN_URL, zip_path, verify=verify, desc="lvis_v1_train.json.zip")
    print("[ann] extracting ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(config.ANN_DIR)
    if not config.ANN_JSON.exists():
        # some mirrors nest the json; find and move it
        found = next(config.ANN_DIR.rglob("lvis_v1_train.json"), None)
        if found and found != config.ANN_JSON:
            found.replace(config.ANN_JSON)
    print(f"[ann] ready: {config.ANN_JSON}")


# ── Image id set from the annotations ─────────────────────────────────────────
def wanted_image_names(limit: int = 0) -> list[str]:
    import lvis_meta
    meta = lvis_meta.load()
    ids = sorted(meta.image_ids)
    if limit:
        ids = ids[:limit]
    return [f"{iid:012d}.jpg" for iid in ids]


# ── Images: zip mode ──────────────────────────────────────────────────────────
def download_images_zip(verify: bool, limit: int, keep_zip: bool) -> None:
    wanted = set(wanted_image_names(limit))
    have = {p.name for p in config.IMAGES_DIR.glob("*.jpg")}
    todo = wanted - have
    print(f"[img] want {len(wanted):,} | have {len(have):,} | to extract {len(todo):,}")
    if not todo:
        print("[img] all wanted images already present.")
        return

    zip_path = config.DATA / "train2017.zip"
    print("[img] downloading COCO train2017.zip (~18 GB, resumable) ...")
    stream_download(COCO_IMG_URL, zip_path, verify=verify, desc="train2017.zip")

    print("[img] selectively extracting wanted members ...")
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in tqdm(zf.infolist(), desc="scan zip", unit="member"):
            base = pathlib.PurePosixPath(info.filename).name
            if base in todo:
                # extract flat into IMAGES_DIR
                with zf.open(info) as src, open(config.IMAGES_DIR / base, "wb") as dst:
                    dst.write(src.read())
                todo.discard(base)
    if not keep_zip:
        zip_path.unlink(missing_ok=True)
        print("[img] removed train2017.zip (pass --keep-zip to retain).")
    if todo:
        print(f"[img][warn] {len(todo)} wanted members not found in zip (first 5): {sorted(todo)[:5]}")


# ── Images: per-image mode ────────────────────────────────────────────────────
def _fetch_one(name: str, verify: bool) -> tuple[str, str | None]:
    dest = config.IMAGES_DIR / name
    if dest.exists():
        return name, None
    try:
        r = requests.get(COCO_IMG_BASE + name, timeout=(15, 120), verify=verify)
        r.raise_for_status()
        tmp = dest.with_suffix(".jpg.part")
        with open(tmp, "wb") as f:
            f.write(r.content)
        tmp.replace(dest)
        return name, None
    except Exception as e:  # noqa: BLE001
        return name, str(e)


def download_images_per_image(verify: bool, limit: int, workers: int) -> None:
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    wanted = wanted_image_names(limit)
    have = {p.name for p in config.IMAGES_DIR.glob("*.jpg")}
    todo = [n for n in wanted if n not in have]
    print(f"[img] want {len(wanted):,} | have {len(have):,} | to fetch {len(todo):,} | workers {workers}")
    if not todo:
        print("[img] nothing to fetch.")
        return
    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_fetch_one, n, verify) for n in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="fetch images", unit="img"):
            name, err = fut.result()
            if err:
                errors.append((name, err))
    if errors:
        print(f"[img][warn] {len(errors)} failed (first 5): {errors[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations-only", action="store_true")
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--mode", choices=["zip", "per-image"], default="zip")
    ap.add_argument("--workers", type=int, default=16, help="per-image mode only")
    ap.add_argument("--limit", type=int, default=0, help="only first N images (dry run)")
    ap.add_argument("--keep-zip", action="store_true", help="keep train2017.zip after extract")
    ap.add_argument("--insecure", action="store_true", help="disable TLS verification")
    args = ap.parse_args()

    config.ensure_dirs()
    verify = not args.insecure
    if args.insecure:
        import urllib3
        urllib3.disable_warnings()

    if not args.images_only:
        download_annotations(verify)
    if not args.annotations_only:
        if args.mode == "zip":
            download_images_zip(verify, args.limit, args.keep_zip)
        else:
            download_images_per_image(verify, args.limit, args.workers)
    print("[done] step 0 complete.")


if __name__ == "__main__":
    main()
