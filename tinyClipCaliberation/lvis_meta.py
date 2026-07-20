"""
lvis_meta.py

Parse lvis_v1_train.json into the structures the pipeline needs:

  - categories : id -> {name, synonyms, def, image_count, instance_count,
                        frequency, synset}
  - positives  : category_id -> set(image_id)   (image has >=1 instance)
  - tag_order  : stable sorted list of the 1,203 category ids (row index used
                 by every downstream artifact)
  - image_ids  : set of all image ids referenced by the split

We parse the JSON directly (no lvis/pycocotools dependency). The train
annotation file is large (~1.5 GB); we drop the heavy per-annotation fields as
we go and keep only (image_id, category_id).

`positives_count(cat_id)` is the per-tag support used for the confidence bucket
and as the Fbeta support column.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Set

import config


@dataclass
class LvisMeta:
    categories: Dict[int, dict]
    positives: Dict[int, Set[int]]
    tag_order: List[int]
    image_ids: Set[int]

    def positives_count(self, cat_id: int) -> int:
        return len(self.positives.get(cat_id, ()))

    def bucket(self, cat_id: int) -> str:
        return config.bucket_for_count(self.positives_count(cat_id))


def load(ann_json: pathlib.Path = config.ANN_JSON) -> LvisMeta:
    ann_json = pathlib.Path(ann_json)
    if not ann_json.exists():
        raise FileNotFoundError(
            f"LVIS annotations not found: {ann_json}\n"
            f"Run 00_download_lvis.py first."
        )

    with open(ann_json, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    categories: Dict[int, dict] = {}
    for c in raw.get("categories", []):
        categories[int(c["id"])] = {
            "id": int(c["id"]),
            "name": c.get("name", ""),
            "synonyms": list(c.get("synonyms", []) or []),
            "def": c.get("def", ""),
            "image_count": int(c.get("image_count", 0)),      # LVIS-published count
            "instance_count": int(c.get("instance_count", 0)),
            "frequency": c.get("frequency", ""),               # 'r' | 'c' | 'f'
            "synset": c.get("synset", ""),
        }

    positives: Dict[int, Set[int]] = {cid: set() for cid in categories}
    image_ids: Set[int] = set()
    for a in raw.get("annotations", []):
        cid = int(a["category_id"])
        iid = int(a["image_id"])
        positives.setdefault(cid, set()).add(iid)
        image_ids.add(iid)

    # Also register every image declared in the split (even if unannotated), so
    # the vision pass and the implicit-negative pool cover the full split.
    for im in raw.get("images", []):
        image_ids.add(int(im["id"]))

    tag_order = sorted(categories.keys())
    return LvisMeta(
        categories=categories,
        positives=positives,
        tag_order=tag_order,
        image_ids=image_ids,
    )


def _selfcheck() -> None:
    """CLI sanity check: python lvis_meta.py"""
    meta = load()
    n_tags = len(meta.tag_order)
    counts = {"weak": 0, "medium": 0, "high": 0}
    empty = 0
    for cid in meta.tag_order:
        n = meta.positives_count(cid)
        counts[config.bucket_for_count(n)] += 1
        if n == 0:
            empty += 1
    print(f"tags: {n_tags}")
    print(f"images referenced: {len(meta.image_ids):,}")
    print(f"buckets: {counts}  (expected ~ weak 337 / medium 461 / high 405)")
    print(f"tags with 0 positives in this split: {empty}")


if __name__ == "__main__":
    _selfcheck()
