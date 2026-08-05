"""
tinyclip_tagger.py

TinyCLIP side of the comparison: encode an image once, cosine it against the
precomputed (4585, 512) tag classifiers, and apply a per-tag threshold.

Threshold policy (project decision):
  - tags mapped to a calibrated LVIS tag use that tag's balanced threshold
  - every other tag uses one global knob supplied by the UI

The image encoder is the Android-exact TinyClipEncoder from
tinyClip_vs_ClipVit32 — reused, not reimplemented, so scores stay comparable
with the LVIS calibration that produced the thresholds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

import config


@dataclass
class TagHit:
    tag: str
    row: int
    score: float          # raw cosine similarity
    threshold: float      # the cutoff actually applied
    calibrated: bool      # True = LVIS-derived, False = global knob
    margin: float         # score - threshold

    @property
    def source(self) -> str:
        return "cal" if self.calibrated else "knob"


class TinyClipTagger:
    """Cosine-similarity tagger over the 4585 RAM++ tags."""

    def __init__(self, classifier_set: str = "templates"):
        self.tags = config.load_ram_tags()
        self.thresholds = np.load(config.THRESHOLDS_4585)  # NaN = uncalibrated
        self._classifiers: dict[str, np.ndarray] = {}
        self._encoder = None
        self.classifier_set = classifier_set
        self.load_classifiers(classifier_set)

    # ── lazy resources ────────────────────────────────────────────────────────
    @property
    def encoder(self):
        if self._encoder is None:
            from tinyclip_encoder import TinyClipEncoder
            self._encoder = TinyClipEncoder()
        return self._encoder

    def load_classifiers(self, name: str) -> np.ndarray:
        """Load and cache a classifier matrix by set name ('templates'|'llm')."""
        if name not in self._classifiers:
            path = config.CLASSIFIER_SETS[name]
            if not path.exists():
                raise FileNotFoundError(
                    f"{path.name} not built yet. Run:\n"
                    f"  python 01_build_text_classifiers.py --source {name}"
                )
            mat = np.load(path)
            if mat.shape != (len(self.tags), config.EMBED_DIM):
                raise ValueError(f"{path.name} has shape {mat.shape}, "
                                 f"expected {(len(self.tags), config.EMBED_DIM)}")
            self._classifiers[name] = mat
        return self._classifiers[name]

    def available_sets(self) -> list[str]:
        return [n for n, p in config.CLASSIFIER_SETS.items() if p.exists()]

    # ── inference ─────────────────────────────────────────────────────────────
    def effective_thresholds(self, knob: float) -> np.ndarray:
        """Per-tag cutoffs: calibrated where we have one, knob elsewhere."""
        thr = self.thresholds.copy()
        thr[~np.isfinite(thr)] = knob
        return thr

    def tag(self, image: Image.Image, knob: float,
            classifier_set: str | None = None,
            top_k: int | None = None) -> tuple[list[TagHit], dict]:
        """Return (hits above threshold, timing/debug info)."""
        name = classifier_set or self.classifier_set
        C = self.load_classifiers(name)

        t0 = time.time()
        vec = self.encoder.encode_image(image)
        t_encode = time.time() - t0

        t1 = time.time()
        scores = (C @ vec).astype(np.float32)
        thr = self.effective_thresholds(knob)
        above = np.flatnonzero(scores >= thr)
        t_score = time.time() - t1

        hits = [
            TagHit(tag=self.tags[i], row=int(i), score=float(scores[i]),
                   threshold=float(thr[i]), calibrated=bool(np.isfinite(self.thresholds[i])),
                   margin=float(scores[i] - thr[i]))
            for i in above
        ]
        # Rank by raw cosine, NOT by margin. Margin is the right thing to
        # *display* (it is comparable across models); it is the wrong thing to
        # rank by, because LVIS weak-bucket tags calibrate down to the 0.20 grid
        # floor. A mediocre 0.35 cosine against a 0.20 floor scores margin +0.15
        # and outranks 'train' at cosine 0.43 over a 0.36 threshold — which
        # buries the correct tags under junk like 'masher' and 'sawbuck'.
        hits.sort(key=lambda h: h.score, reverse=True)
        n_before_topk = len(hits)
        if top_k:
            hits = hits[:top_k]

        info = {
            "classifier_set": name,
            "encode_ms": t_encode * 1000,
            "score_ms": t_score * 1000,
            "n_above": n_before_topk,
            "n_returned": len(hits),
            "n_calibrated_hits": sum(1 for h in hits if h.calibrated),
            "score_max": float(scores.max()),
            "knob": knob,
        }
        return hits, info
