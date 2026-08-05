"""
ram_tagger.py

RAM++ side of the comparison. Wraps the vendored recognize-anything model so
the app gets the same shape of result as the TinyCLIP side: per-tag score,
per-tag threshold, and margin.

Two deviations from vendor/recognize-anything/inference_ram_plus.py, both
deliberate:

  1. We run at image_size=224 by default (project decision, ~3x faster on CPU).
     The OSS checkpoint was trained at 384 and ram_tag_list_threshold.txt was
     tuned at 384, so at 224 those cutoffs are approximate — the app surfaces
     the size in its status line so the confound stays visible.

  2. We call the tagging head directly rather than model.generate_tag(), because
     generate_tag() returns only the thresholded tag *strings* and discards the
     per-tag sigmoid scores we need to show margins.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

import config


@dataclass
class RamHit:
    tag: str
    row: int
    score: float          # sigmoid probability
    threshold: float      # RAM's own per-tag cutoff
    margin: float

    @property
    def source(self) -> str:
        return "ram"


class RamTagger:
    """RAM++ tagger over the same 4585 tags TinyCLIP scores."""

    def __init__(self, image_size: int = config.RAM_IMAGE_SIZE_DEFAULT):
        self.image_size = image_size
        self.tags = config.load_ram_tags()
        self.thresholds = np.asarray(config.load_ram_thresholds(), dtype=np.float32)
        self._model = None
        self._transform = None
        self._loaded_size = None

    # ── lazy model load (~2.8 GB checkpoint, several seconds) ─────────────────
    def _vendor_on_path(self) -> None:
        vendor = config.VENDOR_DIR / "recognize-anything"
        if not vendor.exists():
            raise FileNotFoundError(
                f"{vendor} missing. Clone it with:\n"
                f"  GIT_SSL_NO_VERIFY=1 git clone --depth 1 {config.RAM_REPO_URL} "
                f"{vendor}"
            )
        if str(vendor) not in sys.path:
            sys.path.insert(0, str(vendor))

    def load(self, image_size: int | None = None):
        size = image_size or self.image_size
        if self._model is not None and self._loaded_size == size:
            return self._model

        self._vendor_on_path()
        import torch
        from ram.models import ram_plus
        from ram import get_transform

        if not config.RAM_CHECKPOINT.exists():
            raise FileNotFoundError(
                f"{config.RAM_CHECKPOINT.name} missing. Fetch it with:\n"
                f"  python 00_download_ram.py"
            )

        t0 = time.time()
        model = ram_plus(pretrained=str(config.RAM_CHECKPOINT),
                         image_size=size, vit="swin_l")
        model.eval()
        # CPU only on this machine; ram_batch_inference.py's CUDA-stream path
        # in mldev_asset/ is unusable here.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.device)

        self._model = model
        self._transform = get_transform(image_size=size)
        self._loaded_size = size
        self.load_seconds = time.time() - t0
        return model

    # ── inference ─────────────────────────────────────────────────────────────
    def tag(self, image: Image.Image, image_size: int | None = None,
            top_k: int | None = None,
            threshold_offset: float = 0.0) -> tuple[list[RamHit], dict]:
        """Run RAM++ and return (hits above threshold, timing/debug info).

        threshold_offset shifts every per-tag cutoff, letting the UI trade
        precision for recall without discarding RAM's relative calibration.
        """
        import torch
        import torch.nn.functional as F

        size = image_size or self.image_size
        model = self.load(size)

        t0 = time.time()
        x = self._transform(image).unsqueeze(0).to(self.device)
        t_prep = time.time() - t0

        t1 = time.time()
        with torch.no_grad():
            image_embeds = model.image_proj(model.visual_encoder(x))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long,
                                    device=self.device)

            image_cls_embeds = image_embeds[:, 0, :]
            bs = image_embeds.shape[0]
            des_per_class = int(model.label_embed.shape[0] / model.num_class)

            cls = image_cls_embeds / image_cls_embeds.norm(dim=-1, keepdim=True)
            logits_per_image = (model.reweight_scale.exp() * cls @ model.label_embed.t())
            logits_per_image = logits_per_image.view(bs, -1, des_per_class)
            weight_normalized = F.softmax(logits_per_image, dim=2)

            reshaped = model.label_embed.view(-1, des_per_class, 512)
            label_embed_reweight = (weight_normalized[0].unsqueeze(-1) * reshaped).sum(dim=1)
            label_embed = torch.relu(model.wordvec_proj(label_embed_reweight)).unsqueeze(0)

            tagging_embed = model.tagging_head(
                encoder_embeds=label_embed,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=False,
                mode="tagging",
            )
            logits = model.fc(tagging_embed[0]).squeeze(-1)
            scores = torch.sigmoid(logits).squeeze(0).cpu().numpy().astype(np.float32)
        t_infer = time.time() - t1

        thr = self.thresholds + threshold_offset
        above = np.flatnonzero(scores >= thr)
        hits = [
            RamHit(tag=self.tags[i], row=int(i), score=float(scores[i]),
                   threshold=float(thr[i]), margin=float(scores[i] - thr[i]))
            for i in above
        ]
        # Rank by raw probability, matching the TinyCLIP side (see the note in
        # tinyclip_tagger.tag): margin is for display, score is for ordering.
        hits.sort(key=lambda h: h.score, reverse=True)
        n_before_topk = len(hits)
        if top_k:
            hits = hits[:top_k]

        info = {
            "image_size": size,
            "prep_ms": t_prep * 1000,
            "infer_ms": t_infer * 1000,
            "n_above": n_before_topk,
            "n_returned": len(hits),
            "score_max": float(scores.max()),
            "threshold_offset": threshold_offset,
            "device": str(self.device),
        }
        return hits, info
