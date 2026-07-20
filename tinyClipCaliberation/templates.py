"""
templates.py

Text-side classifier construction inputs:

  - OPENAI_80        : the canonical 80 CLIP/ImageNet zero-shot prompt templates
                       (Radford et al. 2021, the set that adds ~+3.5 pts over a
                       single template).
  - LVIS_TEMPLATES   : a few object-presence templates tuned for LVIS tags.
  - QUESTION_TEMPLATES : VQA-style questions.

Each template is a Python format string with a single "{}" slot filled by a
label string (a tag name or one of its synonyms).

`build_tag_strings()` turns one LVIS category into the full grouped set of text
strings (prompts / descriptions / questions). This is the single extension point
where LLM-generated text can later be appended as an extra source.
"""

from __future__ import annotations

from typing import Dict, List

# ── The 80 OpenAI ImageNet zero-shot templates ────────────────────────────────
OPENAI_80: List[str] = [
    "a bad photo of a {}.",
    "a photo of many {}.",
    "a sculpture of a {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of the {}.",
    "a rendering of a {}.",
    "graffiti of a {}.",
    "a bad photo of the {}.",
    "a cropped photo of the {}.",
    "a tattoo of a {}.",
    "the embroidered {}.",
    "a photo of a hard to see {}.",
    "a bright photo of a {}.",
    "a photo of a clean {}.",
    "a photo of a dirty {}.",
    "a dark photo of the {}.",
    "a drawing of a {}.",
    "a photo of my {}.",
    "the plastic {}.",
    "a photo of the cool {}.",
    "a close-up photo of a {}.",
    "a black and white photo of the {}.",
    "a painting of the {}.",
    "a painting of a {}.",
    "a pixelated photo of the {}.",
    "a sculpture of the {}.",
    "a bright photo of the {}.",
    "a cropped photo of a {}.",
    "a plastic {}.",
    "a photo of the dirty {}.",
    "a jpeg corrupted photo of a {}.",
    "a blurry photo of the {}.",
    "a photo of the {}.",
    "a good photo of the {}.",
    "a rendering of the {}.",
    "a {} in a video game.",
    "a photo of one {}.",
    "a doodle of a {}.",
    "a close-up photo of the {}.",
    "a photo of a {}.",
    "the origami {}.",
    "the {} in a video game.",
    "a sketch of a {}.",
    "a doodle of the {}.",
    "a origami {}.",
    "a low resolution photo of a {}.",
    "the toy {}.",
    "a rendition of the {}.",
    "a photo of the clean {}.",
    "a photo of a large {}.",
    "a rendition of a {}.",
    "a photo of a nice {}.",
    "a photo of a weird {}.",
    "a blurry photo of a {}.",
    "a cartoon {}.",
    "art of a {}.",
    "a sketch of the {}.",
    "a embroidered {}.",
    "a pixelated photo of a {}.",
    "itap of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a good photo of a {}.",
    "a plushie {}.",
    "a photo of the nice {}.",
    "a photo of the small {}.",
    "a photo of the weird {}.",
    "the cartoon {}.",
    "art of the {}.",
    "a drawing of the {}.",
    "a photo of the large {}.",
    "a black and white photo of a {}.",
    "the plushie {}.",
    "a dark photo of a {}.",
    "itap of a {}.",
    "graffiti of the {}.",
    "a toy {}.",
    "itap of my {}.",
    "a photo of a cool {}.",
    "a photo of a small {}.",
    "a tattoo of the {}.",
]

# ── LVIS-specific object-presence templates ───────────────────────────────────
LVIS_TEMPLATES: List[str] = [
    "a photo containing a {}.",
    "there is a {} in the scene.",
    "a photo that contains a {}.",
    "an image with a {} in it.",
    "a {}.",
]

# ── Question templates (VQA-style) ────────────────────────────────────────────
QUESTION_TEMPLATES: List[str] = [
    "is there a {} in the image?",
    "what is this? a {}.",
    "does this image contain a {}?",
]

# ── Description templates (use the WordNet-style `def` gloss) ──────────────────
# Filled with (label, definition).
DESCRIPTION_TEMPLATES: List[str] = [
    "a photo of a {label}, which is {definition}.",
    "{label}: {definition}.",
]


def normalize_label(raw: str) -> str:
    """LVIS names/synonyms use underscores and (sense) suffixes, e.g.
    'baseball_bat', 'bass_(fish)'. Turn them into natural phrases."""
    s = raw.replace("_", " ").strip()
    # Drop a trailing WordNet sense qualifier like '(fish)' or '(computer_equipment)'
    if "(" in s and s.endswith(")"):
        s = s[: s.index("(")].strip()
    return s


def labels_for_category(cat: dict) -> List[str]:
    """Distinct, order-preserving label surface forms for a category:
    the primary name plus every synonym."""
    seen: set = set()
    out: List[str] = []
    for raw in [cat.get("name", "")] + list(cat.get("synonyms", []) or []):
        lbl = normalize_label(raw)
        if lbl and lbl.lower() not in seen:
            seen.add(lbl.lower())
            out.append(lbl)
    return out


def build_tag_strings(cat: dict) -> Dict[str, List[str]]:
    """Return the grouped text strings for one LVIS category:

        {"prompts": [...], "descriptions": [...], "questions": [...]}

    - prompts     : (OPENAI_80 + LVIS_TEMPLATES) x each label surface form
    - descriptions: DESCRIPTION_TEMPLATES filled with (primary label, def gloss)
    - questions   : QUESTION_TEMPLATES x each label surface form

    LLM-generated text can later be added as an extra key (e.g. "llm").
    """
    labels = labels_for_category(cat)
    primary = labels[0] if labels else normalize_label(cat.get("name", ""))
    definition = (cat.get("def") or "").strip().rstrip(".")

    prompts: List[str] = []
    for lbl in labels:
        for tpl in OPENAI_80:
            prompts.append(tpl.format(lbl))
        for tpl in LVIS_TEMPLATES:
            prompts.append(tpl.format(lbl))

    descriptions: List[str] = []
    if definition:
        for tpl in DESCRIPTION_TEMPLATES:
            descriptions.append(tpl.format(label=primary, definition=definition))

    questions: List[str] = []
    for lbl in labels:
        for tpl in QUESTION_TEMPLATES:
            questions.append(tpl.format(lbl))

    return {"prompts": prompts, "descriptions": descriptions, "questions": questions}
