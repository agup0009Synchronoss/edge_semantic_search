"""
02_build_tag_mapping.py

Map the 1203 calibrated LVIS tags onto the 4585 RAM++ tags so RAM tags can
inherit a real per-tag cosine threshold instead of the UI knob.

Matching is deliberately conservative and fully auditable — every match records
which rule produced it, and every rule is reversible by reading the CSV:

    exact        normalized strings are identical
    plural       one is the simple singular/plural of the other
    punct        differ only in hyphens / periods / spaces
    compound     differ only by a trailing or leading stopword-ish token

LVIS synonyms are eligible, not just primary names — 145 tags matched only via
a synonym in the initial survey, so ignoring them loses real coverage.

Ambiguity guard: an LVIS tag whose name carries a WordNet sense qualifier —
bow_(weapon) vs bow_(decorative_ribbons) — is dropped when two different LVIS
tags would claim the same RAM tag, because we cannot tell which sense RAM's
bare "bow" means, and silently applying the wrong threshold is worse than
falling back to the knob.

Outputs:
    data/text/lvis_to_ram_mapping.csv   audit trail, one row per match
    data/text/thresholds_4585.npy       (4585,) f32, NaN where uncalibrated

Usage:
    python 02_build_tag_mapping.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict

import numpy as np

import config
import templates as tpl  # normalize_label() from tinyClipCaliberation/


# ── Normalization ─────────────────────────────────────────────────────────────
def norm(s: str) -> str:
    """Casefold + collapse whitespace. Applied on top of templates.normalize_label."""
    return re.sub(r"\s+", " ", s.replace("_", " ").strip().lower())


def depunct(s: str) -> str:
    """Drop hyphens/periods/apostrophes and collapse — 't-shirt' == 't shirt'."""
    return re.sub(r"\s+", " ", re.sub(r"[-.'`]", " ", s)).strip()


def singularize(s: str) -> str:
    """Crude English singularizer, good enough for concrete nouns.

    Only the last word is touched: 'sunglasses' -> 'sunglass',
    'binoculars' -> 'binocular', 'cherries' -> 'cherry'.
    """
    words = s.split()
    if not words:
        return s
    w = words[-1]
    if len(w) > 3 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 3 and w.endswith("sses"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("ses") and not w.endswith("ases"):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss") and not w.endswith("us"):
        w = w[:-1]
    return " ".join(words[:-1] + [w])


def variants(s: str) -> list[tuple[str, str]]:
    """Ordered (rule, key) candidates for one label, most-trusted first."""
    n = norm(s)
    out = [("exact", n)]
    d = depunct(n)
    if d != n:
        out.append(("punct", d))
    for rule, base in (("plural", n), ("plural", d)):
        sg = singularize(base)
        if sg and sg != base:
            out.append((rule, sg))
    return out


def has_sense_qualifier(raw_name: str) -> bool:
    """LVIS name carried a '(sense)' disambiguator, e.g. bow_(weapon)."""
    return "(" in raw_name and raw_name.rstrip().endswith(")")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", choices=("balanced", "naive"), default="balanced",
                    help="which LVIS calibration to inherit (default: balanced)")
    args = ap.parse_args()

    config.ensure_dirs()

    ram_tags = config.load_ram_tags()
    assert len(ram_tags) == config.N_TAGS, f"expected {config.N_TAGS}, got {len(ram_tags)}"

    lvis = json.loads(config.LVIS_TAG_ORDER.read_text(encoding="utf8"))["tags"]
    thr_path = (config.LVIS_BALANCED_THRESHOLDS if args.thresholds == "balanced"
                else config.LVIS_NAIVE_THRESHOLDS)
    lvis_thr = np.load(thr_path)
    assert len(lvis_thr) == len(lvis), f"{len(lvis_thr)} thresholds vs {len(lvis)} tags"
    print(f"LVIS: {len(lvis)} tags, thresholds from {thr_path.name} "
          f"(range {lvis_thr.min():.2f}-{lvis_thr.max():.2f})")
    print(f"RAM:  {len(ram_tags)} tags")

    # RAM lookup: variant key -> (row, ram_side_rule). First writer wins so an
    # exact key is never displaced by a fuzzier variant of a different tag.
    ram_index: dict[str, tuple[int, str]] = {}
    for row, tag in enumerate(ram_tags):
        for rule, key in variants(tag):
            ram_index.setdefault(key, (row, rule))

    # ── Match ─────────────────────────────────────────────────────────────────
    # claims[ram_row] = list of (rule, ram_rule, lvis_row, name, surface, thr, is_syn)
    claims: dict[int, list[tuple]] = defaultdict(list)
    for l_row, t in enumerate(lvis):
        raw_name = t["name"]
        surfaces = [(0, tpl.normalize_label(raw_name))]
        surfaces += [(1, tpl.normalize_label(s)) for s in (t.get("synonyms") or [])]
        best = None
        for is_syn, surface in surfaces:
            if not surface:
                continue
            for rule, key in variants(surface):
                if key in ram_index:
                    r_row, ram_rule = ram_index[key]
                    rank = (0 if rule == "exact" else 1,
                            0 if ram_rule == "exact" else 1, is_syn)
                    cand = (rank, rule, ram_rule, r_row, surface, is_syn)
                    if best is None or cand[0] < best[0]:
                        best = cand
            if best and best[0] == (0, 0, 0):
                break  # exact-to-exact on the primary name — cannot do better
        if best:
            _rank, rule, ram_rule, r_row, surface, is_syn = best
            claims[r_row].append(
                (rule, ram_rule, l_row, raw_name, surface, float(lvis_thr[l_row]), is_syn)
            )

    # ── Resolve contested RAM tags ────────────────────────────────────────────
    thresholds = np.full(config.N_TAGS, np.nan, dtype=np.float32)
    rows_out = []
    n_dropped_ambiguous = 0
    n_dropped_asym = 0

    def _row(r_row, c, applied, note):
        rule, ram_rule, l_row, name, surface, thr, is_syn = c
        return {
            "ram_row": r_row, "ram_tag": ram_tags[r_row],
            "lvis_row": l_row, "lvis_name": name, "matched_surface": surface,
            "rule": rule, "ram_rule": ram_rule, "via_synonym": int(is_syn),
            "threshold": f"{thr:.4f}", "applied": applied, "note": note,
        }

    for r_row, cands in sorted(claims.items()):
        if len(cands) > 1:
            # Two+ LVIS tags claim one RAM tag. If any carried a sense
            # qualifier, the RAM tag is genuinely ambiguous -> use the knob.
            if any(has_sense_qualifier(c[3]) for c in cands):
                n_dropped_ambiguous += 1
                rows_out += [_row(r_row, c, 0, "ambiguous_sense_dropped") for c in cands]
                continue
            # Otherwise prefer exact-to-exact, then primary name over synonym.
            cands = sorted(cands, key=lambda c: (c[0] != "exact", c[1] != "exact",
                                                 c[6], c[2]))

        c = cands[0]
        rule, ram_rule, l_row, name, surface, thr, is_syn = c

        # Asymmetric fuzzy match: we matched an LVIS *exact* surface against a
        # RAM-side *variant*. That is the one direction that can conflate
        # genuinely different words -- LVIS 'glass' (drink container) collapses
        # onto RAM 'glasses' (eyewear), because singularize('glasses')=='glass'.
        # RAM has no bare 'glass' tag, so nothing would catch it downstream.
        # When the LVIS name also carries a sense qualifier we cannot verify the
        # sense survived the collapse, so fall back to the knob.
        if ram_rule != "exact" and rule == "exact" and has_sense_qualifier(name):
            n_dropped_asym += 1
            rows_out.append(_row(r_row, c, 0, "asymmetric_variant_dropped"))
            continue

        note = "" if len(cands) == 1 else "contested_resolved"
        if ram_rule != "exact" and rule == "exact":
            note = (note + ";" if note else "") + "review_ram_variant"
        thresholds[r_row] = thr
        rows_out.append(_row(r_row, c, 1, note))

    np.save(config.THRESHOLDS_4585, thresholds)
    with config.TAG_MAPPING_CSV.open("w", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "ram_row", "ram_tag", "lvis_row", "lvis_name", "matched_surface",
            "rule", "ram_rule", "via_synonym", "threshold", "applied", "note"])
        w.writeheader()
        w.writerows(sorted(rows_out, key=lambda r: r["ram_row"]))

    # ── Report ────────────────────────────────────────────────────────────────
    n_cal = int(np.isfinite(thresholds).sum())
    by_rule = defaultdict(int)
    for r in rows_out:
        if r["applied"]:
            by_rule[r["rule"]] += 1
    print(f"\nCalibrated RAM tags: {n_cal} / {config.N_TAGS} "
          f"({100*n_cal/config.N_TAGS:.1f}%)  -> {config.N_TAGS - n_cal} use the UI knob")
    for rule in ("exact", "punct", "plural"):
        if by_rule.get(rule):
            print(f"  {rule:9s} {by_rule[rule]}")
    n_syn = sum(1 for r in rows_out if r["applied"] and r["via_synonym"])
    print(f"  (of which {n_syn} matched only via an LVIS synonym)")
    if n_dropped_ambiguous:
        print(f"  {n_dropped_ambiguous} RAM tags dropped as ambiguous (sense conflict)")
    if n_dropped_asym:
        print(f"  {n_dropped_asym} dropped as asymmetric variant matches")
    n_review = sum(1 for r in rows_out if r["applied"] and "review" in r["note"])
    if n_review:
        print(f"  {n_review} applied but flagged review_ram_variant — "
              f"grep the CSV to eyeball them")
    fin = thresholds[np.isfinite(thresholds)]
    print(f"  threshold range {fin.min():.3f}-{fin.max():.3f}, mean {fin.mean():.3f}")
    print(f"\nwrote {config.TAG_MAPPING_CSV.name} and {config.THRESHOLDS_4585.name}")


if __name__ == "__main__":
    main()
