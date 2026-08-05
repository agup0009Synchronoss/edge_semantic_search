"""
ingest_descriptions.py

Validate the ChatGPT-generated tag descriptions before they become embeddings.

Drop whatever ChatGPT returns into data/llm_desc/ (any number of .json files,
any chunking) and run this. It checks the union of all files against the
canonical 4585-tag list and tells you exactly what is missing or malformed, so
you only have to redo the broken chunks rather than the whole set.

Accepts either shape, since ChatGPT is inconsistent about wrapping:
    [{"tag": "...", "descriptions": [...]}, ...]
    {"results": [{"tag": "...", "descriptions": [...]}, ...]}

Tag matching is exact against ram_tag_list.txt, with a case-insensitive
fallback that is reported separately — RAM tags are case-sensitive
("3D CG rendering") and a silent case fix would hide a truncated response.

Usage:
    python ingest_descriptions.py                 # validate + report
    python ingest_descriptions.py --missing-out missing.json
    python ingest_descriptions.py --expect 10     # require exactly N per tag
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import config


def load_files() -> tuple[dict[str, list[str]], list[str], dict[str, str]]:
    """Returns (by_exact_tag, problems, casefold_alias -> canonical)."""
    canonical = config.load_ram_tags()
    canon_set = set(canonical)
    ci_map = {t.casefold(): t for t in canonical}

    by_tag: dict[str, list[str]] = defaultdict(list)
    problems: list[str] = []
    case_fixed: dict[str, str] = {}

    files = sorted(config.LLM_DESC_DIR.glob("*.json"))
    if not files:
        problems.append(f"no .json files found in {config.LLM_DESC_DIR}")
        return by_tag, problems, case_fixed

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError as e:
            problems.append(f"{path.name}: invalid JSON at line {e.lineno} — {e.msg}")
            continue

        if isinstance(payload, dict):
            records = payload.get("results") or payload.get("tags") or []
        elif isinstance(payload, list):
            records = payload
        else:
            problems.append(f"{path.name}: top level is {type(payload).__name__}, "
                            f"expected list or object")
            continue

        n_ok = 0
        for i, rec in enumerate(records):
            if not isinstance(rec, dict):
                problems.append(f"{path.name}[{i}]: entry is {type(rec).__name__}")
                continue
            tag = rec.get("tag")
            descs = rec.get("descriptions")
            if not isinstance(tag, str) or not tag.strip():
                problems.append(f"{path.name}[{i}]: missing/empty 'tag'")
                continue
            if not isinstance(descs, list):
                problems.append(f"{path.name}[{i}] '{tag}': 'descriptions' is not a list")
                continue

            key = tag
            if key not in canon_set:
                alt = ci_map.get(tag.casefold())
                if alt is None:
                    problems.append(f"{path.name}[{i}]: '{tag}' is not a RAM tag")
                    continue
                case_fixed[tag] = alt
                key = alt

            clean = [d.strip() for d in descs if isinstance(d, str) and d.strip()]
            if clean:
                by_tag[key].extend(clean)
                n_ok += 1
        print(f"  {path.name}: {n_ok}/{len(records)} usable records")

    return by_tag, problems, case_fixed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expect", type=int, default=10,
                    help="expected descriptions per tag (default 10)")
    ap.add_argument("--missing-out", metavar="PATH",
                    help="write the still-missing tags as a ChatGPT-ready chunk file")
    args = ap.parse_args()

    config.ensure_dirs()
    canonical = config.load_ram_tags()

    print(f"Scanning {config.LLM_DESC_DIR}")
    by_tag, problems, case_fixed = load_files()

    covered = [t for t in canonical if by_tag.get(t)]
    missing = [t for t in canonical if not by_tag.get(t)]
    short = {t: len(by_tag[t]) for t in covered if len(by_tag[t]) < args.expect}
    over = {t: len(by_tag[t]) for t in covered if len(by_tag[t]) > args.expect}

    print(f"\n{'='*60}")
    print(f"  tags covered      {len(covered)} / {len(canonical)} "
          f"({100*len(covered)/len(canonical):.1f}%)")
    print(f"  tags missing      {len(missing)}")
    print(f"  fewer than {args.expect}    {len(short)}")
    print(f"  more than {args.expect}     {len(over)}")
    total = sum(len(v) for v in by_tag.values())
    print(f"  descriptions      {total} "
          f"({total/max(len(covered),1):.1f} per covered tag)")

    if case_fixed:
        print(f"\n  {len(case_fixed)} tags matched only after case-folding "
              f"(ChatGPT altered capitalization):")
        for bad, good in list(case_fixed.items())[:5]:
            print(f"    '{bad}' -> '{good}'")

    if short:
        print(f"\n  short tags (first 10): "
              f"{list(short.items())[:10]}")

    if problems:
        print(f"\n  {len(problems)} problems:")
        for p in problems[:15]:
            print(f"    ! {p}")
        if len(problems) > 15:
            print(f"    ... and {len(problems)-15} more")

    if missing:
        print(f"\n  missing (first 10): {missing[:10]}")

    if args.missing_out and (missing or short):
        redo = missing + sorted(short)
        out = {"chunk_id": 0, "total_chunks": 1, "tags": redo}
        p = config.LLM_DESC_DIR.parent / args.missing_out
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf8")
        print(f"\n  wrote {p} with {len(redo)} tags to re-request")

    print(f"{'='*60}")
    if not missing and not short and not problems:
        print("All tags covered. Build the classifiers with:")
        print("  python 01_build_text_classifiers.py --source llm")
    else:
        print("Incomplete. Re-request the gaps, or build partial with:")
        print("  python 01_build_text_classifiers.py --source llm --allow-partial")


if __name__ == "__main__":
    main()
