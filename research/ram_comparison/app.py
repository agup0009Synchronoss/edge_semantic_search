"""
app.py — three-way qualitative tag comparison.

Lanes: RAM++ · TinyCLIP/templates · TinyCLIP/llm. All three score the SAME
4585-tag vocabulary, so disagreement is never about vocabulary coverage.

The two TinyCLIP lanes are the A/B that motivated this app: identical encoder,
identical thresholds, identical image — the ONLY difference is the text each
tag's classifier embedding was built from (88 templated prompts vs 10 generated
visual descriptions). Both are scored from a single shared image encode, so the
third lane costs a matmul rather than another 200 ms of vision encoder.

What is comparable and what is not:
  - RAM++ emits a sigmoid probability (~0.65-0.95 cutoffs)
  - TinyCLIP emits a cosine similarity (~0.20-0.40 cutoffs)
Those scales are NOT comparable across lanes. The margin over each tag's own
threshold is, which is why every panel prints it in bold. Ranking is still by
raw score — see tinyclip_tagger.tag for why margin is a bad sort key.

Results are split into 7 disjoint Venn regions so every tag any lane fired on
appears exactly once, with the templates-vs-llm head-to-head broken out
separately.

Run:
    ./venv_ramclip/bin/python app.py          # Linux
    ./venv_ramclip/Scripts/python.exe app.py  # Windows
"""

from __future__ import annotations

import csv
import os
import threading

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Must precede any transformers/huggingface_hub import: RAM's init_tokenizer()
# fetches bert-base-uncased from the Hub on every model construction.
# config comes first because it is what puts research/common on sys.path.
import config      # noqa: F401,E402
import ssl_bypass  # noqa: F401,E402

import gradio as gr
from PIL import Image

from ram_tagger import RamTagger
from tinyclip_tagger import TinyClipTagger

# ── Lazy singletons: RAM++ is a 2.8 GB load, don't pay it at import ───────────
# Locked because the naive `if _x is None` is a check-then-act race once more
# than one request can be in flight: two threads both see None and both
# construct, which for RAM++ means loading 2.8 GB of weights twice and very
# likely an OOM.
_tiny: TinyClipTagger | None = None
_ram: RamTagger | None = None
_tiny_lock = threading.Lock()
_ram_lock = threading.Lock()


def get_tiny() -> TinyClipTagger:
    global _tiny
    if _tiny is None:
        with _tiny_lock:
            if _tiny is None:
                _tiny = TinyClipTagger()
    return _tiny


def get_ram() -> RamTagger:
    global _ram
    if _ram is None:
        with _ram_lock:
            if _ram is None:
                _ram = RamTagger()
    return _ram


def _fmt_hits(hits, score_label: str) -> str:
    """Markdown list: tag, raw score, margin over threshold, threshold source."""
    if not hits:
        return "_no tags above threshold_"
    lines = []
    for h in hits:
        src = getattr(h, "source", "")
        badge = "" if src == "ram" else f" `{src}`"
        lines.append(
            f"- **{h.tag}** — {score_label} {h.score:.3f} "
            f"(thr {h.threshold:.3f}, **{h.margin:+.3f}**){badge}"
        )
    return "\n".join(lines)


def _fmt_multi(tags, lanes, limit: int | None = None) -> str:
    """Agreement panel across N lanes.

    lanes: [(short_label, by_tag dict, score_label), ...]. A tag only prints the
    lanes it actually appears in, so a panel for "templates + RAM, not llm" does
    not render an empty llm row.

    `limit` caps how many are rendered but the caller has already computed the
    region over the FULL above-threshold sets — the trailing count makes the
    difference between "few agreements" and "many, truncated" unambiguous.
    """
    if not tags:
        return "_none_"
    total = len(tags)
    if limit and total > limit:
        tags = tags[:limit]
    out = []
    for t in tags:
        parts = []
        for label, by, score_label in lanes:
            h = by.get(t)
            if h is None:
                continue
            src = getattr(h, "source", "")
            badge = "" if src == "ram" else f" `{src}`"
            parts.append(f"{label} {score_label} {h.score:.3f} "
                         f"(thr {h.threshold:.3f}, **{h.margin:+.3f}**){badge}")
        out.append(f"- **{t}**  \n  " + " · ".join(parts))
    if limit and total > limit:
        out.append(f"\n_… and {total - limit} more (showing top {limit} of {total})_")
    return "\n".join(out)


def _venn3(ram_hits, tpl_hits, llm_hits):
    """Split three tag sets into the 7 disjoint Venn regions.

    Every region is ordered by the score of a lane that is actually present in
    it — sorting by a lane a tag is absent from would KeyError, and the three
    scales (sigmoid vs two cosines) cannot be combined into one ranking anyway.
    """
    ram_by = {h.tag: h for h in ram_hits}
    tpl_by = {h.tag: h for h in tpl_hits}
    llm_by = {h.tag: h for h in llm_hits}
    R, T, L = set(ram_by), set(tpl_by), set(llm_by)

    def by_score(tags, table):
        return sorted(tags, key=lambda t: -table[t].score)

    regions = {
        # consensus
        "all3":      by_score(R & T & L, ram_by),
        # the head-to-head: RAM++ corroborates exactly one text source
        "tpl_ram":   by_score((T & R) - L, ram_by),
        "llm_ram":   by_score((L & R) - T, ram_by),
        # TinyCLIP agrees with itself, RAM++ dissents
        "tc_both":   by_score((T & L) - R, tpl_by),
        # solo calls
        "ram_only":  by_score(R - T - L, ram_by),
        "tpl_only":  by_score(T - R - L, tpl_by),
        "llm_only":  by_score(L - R - T, llm_by),
    }
    return regions, ram_by, tpl_by, llm_by


# status + 3 lane lists + 7 Venn regions + table + csv. Asserted against the
# outputs list in build_ui() so the two cannot drift apart silently — a
# mismatch here surfaces in Gradio as components updating with the wrong value.
N_OUTPUTS = 13


def run(image, knob, override_all, ram_size, ram_offset, top_k):
    if image is None:
        return ("Upload an image first.",) + ("",) * 10 + ([], None)

    img = image if isinstance(image, Image.Image) else Image.fromarray(image)
    k = int(top_k) or None

    tiny = get_tiny()
    available = tiny.available_sets()
    wanted = [s for s in ("templates", "llm") if s in available]

    # Deliberately fetch the FULL above-threshold sets (top_k=None) and apply
    # top-K only for display. Computing agreement on already-truncated lists
    # compares different depths of each lane's distribution — templates can fire
    # 500 tags where llm fires 79, so a top-10-vs-top-10 intersection reports
    # near-zero overlap that is an artifact of the cut, not a real disagreement.
    # One encode, both classifier matrices. See TinyClipTagger.tag_sets.
    tiny_res = tiny.tag_sets(img, knob=knob, sets=wanted, top_k=None,
                             override_all=override_all)
    tpl_all, tpl_info = tiny_res.get("templates", ([], None))
    llm_all, llm_info = tiny_res.get("llm", ([], None))

    ram = get_ram()
    ram_all, ram_info = ram.tag(img, image_size=int(ram_size), top_k=None,
                                threshold_offset=ram_offset)

    regions, ram_by, tpl_by, llm_by = _venn3(ram_all, tpl_all, llm_all)

    # Display slices — the lanes show each model's headline predictions.
    ram_hits = ram_all[:k] if k else ram_all
    tpl_hits = tpl_all[:k] if k else tpl_all
    llm_hits = llm_all[:k] if k else llm_all

    def tc_line(label, info, shown):
        if info is None:
            return (f"**TinyCLIP · {label}** — classifier set not built "
                    f"(`python 01_build_text_classifiers.py --source {label}`)")
        return (
            f"**TinyCLIP · {label}** {info['n_above']} tags above threshold "
            f"(showing {shown}"
            + (", knob on ALL tags" if info["override_all"]
               else f", {info['n_calibrated_hits']} calibrated")
            + f") · {info['score_ms']:.1f} ms score"
        )

    encode_ms = (tpl_info or llm_info or {}).get("encode_ms", 0.0)
    status = (
        f"**RAM++** {ram_info['n_above']} tags above threshold "
        f"(showing {len(ram_hits)}) · {ram_info['image_size']}px · "
        f"{ram_info['infer_ms']:.0f} ms · {ram_info['device']}"
        + (f" · offset {ram_offset:+.2f}" if ram_offset else "")
        + "\n\n" + tc_line("templates", tpl_info, len(tpl_hits))
        + "\n\n" + tc_line("llm", llm_info, len(llm_hits))
        + f"\n\n_knob {knob:.3f}"
        + (" applied to ALL tags_" if override_all else " on uncalibrated tags only_")
        + f" · one shared image encode: {encode_ms:.0f} ms"
        "\n\n"
        "**Agreement** — over every tag above threshold, not just the shown top-K:  \n"
        f"{len(regions['all3'])} all three · "
        f"{len(regions['tpl_ram'])} templates+RAM · {len(regions['llm_ram'])} llm+RAM · "
        f"{len(regions['tc_both'])} TinyCLIP-consensus · "
        f"{len(regions['ram_only'])} RAM-only · {len(regions['tpl_only'])} templates-only · "
        f"{len(regions['llm_only'])} llm-only"
    )
    if ram_info["image_size"] != 384:
        status += ("\n\n> ⚠️ RAM++ is running at "
                   f"{ram_info['image_size']}px but the checkpoint and its per-tag "
                   "thresholds were tuned at 384px — its scores are shifted here.")

    # Merged table / CSV export — one row per tag, all three lanes side by side.
    rows = []
    for tag in sorted(set(ram_by) | set(tpl_by) | set(llm_by)):
        r, t, l = ram_by.get(tag), tpl_by.get(tag), llm_by.get(tag)
        present = "".join(c for c, h in (("R", r), ("T", t), ("L", l)) if h)
        rows.append([
            tag,
            f"{r.score:.4f}" if r else "", f"{r.margin:+.4f}" if r else "",
            f"{t.score:.4f}" if t else "", f"{t.margin:+.4f}" if t else "",
            f"{l.score:.4f}" if l else "", f"{l.margin:+.4f}" if l else "",
            (t or l).source if (t or l) else "",
            present,
        ])

    csv_path = config.CACHE_DIR / "comparison.csv"
    config.ensure_dirs()
    with csv_path.open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "ram_score", "ram_margin",
                    "templates_cosine", "templates_margin",
                    "llm_cosine", "llm_margin",
                    "tinyclip_thr_source", "found_by"])
        w.writerows(rows)

    RAM = ("RAM", ram_by, "p")
    TPL = ("tpl", tpl_by, "cos")
    LLM = ("llm", llm_by, "cos")

    # Regions are computed over everything above threshold; cap the rendered
    # list so a lane firing 500 tags does not produce an unreadable wall.
    lim = k or 50

    return (
        status,
        _fmt_hits(ram_hits, "p"),
        _fmt_hits(tpl_hits, "cos"),
        _fmt_hits(llm_hits, "cos"),
        _fmt_multi(regions["all3"], [RAM, TPL, LLM], lim),
        _fmt_multi(regions["tc_both"], [TPL, LLM], lim),
        _fmt_multi(regions["ram_only"], [RAM], lim),
        _fmt_multi(regions["tpl_ram"], [RAM, TPL], lim),
        _fmt_multi(regions["llm_ram"], [RAM, LLM], lim),
        _fmt_multi(regions["tpl_only"], [TPL], lim),
        _fmt_multi(regions["llm_only"], [LLM], lim),
        rows,
        str(csv_path),
    )


def build_ui() -> gr.Blocks:
    missing_note = ""
    try:
        avail = get_tiny().available_sets()
        gaps = [s for s in ("templates", "llm") if s not in avail]
        if gaps:
            missing_note = (
                "\n\n> ⚠️ Classifier set(s) not built: "
                + ", ".join(f"`{g}`" for g in gaps)
                + ". Run `python 01_build_text_classifiers.py --source <set>`. "
                  "That lane will stay empty."
            )
    except Exception:  # noqa: BLE001 — UI should still render if artifacts are missing
        pass

    with gr.Blocks(title="RAM++ vs TinyCLIP (templates vs llm)") as demo:
        gr.Markdown(
            "# Three-way tag comparison\n"
            "**RAM++** · **TinyCLIP/templates** · **TinyCLIP/llm** — all scoring the "
            "*same 4585-tag vocabulary*, so any disagreement is about the models and "
            "the text behind them, never about vocabulary coverage.\n\n"
            "The two TinyCLIP lanes differ **only** in the text used to build each tag's "
            "classifier embedding: 88 templated prompt strings versus 10 generated visual "
            "descriptions. Same encoder, same thresholds, same image — one shared image "
            "encode feeds both, so the third lane is nearly free.\n\n"
            "Each list ranks by that model's own score. The **bold margin** is how far a tag "
            "sits above its own threshold — *that* is comparable across lanes; the raw "
            "scores are not (RAM++ sigmoid vs TinyCLIP cosine)."
            + missing_note
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Image", type="pil", height=320)
                go = gr.Button("Run comparison", variant="primary")
            with gr.Column(scale=1):
                knob = gr.Slider(
                    *config.UNCALIBRATED_THRESHOLD_RANGE, step=0.005,
                    value=config.DEFAULT_UNCALIBRATED_THRESHOLD,
                    label="TinyCLIP threshold for UNCALIBRATED tags (the knob)",
                    info="Applies to the ~3548 tags with no LVIS calibration, in BOTH "
                         "TinyCLIP lanes. Calibrated tags keep their own cutoff — "
                         "unless the toggle below is on.",
                )
                override_all = gr.Checkbox(
                    value=False,
                    label="Apply this threshold to ALL tags (ignore calibration)",
                    info="Overrides the 1037 LVIS-calibrated cutoffs too, so every tag "
                         "uses the knob above. Every hit will then show `knob`, even "
                         "tags that do have a calibrated value on file.",
                )
                with gr.Row():
                    ram_size = gr.Dropdown(
                        choices=[str(s) for s in config.RAM_IMAGE_SIZES],
                        value=str(config.RAM_IMAGE_SIZE_DEFAULT),
                        label="RAM++ input size",
                        info="384 = as trained; 224 = ~3x faster, thresholds shift",
                    )
                    top_k = gr.Number(value=config.DEFAULT_TOP_K, precision=0,
                                      label="Top-K per lane (0 = all)")
                ram_offset = gr.Slider(-0.30, 0.30, step=0.01, value=0.0,
                                       label="RAM++ threshold offset",
                                       info="Shifts all RAM cutoffs, preserving their relative calibration")

        status = gr.Markdown()

        # ── The three lanes ───────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### RAM++")
                ram_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### TinyCLIP · templates\n_88 prompt strings/tag_")
                tpl_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### TinyCLIP · llm\n_10 generated descriptions/tag_")
                llm_out = gr.Markdown()

        # ── Consensus ─────────────────────────────────────────────────────────
        gr.Markdown(
            "## Agreement\n"
            "Seven disjoint regions — every tag any lane fired on appears in exactly one."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### ✅ All three\n_strongest consensus_")
                all3_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### TinyCLIP consensus\n_both text sources, RAM++ dissents_")
                tc_both_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### RAM++ only\n_both TinyCLIP lanes miss it_")
                ram_only_out = gr.Markdown()

        # ── The actual head-to-head ───────────────────────────────────────────
        gr.Markdown(
            "## templates vs llm — head to head\n"
            "The two lanes differ only in text source, so these four regions are the "
            "A/B. RAM++ is not ground truth, but where it corroborates exactly one "
            "side, that side is the more likely correct call."
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### templates + RAM++\n_llm missed it_")
                tpl_ram_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### llm + RAM++\n_templates missed it_")
                llm_ram_out = gr.Markdown()
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### templates only\n_unsupported by either other lane_")
                tpl_only_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### llm only\n_unsupported by either other lane_")
                llm_only_out = gr.Markdown()

        with gr.Accordion("Merged table (every tag, all three lanes)", open=False):
            table = gr.Dataframe(
                headers=["tag", "RAM p", "RAM margin",
                         "tpl cos", "tpl margin",
                         "llm cos", "llm margin",
                         "TC thr src", "found by"],
                wrap=True,
            )
            csv_file = gr.File(label="Download comparison.csv")

        outputs = [status, ram_out, tpl_out, llm_out,
                   all3_out, tc_both_out, ram_only_out,
                   tpl_ram_out, llm_ram_out, tpl_only_out, llm_only_out,
                   table, csv_file]
        assert len(outputs) == N_OUTPUTS, (
            f"outputs has {len(outputs)} components but run() returns {N_OUTPUTS}"
        )
        inputs = [image, knob, override_all, ram_size, ram_offset, top_k]
        go.click(run, inputs=inputs, outputs=outputs)

    return demo


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


if __name__ == "__main__":
    demo = build_ui()

    # ── Warm up before serving ────────────────────────────────────────────────
    # Otherwise the FIRST visitor pays the whole cold start: ~6 s for the 2.8 GB
    # checkpoint plus the classifier load, with no feedback in the UI. On a
    # shared link that reads as a broken app. Set GRADIO_WARMUP=0 to skip.
    if _env_flag("GRADIO_WARMUP", True):
        print("[app] warming up models before serving ...", flush=True)
        try:
            get_tiny()
            get_ram().load()
        except Exception as e:  # missing artifacts shouldn't stop the UI booting
            print(f"[app] warmup failed ({type(e).__name__}: {e})\n"
                  f"      the UI will still start; the error will surface on first use",
                  flush=True)

    # ── Concurrency ───────────────────────────────────────────────────────────
    # Gradio queues by default, but at a concurrency limit of 1 — every extra
    # user waits for the previous inference to finish end to end. A single
    # request is not one homogeneous unit of work though: RAM++ is GPU (or CPU)
    # torch while TinyCLIP is onnxruntime on CPU, so letting a couple run
    # concurrently overlaps those and improves throughput for multiple users.
    #
    # Kept modest by default: the GPU serialises the RAM++ forward pass anyway,
    # and every concurrent request holds its own activations. Raising this too
    # far buys queueing latency and OOM risk rather than throughput. Tune with
    # GRADIO_CONCURRENCY once you know your VRAM.
    concurrency = int(os.environ.get("GRADIO_CONCURRENCY", "3"))
    demo.queue(
        default_concurrency_limit=concurrency,
        # Bound the backlog so a burst gets a clear "queue is full" instead of
        # an unbounded wait that looks like a hang.
        max_size=int(os.environ.get("GRADIO_QUEUE_SIZE", "32")),
    )

    # ── Sharing ───────────────────────────────────────────────────────────────
    # Defaults to True: this app is normally run on a remote box for other
    # people to try, and a *.gradio.live tunnel is the path of least resistance
    # when you cannot open a port.
    #
    # That link is PUBLIC and unauthenticated for its ~72h lifetime — anyone
    # holding it can upload images and consume the GPU. Set GRADIO_SHARE=0 for
    # local-only, or GRADIO_AUTH=user:pass to put a login in front of it.
    share = _env_flag("GRADIO_SHARE", True)

    auth = None
    if os.environ.get("GRADIO_AUTH"):
        user, _, pw = os.environ["GRADIO_AUTH"].partition(":")
        auth = (user, pw)

    if share and not auth:
        print("\n[app] share=True — the *.gradio.live link is PUBLIC and "
              "unauthenticated.\n"
              "      GRADIO_SHARE=0 disables it; GRADIO_AUTH=user:pass adds a "
              "login.\n", flush=True)

    demo.launch(
        server_name=os.environ.get("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_PORT", config.GRADIO_PORT_DEFAULT)),
        share=share,
        auth=auth,
        root_path=os.environ.get("GRADIO_ROOT_PATH") or None,
        # Surface real tracebacks in the UI; without this a model-load failure
        # shows only "the upstream Gradio app has raised an exception".
        show_error=True,
    )
