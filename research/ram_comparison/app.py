"""
app.py — RAM++ vs TinyCLIP qualitative tag comparison.

Upload an image, set the threshold knob for uncalibrated TinyCLIP tags, hit Run.
Both models score the SAME 4585-tag vocabulary, so disagreements are about the
models rather than about vocabulary coverage.

What is comparable and what is not:
  - RAM++ emits a sigmoid probability (~0.65-0.95 cutoffs)
  - TinyCLIP emits a cosine similarity (~0.20-0.40 cutoffs)
These scales are NOT comparable. The margin over each tag's own threshold is,
which is why every panel ranks by margin and shows it explicitly.

Run:
    ./venv_ramclip/Scripts/python.exe app.py
"""

from __future__ import annotations

import csv
import os

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
_tiny: TinyClipTagger | None = None
_ram: RamTagger | None = None


def get_tiny() -> TinyClipTagger:
    global _tiny
    if _tiny is None:
        _tiny = TinyClipTagger()
    return _tiny


def get_ram() -> RamTagger:
    global _ram
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


def _fmt_both(tags, ram_by, tiny_by) -> str:
    """Agreement panel: show BOTH models' numbers for each shared tag."""
    if not tags:
        return "_no tags found by both_"
    lines = []
    for t in tags:
        r, c = ram_by[t], tiny_by[t]
        lines.append(
            f"- **{t}**  \n"
            f"  RAM p {r.score:.3f} (thr {r.threshold:.3f}, **{r.margin:+.3f}**) · "
            f"TinyCLIP cos {c.score:.3f} (thr {c.threshold:.3f}, **{c.margin:+.3f}**) `{c.source}`"
        )
    return "\n".join(lines)


def _split(ram_hits, tiny_hits):
    """Disjoint agreement sets, keyed on tag string (same vocabulary both sides)."""
    ram_by = {h.tag: h for h in ram_hits}
    tiny_by = {h.tag: h for h in tiny_hits}
    # Order by each model's own score (see tinyclip_tagger.tag for why margin is
    # a bad sort key). 'Both' uses TinyCLIP's score only as a tiebreak-free
    # ordering — the two scales cannot be summed meaningfully.
    both = sorted(set(ram_by) & set(tiny_by), key=lambda t: -ram_by[t].score)
    ram_only = sorted(set(ram_by) - set(tiny_by), key=lambda t: -ram_by[t].score)
    tiny_only = sorted(set(tiny_by) - set(ram_by), key=lambda t: -tiny_by[t].score)
    return both, ram_only, tiny_only, ram_by, tiny_by


def run(image, knob, classifier_set, ram_size, ram_offset, top_k):
    if image is None:
        return ("Upload an image first.", "", "", "", "", "", [], None)

    img = image if isinstance(image, Image.Image) else Image.fromarray(image)

    tiny = get_tiny()
    tiny_hits, tiny_info = tiny.tag(img, knob=knob,
                                    classifier_set=classifier_set,
                                    top_k=int(top_k) or None)
    ram = get_ram()
    ram_hits, ram_info = ram.tag(img, image_size=int(ram_size),
                                 top_k=int(top_k) or None,
                                 threshold_offset=ram_offset)

    both, ram_only, tiny_only, ram_by, tiny_by = _split(ram_hits, tiny_hits)

    status = (
        f"**RAM++** {ram_info['n_above']} tags above threshold "
        f"(showing {ram_info['n_returned']}) · {ram_info['image_size']}px · "
        f"{ram_info['infer_ms']:.0f} ms · {ram_info['device']}"
        + (f" · offset {ram_offset:+.2f}" if ram_offset else "")
        + "\n\n"
        f"**TinyCLIP** {tiny_info['n_above']} tags above threshold "
        f"(showing {tiny_info['n_returned']}, {tiny_info['n_calibrated_hits']} calibrated) · "
        f"`{tiny_info['classifier_set']}` · knob {knob:.2f} · "
        f"{tiny_info['encode_ms']:.0f} ms encode + {tiny_info['score_ms']:.1f} ms score"
        "\n\n"
        f"**Agreement** {len(both)} both · {len(ram_only)} RAM-only · "
        f"{len(tiny_only)} TinyCLIP-only"
    )
    if ram_info["image_size"] != 384:
        status += ("\n\n> ⚠️ RAM++ is running at "
                   f"{ram_info['image_size']}px but the checkpoint and its per-tag "
                   "thresholds were tuned at 384px — its scores are shifted here.")

    # Merged table for export
    rows = []
    for tag in sorted(set(ram_by) | set(tiny_by)):
        r, t = ram_by.get(tag), tiny_by.get(tag)
        rows.append([
            tag,
            f"{r.score:.4f}" if r else "",
            f"{r.threshold:.3f}" if r else "",
            f"{r.margin:+.4f}" if r else "",
            f"{t.score:.4f}" if t else "",
            f"{t.threshold:.3f}" if t else "",
            f"{t.margin:+.4f}" if t else "",
            (t.source if t else ""),
            "both" if (r and t) else ("RAM only" if r else "TinyCLIP only"),
        ])

    csv_path = config.CACHE_DIR / "comparison.csv"
    config.ensure_dirs()
    with csv_path.open("w", encoding="utf8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "ram_score", "ram_threshold", "ram_margin",
                    "tinyclip_cosine", "tinyclip_threshold", "tinyclip_margin",
                    "tinyclip_thr_source", "verdict"])
        w.writerows(rows)

    return (
        status,
        _fmt_hits(ram_hits, "p"),
        _fmt_hits(tiny_hits, "cos"),
        _fmt_both(both, ram_by, tiny_by),
        _fmt_hits([ram_by[t] for t in ram_only], "p"),
        _fmt_hits([tiny_by[t] for t in tiny_only], "cos"),
        rows,
        str(csv_path),
    )


def build_ui() -> gr.Blocks:
    tiny_sets = []
    try:
        tiny_sets = get_tiny().available_sets()
    except Exception:  # noqa: BLE001 — UI should still render if artifacts are missing
        pass
    default_set = "templates" if "templates" in tiny_sets else (tiny_sets[0] if tiny_sets else "templates")

    with gr.Blocks(title="RAM++ vs TinyCLIP") as demo:
        gr.Markdown(
            "# RAM++ vs TinyCLIP — tag comparison\n"
            "Both models score the **same 4585-tag vocabulary**. RAM++ uses its own "
            "per-tag thresholds; TinyCLIP uses LVIS-calibrated thresholds where a tag "
            "maps (`cal`, 1037 tags) and the global knob everywhere else (`knob`).\n\n"
            "Each list is ranked by that model's own score. The **bold margin** is how far "
            "a tag sits above its own threshold — that part *is* comparable across models, "
            "the raw scores are not (sigmoid vs cosine)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(label="Image", type="pil", height=320)
                go = gr.Button("Run comparison", variant="primary")
            with gr.Column(scale=1):
                knob = gr.Slider(
                    *config.UNCALIBRATED_THRESHOLD_RANGE, step=0.01,
                    value=config.DEFAULT_UNCALIBRATED_THRESHOLD,
                    label="TinyCLIP threshold for UNCALIBRATED tags (the knob)",
                    info="Applies to the ~3548 tags with no LVIS calibration. "
                         "Calibrated tags keep their own cutoff.",
                )
                classifier_set = gr.Radio(
                    choices=["templates", "llm"], value=default_set,
                    label="TinyCLIP text source",
                    info="templates = 88 prompt strings/tag · llm = ChatGPT visual descriptions",
                )
                with gr.Row():
                    ram_size = gr.Dropdown(
                        choices=[str(s) for s in config.RAM_IMAGE_SIZES],
                        value=str(config.RAM_IMAGE_SIZE_DEFAULT),
                        label="RAM++ input size",
                        info="384 = as trained; 224 = ~3x faster, thresholds shift",
                    )
                    top_k = gr.Number(value=config.DEFAULT_TOP_K, precision=0,
                                      label="Top-K per model (0 = all)")
                ram_offset = gr.Slider(-0.30, 0.30, step=0.01, value=0.0,
                                       label="RAM++ threshold offset",
                                       info="Shifts all RAM cutoffs, preserving their relative calibration")

        status = gr.Markdown()

        with gr.Row():
            with gr.Column():
                gr.Markdown("#### RAM++")
                ram_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### TinyCLIP")
                tiny_out = gr.Markdown()

        gr.Markdown("### Agreement")
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Both")
                both_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### RAM++ only")
                ram_only_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("#### TinyCLIP only")
                tiny_only_out = gr.Markdown()

        gr.Markdown("### Merged table")
        table = gr.Dataframe(
            headers=["tag", "RAM p", "RAM thr", "RAM margin",
                     "TinyCLIP cos", "TC thr", "TC margin", "TC thr src", "verdict"],
            wrap=True,
        )
        csv_file = gr.File(label="Download comparison.csv")

        outputs = [status, ram_out, tiny_out, both_out, ram_only_out,
                   tiny_only_out, table, csv_file]
        inputs = [image, knob, classifier_set, ram_size, ram_offset, top_k]
        go.click(run, inputs=inputs, outputs=outputs)

        # Headers for the side-by-side lanes (gr.Markdown has no visible label)
        ram_out.label = "RAM++"
        tiny_out.label = "TinyCLIP"

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name=os.environ.get("GRADIO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_PORT", config.GRADIO_PORT_DEFAULT)),
        share=os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
        root_path=os.environ.get("GRADIO_ROOT_PATH") or None,
        # Surface real tracebacks in the UI; without this a model-load failure
        # shows only "the upstream Gradio app has raised an exception".
        show_error=True,
    )
