#!/usr/bin/env python3
"""
Build a short animated GIF that highlights the provenance story (relationship → source quote).

Run from repo root:
  python3 scripts/generate_provenance_demo_gif.py

Requires Pillow (see requirements.txt).
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _try_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("Helvetica", "Arial Unicode MS", "DejaVuSans", "DejaVu Sans"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_panel(title: str, body_lines: list[str], highlight_line: int | None) -> Image.Image:
    w, h = 920, 520
    img = Image.new("RGB", (w, h), color="#f7f7fb")
    draw = ImageDraw.Draw(img)
    font_title = _try_font(26)
    font_body = _try_font(20)
    font_small = _try_font(16)

    draw.rectangle((0, 0, w, 56), fill="#1a1a2e")
    draw.text((24, 14), title, fill="#ffffff", font=font_title)

    y = 72
    for i, line in enumerate(body_lines):
        fill = "#111111"
        if highlight_line is not None and i == highlight_line:
            draw.rounded_rectangle((16, y - 4, w - 16, y + 26), radius=8, fill="#fff3cd", outline="#c9a227", width=2)
            fill = "#1a1a2e"
        draw.text((28, y), line, fill=fill, font=font_body)
        y += 34

    draw.text((24, h - 40), "Qualitative model drafter — verify before you trust", fill="#555555", font=font_small)
    return img


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "static" / "demo-provenance.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []

    frames.append(
        _draw_panel(
            "Step 1 — Extracted relationship",
            [
                "Workload → Stress   (direction: positive)",
                "Mechanism: more deadlines and overload increase felt pressure.",
                "Evidence strength: direct",
            ],
            highlight_line=0,
        )
    )

    frames.append(
        _draw_panel(
            "Step 2 — Click to trace provenance",
            [
                "Workload → Stress   (direction: positive)",
                "▶ Supporting quote opens beside the participant chunk.",
                "Chunk id: respondent_1_chunk_0",
            ],
            highlight_line=1,
        )
    )

    quote = (
        '“I feel overwhelmed when I have too many deadlines at work. My manager '
        "doesn't provide clear guidance, which makes it worse.”"
    )
    if len(quote) > 92:
        q1, q2 = quote[:92] + "…", "…" + quote[45:]
    else:
        q1, q2 = quote, ""

    body = [
        "Source chunk (verbatim participant text)",
        q1,
        q2.strip("…") if q2 else "",
        "Every arrow should be checkable against text like this.",
    ]
    # Drop empty second quote line if unused
    body = [b for b in body if b]
    frames.append(_draw_panel("Step 3 — Participant quote in context", body, highlight_line=1))

    frames.append(
        _draw_panel(
            "Exports keep the chain",
            [
                "Markdown + DOCX include an evidence appendix.",
                "JSON bundle stores chunk_text_by_id for audit trails.",
                "Cite the quote, not the model.",
            ],
            highlight_line=None,
        )
    )

    durations = [900, 1100, 1600, 1200]
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
