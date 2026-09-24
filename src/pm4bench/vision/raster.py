from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=128)
def load_font(path: str, size: int):
    from PIL import ImageFont, features

    if not features.check_feature("raqm"):
        raise RuntimeError("Pillow needs libraqm for multilingual shaping; install the vision extra.")
    font = ImageFont.truetype(path, size, layout_engine=ImageFont.Layout.RAQM)
    try:
        names = font.get_variation_names()
    except OSError:  # Static fonts have no variation table.
        names = []
    if names:
        if b"Regular" not in names:
            raise ValueError(f"Variable font {Path(path).name} has no Regular instance")
        font.set_variation_by_name(b"Regular")
    return font


def missing_glyphs(font_path: Path, texts: list[str]) -> list[str]:
    from fontTools.ttLib import TTFont

    with TTFont(font_path) as font:
        cmap = font.getBestCmap()
    # Bidi controls, joiners and variation selectors do not require visible glyphs.
    import unicodedata

    missing = sorted({
        ord(char) for text in texts for char in text
        if not char.isspace() and unicodedata.category(char) not in {"Cf", "Cc"}
        and not 0xFE00 <= ord(char) <= 0xFE0F and ord(char) not in cmap
    })
    return [f"U+{code:04X}" for code in missing]


def text_image(text: str, language: str, font_path: Path):
    from PIL import Image, ImageDraw

    width = 1280
    font = load_font(str(font_path), 42)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = []
    # Preserve the original script-specific wrapping and 10 px line spacing.
    for paragraph in text.strip().split("\n"):
        character_wrap = language in {"zh", "th", "ko"}
        words = paragraph if character_wrap else paragraph.split(" ")
        separator = "" if character_wrap else " "
        line = ""
        for word in words:
            candidate = line + separator + word if line else word
            box = probe.textbbox((0, 0), candidate, font=font)
            if box[2] - box[0] <= width:
                line = candidate
            else:
                if not line:
                    raise ValueError("A word exceeds the MIQA canvas width with this font")
                lines.append(line)
                line = word
        if line:
            lines.append(line)
    if not lines:
        raise ValueError("Empty MIQA text block")
    # The historical renderer appends a newline to the last wrapped line.
    lines[-1] += "\n"
    boxes = [probe.textbbox((0, 0), line, font=font) for line in lines]
    # Font ink can start below the draw origin. Summing bbox heights alone
    # cuts off the final line's descenders (especially with Arabic fonts).
    placements = []
    y = 0
    ink_bottom = 0
    for line, box in zip(lines, boxes):
        x = width - box[2] if language == "ar" else max(0, -box[0])
        placements.append((line, x, y))
        ink_bottom = max(ink_bottom, y + box[3])
        y += box[3] - box[1] + 10
    height = max(y, ink_bottom + 2)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for line, x, y in placements:
        draw.text((x, y), line, font=font, fill="black")
    return image


def render_miqa(*args, **kwargs):
    """Compatibility wrapper for the task compositor."""
    from .tasks.miqa import render_miqa as render
    return render(*args, **kwargs)


def render_msocr(*args, **kwargs):
    """Compatibility wrapper for the task compositor."""
    from .tasks.msocr import render_msocr as render
    return render(*args, **kwargs)
