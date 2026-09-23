from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ..io import resolve_asset


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


def render_msocr(plan: dict, font_path: Path, destination: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", tuple(plan["image_size"]), "white")
    draw = ImageDraw.Draw(image)
    y = 10
    for line in plan["lines"]:
        font = load_font(str(font_path), line["font_size"])
        text = line["text"]
        width = font.getlength(text)
        if width > image.width:
            raise ValueError(f"{plan['id']}: text exceeds canvas width with this font")
        draw.text(((image.width - width) // 2, y), text, font=font, fill="black")
        y += max(int(line["font_size"] * 1.5), 20)
    image.save(destination)


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
    height = sum(box[3] - box[1] + 10 for box in boxes)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = 0
    for line, box in zip(lines, boxes):
        x = width - (box[2] - box[0]) if language == "ar" else 0
        draw.text((x, y), line, font=font, fill="black")
        y += box[3] - box[1] + 10
    return image


def render_miqa(plan: dict, root: Path, font_path: Path, destination: Path) -> None:
    from PIL import Image

    images = []
    for block in plan["blocks"]:
        if block["type"] == "text":
            image = text_image(block["text"], plan["language"], font_path)
        else:
            with Image.open(resolve_asset(root, block["path"])) as source:
                image = source.convert("RGB")
            if image.width > 1200:
                image = image.resize((1200, int(1200 * image.height / image.width)))
            if image.height > 700:
                image = image.resize((int(700 * image.width / image.height), 700))
        images.append(image)
    padding = 20
    height = sum(image.height for image in images) + 2 * padding + 10 * len(images)
    canvas = Image.new("RGB", (1320, height), "white")
    y = padding
    for image in images:
        x = canvas.width - padding - image.width if plan["language"] == "ar" else padding
        canvas.paste(image, (x, y))
        y += image.height + 10
    canvas.save(destination)
