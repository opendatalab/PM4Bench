"""MIQA input planning and raster compositor."""
from __future__ import annotations

import re
from pathlib import Path

from ...data.benchmark import image_map, text_digest
from ...io import resolve_asset
from ..raster import text_image

IMAGE_LABELS = {
    'ar': 'صورة', 'cs': 'Obrázek', 'en': 'Image', 'hu': 'Kép', 'ko': '이미지',
    'ru': 'Изображение', 'sr': 'Слика', 'th': 'ภาพ', 'vi': 'Hình ảnh', 'zh': '图片',
}


def miqa_plan(row: dict, root: Path) -> dict:
    """Retain source image labels, including non-consecutive labels such as 1, 3, 4."""
    images = image_map(row, root)
    rendered = row["rendered_text"]
    blocks = []
    remaining = rendered
    label = re.escape(IMAGE_LABELS[row["language"]])
    label_pattern = re.compile(
        rf"^(\d+) {label}:\n" if row["language"] == "ar" else rf"^{label} (\d+):\n"
    )
    labels = []
    for index in sorted(images):
        match = label_pattern.match(remaining)
        if not match:
            raise ValueError(f"{row['id']}: missing image label {index} in rendered_text")
        labels.append(int(match[1]))
        blocks.extend([
            {"type": "text", "text": match[0]},
            {"type": "image", "path": images[index], "source_number": int(match[1])},
        ])
        remaining = remaining[match.end():]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{row['id']}: duplicate source image labels")
    # This is the historical single-question renderer's prefix removal. It is
    # checked against the released OCR text before any content can be rendered.
    question = row["question"]
    question = question.split("<ImageHere>.")[-1].split("<ImageHere>。")[-1]
    question = question.split("<ImageHere>")[-1].strip()
    if remaining != question + "\n":
        raise ValueError(f"{row['id']}: question and rendered_text disagree")
    blocks.append({"type": "text", "text": remaining})
    if "".join(b["text"] for b in blocks if b["type"] == "text") != rendered:
        raise ValueError(f"{row['id']}: rendered text changed")
    return {
        "id": row["id"], "task": "miqa", "language": row["language"],
        "question_sha256": text_digest(row["question"]),
        "rendered_text_sha256": text_digest(rendered), "blocks": blocks,
    }


def render_miqa(plan: dict, root: Path, font_path: Path, destination: Path) -> None:
    from PIL import Image

    images = []
    for block in plan['blocks']:
        if block['type'] == 'text':
            image = text_image(block['text'], plan['language'], font_path)
        else:
            with Image.open(resolve_asset(root, block['path'])) as source:
                image = source.convert('RGB')
            if image.width > 1200:
                image = image.resize((1200, int(1200 * image.height / image.width)))
            if image.height > 700:
                image = image.resize((int(700 * image.width / image.height), 700))
        images.append(image)
    padding = 20
    height = sum(image.height for image in images) + 2 * padding + 10 * len(images)
    canvas = Image.new('RGB', (1320, height), 'white')
    y = padding
    for image in images:
        x = canvas.width - padding - image.width if plan['language'] == 'ar' else padding
        canvas.paste(image, (x, y))
        y += image.height + 10
    canvas.save(destination)
