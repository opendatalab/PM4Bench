"""MSOCR input planning and multi-scale raster compositor."""
from __future__ import annotations

from pathlib import Path

from ...data.benchmark import text_digest
from ..raster import load_font


def msocr_plan(row: dict, root: Path) -> dict:
    del root
    lines = row['lines']
    gradient = list(range(40, 0, -2))
    if [line['font_size'] for line in lines] != gradient or row['font_gradient'] != gradient:
        raise ValueError(f"{row['id']}: expected the 40-to-2 pixel font gradient")
    if ' '.join(line['text'] for line in lines) != row['ground_truth']:
        raise ValueError(f"{row['id']}: lines and ground_truth disagree")
    if row['image_size'] != [1280, 720] or any(not line['text'] for line in lines):
        raise ValueError(f"{row['id']}: invalid image dimensions or empty text")
    return {'id': row['id'], 'task': 'msocr', 'language': row['language'],
            'image_size': row['image_size'], 'lines': lines,
            'ground_truth_sha256': text_digest(row['ground_truth'])}


def render_msocr(plan: dict, font_path: Path, destination: Path) -> None:
    from PIL import Image, ImageDraw

    image = Image.new('RGB', tuple(plan['image_size']), 'white')
    draw = ImageDraw.Draw(image)
    y = 10
    for line in plan['lines']:
        font = load_font(str(font_path), line['font_size'])
        text = line['text']
        width = font.getlength(text)
        if width > image.width:
            raise ValueError(f"{plan['id']}: text exceeds canvas width with this font")
        draw.text(((image.width - width) // 2, y), text, font=font, fill='black')
        y += max(int(line['font_size'] * 1.5), 20)
    image.save(destination)
