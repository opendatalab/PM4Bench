from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..io import iter_jsonl, resolve_asset

RESOURCE_ROOT = Path(__file__).parent / "resources"
FONTS = {
    "ar": "NotoSansArabic-Regular.ttf",
    "cs": "NotoSans-Regular.ttf",
    "en": "NotoSans-Regular.ttf",
    "hu": "NotoSans-Regular.ttf",
    "ko": "NotoSansKR-Regular.ttf",
    "ru": "NotoSans-Regular.ttf",
    "sr": "NotoSans-Regular.ttf",
    "th": "NotoSansThai-Regular.ttf",
    "vi": "NotoSans-Regular.ttf",
    "zh": "NotoSansSC-Regular.ttf",
}
IMAGE_LABELS = {
    "ar": "صورة", "cs": "Obrázek", "en": "Image", "hu": "Kép", "ko": "이미지",
    "ru": "Изображение", "sr": "Слика", "th": "ภาพ", "vi": "Hình ảnh", "zh": "图片",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_manifest(root: Path, task: str, language: str, *, custom: bool = False):
    relative = f"data/{task}/{language}.jsonl"
    manifest = resolve_asset(root, relative)
    digest = sha256(manifest)
    lock = json.loads((RESOURCE_ROOT / "inputs.lock.json").read_text(encoding="utf-8"))
    if not custom and digest != lock["manifests"].get(relative):
        raise ValueError(
            f"{relative}: does not match the published v2.0.0 text manifest. "
            "Use that dataset revision, or --allow-custom-manifest for a separate experiment."
        )
    rows = list(iter_jsonl(manifest))
    seen = set()
    for row in rows:
        if row["task"] != task or row["language"] != language:
            raise ValueError(f"{relative}: task/language mismatch")
        key = str(row["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", key) or key in seen:
            raise ValueError(f"{relative}: invalid or duplicate id {key!r}")
        seen.add(key)
    if not rows:
        raise ValueError(f"{relative}: empty manifest")
    return rows, digest


def image_map(row: dict, root: Path) -> dict[int, str]:
    result = {}
    for name in row["traditional_images"]:
        path = resolve_asset(root, name)
        match = re.fullmatch(r"image_(\d+)\.png", path.name)
        if not match or not path.is_file():
            raise ValueError(f"{row['id']}: invalid/missing traditional image {name}")
        index = int(match[1])
        if index in result:
            raise ValueError(f"{row['id']}: duplicate image index {index}")
        result[index] = name
    if set(result) != set(range(1, len(result) + 1)):
        raise ValueError(f"{row['id']}: non-contiguous asset indices")
    return result


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


def msocr_plan(row: dict, root: Path) -> dict:
    del root
    lines = row["lines"]
    gradient = list(range(40, 0, -2))
    if [line["font_size"] for line in lines] != gradient or row["font_gradient"] != gradient:
        raise ValueError(f"{row['id']}: expected the 40-to-2 pixel font gradient")
    if " ".join(line["text"] for line in lines) != row["ground_truth"]:
        raise ValueError(f"{row['id']}: lines and ground_truth disagree")
    if row["image_size"] != [1280, 720] or any(not line["text"] for line in lines):
        raise ValueError(f"{row['id']}: invalid image dimensions or empty text")
    return {
        "id": row["id"], "task": "msocr", "language": row["language"],
        "image_size": row["image_size"], "lines": lines,
        "ground_truth_sha256": text_digest(row["ground_truth"]),
    }
