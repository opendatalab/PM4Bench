from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

ANSWER_TAG_RE = re.compile(
    r"<\s*Start\s*of\s*Answer\s*>(.*?)<\s*End\s*of\s*Answer\s*>",
    re.DOTALL | re.IGNORECASE,
)
COORD_RE = re.compile(
    r"(?:['\"]?\s*x\s*['\"]?\s*[:=]\s*)?"
    r"(-?\d+(?:\.\d+)?)\s*[,，]\s*"
    r"(?:['\"]?\s*y\s*['\"]?\s*[:=]\s*)?"
    r"(-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value or ""))


def extract_answer_letter(value: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", value or "", flags=re.DOTALL)
    explicit = re.findall(r"(?:answer|option)\s*[:：]?\s*\(?([A-J])\)?", text, re.IGNORECASE)
    if explicit:
        return explicit[-1].upper()
    boxed = re.findall(r"\\boxed\{\s*([A-J])\s*\}", text, re.IGNORECASE)
    if boxed:
        return boxed[-1].upper()
    standalone = re.findall(r"(?<![A-Za-z])([A-J])(?![A-Za-z])", text, re.IGNORECASE)
    return standalone[-1].upper() if standalone else None


def mdur_correct(response: str, answer_key: str) -> bool:
    return extract_answer_letter(response) == answer_key.strip().upper()


def ocr_contains(response: str, reference: str) -> bool:
    normalized_reference = normalize_text(reference)
    return bool(normalized_reference) and normalized_reference in normalize_text(response)


def miqa_ocr_reference(rendered_text: str) -> str:
    """Extract the instruction substring used by the paper's MIQA OCR probe."""
    last_colon_newline = rendered_text.rfind(":\n")
    last_newline = rendered_text.rfind("\n")
    if last_colon_newline == -1 or last_newline <= last_colon_newline:
        return ""
    return rendered_text[last_colon_newline + 2 : last_newline].strip()


def miqa_ocr_correct(response: str, rendered_text: str) -> bool:
    prediction = normalize_text(response.replace("\n", " "))
    reference = normalize_text(miqa_ocr_reference(rendered_text))
    return len(prediction) > 10 and bool(reference) and reference in prediction


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        return levenshtein_distance(right, left)
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def normalized_edit_similarity(response: str, reference: str) -> float:
    prediction = normalize_text(response)
    target = normalize_text(reference)
    if not prediction and not target:
        return 1.0
    denominator = max(len(prediction), len(target))
    if denominator == 0:
        return 0.0
    return 1.0 - levenshtein_distance(prediction, target) / denominator


def _clean_msocr_text(value: str) -> str:
    if not value:
        return ""
    value = (
        value.split("<start>")[-1]
        .split("<Start>")[-1]
        .split("<end>")[0]
        .split("<End>")[0]
        .split("</end>")[0]
        .strip()
    )
    value = re.sub(r"[^\w\u0600-\u06FF\u0E00-\u0E7F]", "", value, flags=re.UNICODE)
    return re.sub(r"<.*?>", "", value)


def msocr_score(response: str, lines: list[dict[str, Any]], maximum: int = 40) -> float:
    """Return the paper's 0-40 first-recognition-error score."""
    prediction = _clean_msocr_text(response)
    if not prediction:
        return 0.0
    reference_chars = []
    font_sizes = []
    for line in lines:
        target = _clean_msocr_text(str(line["text"]))
        reference_chars.extend(target)
        font_sizes.extend([int(line["font_size"])] * len(target))
    for index, (predicted, reference) in enumerate(zip(prediction, reference_chars)):
        if predicted != reference:
            return float(maximum - font_sizes[max(0, index - 1)])
    if len(prediction) != len(reference_chars):
        boundary = min(len(prediction), len(reference_chars))
        if boundary < len(font_sizes):
            return float(maximum - font_sizes[boundary])
    return float(maximum)


ANSWER_TRIM_CHARS = "\"'`“”‘’«»「」『』()[]{}<>.,;:!?、。，；：！？ \t\n\r"


def _strip_symbols(value: str) -> str:
    return "".join(
        character
        for character in value
        if unicodedata.category(character) != "So" and character != "\ufe0f"
    )


def normalize_mgui_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = _strip_symbols(normalized)
    return re.sub(r"\s+", "", normalized).casefold()


def extract_answer_text(response: str) -> str:
    if not response:
        return ""
    stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    tagged = ANSWER_TAG_RE.findall(stripped) or ANSWER_TAG_RE.findall(response)
    if tagged:
        candidate = tagged[-1].strip().strip(ANSWER_TRIM_CHARS)
        if candidate:
            return candidate
    for line in reversed(stripped.splitlines()):
        candidate = line.strip().strip(ANSWER_TRIM_CHARS)
        if candidate and not re.match(r"(?i)^<?\s*(start|end)\s*of\s*answer", candidate):
            return candidate
    return stripped.strip()


def mgui_text_correct(response: str, reference: str) -> bool:
    prediction = normalize_mgui_text(extract_answer_text(response))
    target = normalize_mgui_text(reference)
    return bool(target) and (target == prediction or target in prediction or prediction in target)


def extract_coords(response: str) -> tuple[float, float] | None:
    if not response:
        return None
    stripped = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    tagged = ANSWER_TAG_RE.findall(stripped) or ANSWER_TAG_RE.findall(response)
    search_spaces = ([tagged[-1]] if tagged else []) + [stripped, response]
    for text in search_spaces:
        matches = COORD_RE.findall(text)
        if matches:
            x, y = matches[-1]
            return float(x), float(y)
    return None


def denormalize_coords(
    coords: tuple[float, float], image_size: list[int] | tuple[int, int], scale: int = 1000
) -> tuple[float, float]:
    width, height = image_size
    return coords[0] / scale * width, coords[1] / scale * height


def point_in_bbox(coords: tuple[float, float], bbox: dict[str, float]) -> bool:
    x, y = coords
    return (
        float(bbox["x"]) <= x <= float(bbox["x"]) + float(bbox["width"])
        and float(bbox["y"]) <= y <= float(bbox["y"]) + float(bbox["height"])
    )


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0
