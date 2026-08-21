import math

from pm4bench.metrics import (
    denormalize_coords,
    extract_answer_letter,
    extract_coords,
    mdur_correct,
    mgui_text_correct,
    miqa_ocr_correct,
    msocr_score,
    normalized_edit_similarity,
    point_in_bbox,
)
from pm4bench.miqa import aggregate_scores, extract_scores


def test_extract_answer_letter() -> None:
    assert extract_answer_letter("Reasoning... Answer: (B)") == "B"
    assert mdur_correct(r"final: \boxed{c}", "C")


def test_edit_similarity() -> None:
    assert normalized_edit_similarity("a b c", "abc") == 1.0
    assert math.isclose(normalized_edit_similarity("abc", "abd"), 2 / 3)


def test_msocr_first_error() -> None:
    lines = [{"text": "Large", "font_size": 40}, {"text": "small", "font_size": 20}]
    assert msocr_score("Large small", lines) == 40
    assert msocr_score("Large sx", lines) == 20
    assert msocr_score("Large wrong", lines) == 0
    assert msocr_score("", lines) == 0


def test_gui_coordinates() -> None:
    coords = extract_coords("<Start of Answer>[500, 500]<End of Answer>")
    assert coords == (500.0, 500.0)
    pixels = denormalize_coords(coords, [1280, 800])
    assert pixels == (640.0, 400.0)
    assert point_in_bbox(pixels, {"x": 600, "y": 350, "width": 100, "height": 100})


def test_ocr_diagnostics() -> None:
    rendered = "Header:\nTarget instruction\nFooter"
    assert miqa_ocr_correct("prefix Target instruction suffix", rendered)
    assert mgui_text_correct("<Start of Answer>⭐ Sign up<End of Answer>", "Sign up")


def test_miqa_score_parsing() -> None:
    text = "summary {'Creativity': 7, 'Richness': 8, 'Visual Perception': 9, " \
        "'Logical Coherence': 8, 'Answer Accuracy': 7, " \
        "'Image Relationship Understanding': 9, 'Overall Score': 8}"
    scores = extract_scores(text)
    assert scores is not None
    assert math.isclose(aggregate_scores([{"scores": scores}]), 80.0)
