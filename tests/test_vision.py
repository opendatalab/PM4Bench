import copy
import json
from pathlib import Path

import pytest

from pm4bench.vision.inputs import load_manifest, miqa_plan, msocr_plan
from pm4bench.vision.render import render_vision


def miqa_row(tmp_path, language="en"):
    paths = []
    for index in (1, 2, 3):
        path = tmp_path / "assets" / f"image_{index}.png"
        path.parent.mkdir(exist_ok=True)
        path.touch()
        paths.append(path.relative_to(tmp_path).as_posix())
    question = "Compare Image1, Image3, and Image4."
    labels = "Image 1:\nImage 3:\nImage 4:\n"
    if language == "ar":
        labels = "1 صورة:\n3 صورة:\n4 صورة:\n"
    return {
        "id": "0_18", "task": "miqa", "language": language,
        "question": question, "traditional_images": paths,
        "rendered_text": labels + question + "\n",
    }


@pytest.mark.parametrize("language", ["en", "ar"])
def test_miqa_preserves_nonconsecutive_labels(tmp_path, language):
    row = miqa_row(tmp_path, language)
    plan = miqa_plan(row, tmp_path)
    assert [b["source_number"] for b in plan["blocks"] if b["type"] == "image"] == [1, 3, 4]
    assert "".join(b["text"] for b in plan["blocks"] if b["type"] == "text") == row["rendered_text"]


def test_miqa_checks_question_after_placeholder_prefix(tmp_path):
    row = miqa_row(tmp_path)
    row["question"] = "Image1: <ImageHere>. Image3: <ImageHere>. " + row["question"]
    miqa_plan(row, tmp_path)
    row["question"] += " A silently changed sentence."
    with pytest.raises(ValueError, match="disagree"):
        miqa_plan(row, tmp_path)


def test_miqa_sorts_double_digit_assets_numerically(tmp_path):
    row = miqa_row(tmp_path)
    paths = []
    for index in range(1, 12):
        path = tmp_path / "assets" / f"image_{index}.png"
        path.touch()
        paths.append(path.relative_to(tmp_path).as_posix())
    row["traditional_images"] = sorted(paths)  # v2.0.0's 1, 10, 11, 2, ... listing
    row["rendered_text"] = "".join(f"Image {i}:\n" for i in range(1, 12)) + row["question"] + "\n"
    plan = miqa_plan(row, tmp_path)
    assert [b["path"] for b in plan["blocks"] if b["type"] == "image"] == paths


def test_msocr_uses_exact_ordered_lines(tmp_path):
    lines = [{"font_size": n, "text": f"Line {n}"} for n in range(40, 0, -2)]
    row = {
        "id": "en_001", "task": "msocr", "language": "en", "image_size": [1280, 720],
        "font_gradient": list(range(40, 0, -2)), "lines": lines,
        "ground_truth": " ".join(x["text"] for x in lines),
    }
    before = copy.deepcopy(row)
    assert msocr_plan(row, tmp_path)["lines"] == lines
    assert row == before
    row["lines"][0]["text"] = "Replacement text"
    with pytest.raises(ValueError, match="disagree"):
        msocr_plan(row, tmp_path)


def test_manifest_rejects_unpublished_text_by_default(tmp_path):
    path = tmp_path / "data/miqa/en.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(miqa_row(tmp_path)) + "\n")
    with pytest.raises(ValueError, match="published v2.0.0"):
        load_manifest(tmp_path, "miqa", "en")
    assert len(load_manifest(tmp_path, "miqa", "en", custom=True)[0]) == 1


def test_audit_does_not_need_renderer_and_protects_dataset(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    row = miqa_row(root)
    manifest = root / "data/miqa/en.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(row) + "\n")
    output = tmp_path / "audit"
    report = render_vision(root, output, "miqa", ("en",), audit_only=True,
                           allow_custom_manifest=True)
    assert report["records"] == 1
    assert report["images"] == 0
    assert json.loads((output / "plans.jsonl").read_text())["id"] == "0_18"
    with pytest.raises(ValueError, match="outside"):
        render_vision(root, root / "output", "miqa", ("en",), audit_only=True)
    with pytest.raises(FileExistsError):
        render_vision(root, output, "miqa", ("en",), audit_only=True)


def test_raster_smoke(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from pm4bench.vision.raster import render_miqa, render_msocr

    # A test font only; released synthesis requires the per-language Noto fonts.
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.exists():
        pytest.skip("system test font unavailable")
    row = miqa_row(tmp_path)
    for name in row["traditional_images"]:
        Image.new("RGB", (180, 90), "blue").save(tmp_path / name)
    render_miqa(miqa_plan(row, tmp_path), tmp_path, font, tmp_path / "miqa.png")
    with Image.open(tmp_path / "miqa.png") as image:
        assert image.width == 1320
        assert image.height > 3 * 90
    render_msocr({"id": "test", "image_size": [1280, 720],
                  "lines": [{"text": "Keep exact words", "font_size": 40}]},
                 font, tmp_path / "msocr.png")
    with Image.open(tmp_path / "msocr.png") as image:
        assert image.size == (1280, 720)
        assert image.getextrema()[0][0] == 0


@pytest.mark.parametrize("language,text", [("en", "gypq"), ("ar", "1 صورة:")])
def test_text_blocks_keep_descenders(language, text):
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw, ImageOps

    from pm4bench.vision.raster import load_font, text_image

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font_path.exists():
        pytest.skip("system test font unavailable")
    font = load_font(str(font_path), 42)
    reference = Image.new("RGB", (1280, 256), "white")
    draw = ImageDraw.Draw(reference)
    box = draw.textbbox((0, 0), text + "\n", font=font)
    x = 1280 - box[2] if language == "ar" else max(0, -box[0])
    draw.text((x, 0), text + "\n", font=font, fill="black")
    result = text_image(text, language, font_path)
    assert ImageOps.invert(result).getbbox() == ImageOps.invert(reference).getbbox()
