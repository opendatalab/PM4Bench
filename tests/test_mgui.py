import json

from pm4bench.mgui.render import compare_gt, normalize_historical_text


def test_normalize_historical_method_address():
    left = "Copy <built-in method copy of dict object at 0x123abc>"
    right = "Copy <built-in method copy of dict object at 0x999def>"
    assert normalize_historical_text(left) == normalize_historical_text(right)


def test_compare_gt_accepts_only_address_drift(tmp_path):
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    base = {
        "all_qid_boxes": {
            "copy": {
                "x": 1.0,
                "y": 2.0,
                "width": 3.0,
                "height": 4.0,
                "text": "<built-in method copy of dict object at 0x123abc>",
            }
        },
        "questions": [],
    }
    other = json.loads(json.dumps(base))
    other["all_qid_boxes"]["copy"]["text"] = (
        "<built-in method copy of dict object at 0x999def>"
    )
    (candidate / "sample.json").write_text(json.dumps(base), encoding="utf-8")
    (reference / "sample.json").write_text(json.dumps(other), encoding="utf-8")
    assert compare_gt(candidate, reference)["passed"]


def test_structure_comparison_ignores_geometry_only(tmp_path):
    candidate = tmp_path / "candidate"
    reference = tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    base = {
        "all_qid_boxes": {
            "target": {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0, "text": "Go"}
        },
        "questions": [],
    }
    other = json.loads(json.dumps(base))
    other["all_qid_boxes"]["target"]["width"] = 99.0
    (candidate / "sample.json").write_text(json.dumps(base), encoding="utf-8")
    (reference / "sample.json").write_text(json.dumps(other), encoding="utf-8")
    assert not compare_gt(candidate, reference)["passed"]
    result = compare_gt(candidate, reference, compare_geometry=False)
    assert result["passed"]
    assert result["comparison_mode"] == "structure"
