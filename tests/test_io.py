import json

import pytest

from pm4bench.io import load_predictions, resolve_asset, write_jsonl


def test_prediction_round_trip_and_duplicates(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    write_jsonl(path, [{"id": "a", "response": "A"}])
    assert load_predictions(path)["a"]["response"] == "A"
    path.write_text(
        json.dumps({"id": "a", "response": "A"}) + "\n"
        + json.dumps({"id": "a", "response": "B"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_predictions(path)


def test_asset_resolution_stays_inside_snapshot(tmp_path) -> None:
    asset = tmp_path / "assets" / "image.png"
    asset.parent.mkdir()
    asset.touch()
    assert resolve_asset(tmp_path, "assets/image.png") == asset
    with pytest.raises(ValueError, match="escapes"):
        resolve_asset(tmp_path, "../outside.png")
