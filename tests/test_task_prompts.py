import json
import sys
from pathlib import Path

import pytest

from pm4bench.cli import build_parser, main
from pm4bench.data import LANGUAGES
from pm4bench.metrics import (
    extract_answer_letter,
    extract_answer_text,
    extract_coords,
    msocr_score,
    ocr_contains,
)
from pm4bench.task_prompts import (
    RESOURCE,
    describe_prompts,
    load_task_prompts,
    prompt_entry,
    prompt_ids,
    prompt_text,
    response_format,
)

RELEASED = (
    "mdur-ocr-vision",
    "mdur-vqa-traditional",
    "mdur-vqa-vision",
    "mdur-vqa-vision-reference-text",
    "mgui-ocr-screenshot",
    "mgui-vqa-screenshot",
    "miqa-ocr-vision",
    "miqa-vqa-traditional",
    "miqa-vqa-vision",
    "msocr-ocr-vision",
)
TEMPLATES = ("mgui-vqa-screenshot", "mgui-ocr-screenshot")


def test_released_prompt_ids_are_frozen() -> None:
    assert prompt_ids() == RELEASED


def test_every_prompt_covers_every_language() -> None:
    for prompt_id in RELEASED:
        text = prompt_entry(prompt_id)["text"]
        assert set(text) == set(LANGUAGES), prompt_id
        for language, value in text.items():
            assert isinstance(value, str) and value.strip(), (prompt_id, language)
            assert not value.startswith((" ", "\t", "\n")), (prompt_id, language)


def test_trailing_whitespace_is_preserved() -> None:
    """These exact endings are load-bearing for paper-era query assembly."""
    assert prompt_text("miqa-vqa-traditional", "en").endswith("Question: ")
    assert prompt_text("mdur-vqa-traditional", "en").endswith("Question:")


def test_response_markers_appear_in_every_language() -> None:
    for prompt_id in RELEASED:
        literals = response_format(prompt_id)["prompt_literals"]
        assert literals, prompt_id
        for literal in literals:
            for language, value in prompt_entry(prompt_id)["text"].items():
                assert literal in value, (prompt_id, language, literal)


def test_composition_discipline() -> None:
    for prompt_id in RELEASED:
        entry = prompt_entry(prompt_id)
        for value in entry["text"].values():
            if entry["composition"] == "format_template":
                assert "{question}" in value
                assert value.format(question="Q")
            else:
                assert "{question}" not in value
        assert (prompt_id in TEMPLATES) == (entry["composition"] == "format_template")


def test_mdur_ocr_prompt_is_not_a_format_template() -> None:
    """The JSON schema it asks for contains literal braces."""
    with pytest.raises(KeyError):
        prompt_text("mdur-ocr-vision", "en").format(question="Q")


def test_prefix_prompts_declare_their_reproducibility_limit() -> None:
    prefixes = {identifier for identifier in RELEASED
                if prompt_entry(identifier)["composition"] == "prefix"}
    assert prefixes == {"mdur-vqa-traditional", "mdur-vqa-vision-reference-text",
                        "miqa-vqa-traditional"}
    for prompt_id in prefixes:
        limit = prompt_entry(prompt_id)["reproducibility"]
        assert limit["reconstructable_from_released_fields"] is False
        assert 0 < limit["reconstructed_records"] < limit["total_records"]
        assert limit["blocking_causes"]


def test_example_responses_parse_as_documented() -> None:
    """The contract that stops a metric change from silently lowering scores."""
    formats = load_task_prompts()["response_formats"]
    assert extract_answer_letter(formats["option_letter_in_angles"]["example_response"]) == "C"
    assert extract_answer_text(formats["answer_tag"]["example_response"]) == "One way"
    assert extract_coords(formats["answer_tag_pixel_xy"]["example_response"]) == (318.0, 312.0)
    assert msocr_score(formats["lowercase_start_end"]["example_response"],
                       [{"font_size": 40, "text": "Spaniards Marrakesh"}]) == 40.0
    assert ocr_contains(formats["mdur_ocr_json"]["example_response"],
                        "Which is not a language?")


def test_evaluate_with_matches_the_live_cli() -> None:
    parser = build_parser()
    for prompt_id in RELEASED:
        target = prompt_entry(prompt_id)["evaluate_with"]
        if target["command"] != "evaluate":
            assert target["command"] == "judge-miqa"
            continue
        parsed = parser.parse_args(["evaluate", "--task", target["task"],
                                    "--manifest", "m", "--predictions", "p"])
        assert parsed.task == target["task"], prompt_id


def test_resource_is_canonically_serialized() -> None:
    expected = json.dumps(load_task_prompts(), ensure_ascii=False, indent=2,
                          sort_keys=True) + "\n"
    assert RESOURCE.read_text(encoding="utf-8") == expected


def test_published_fields_are_whitelisted() -> None:
    """Keeps internal source paths and legacy axes out of the public resource."""
    allowed = {"task", "probe", "setting", "evaluate_with", "composition",
               "response_format", "reproducibility", "verification", "digest", "text"}
    for prompt_id in RELEASED:
        assert set(prompt_entry(prompt_id)) <= allowed, prompt_id


def test_documentation_lists_every_prompt_and_marker() -> None:
    page = Path(__file__).resolve().parents[1] / "docs" / "PROMPTS.md"
    if not page.is_file():
        pytest.skip("documentation is not part of the installed package")
    content = page.read_text(encoding="utf-8")
    for prompt_id in RELEASED:
        assert prompt_id in content, prompt_id
    for detail in load_task_prompts()["response_formats"].values():
        for literal in detail["prompt_literals"]:
            assert literal in content, literal


def test_describe_prompts_filters_and_elides_text() -> None:
    index = describe_prompts()
    assert set(index["prompts"]) == set(RELEASED)
    assert all("text" not in entry for entry in index["prompts"].values())
    single = describe_prompts(prompt_id="mdur-vqa-vision", language="en", include_text=True)
    assert set(single["prompts"]["mdur-vqa-vision"]["text"]) == {"en"}
    assert set(describe_prompts(task="mgui")["prompts"]) == set(TEMPLATES)
    with pytest.raises(ValueError):
        describe_prompts(prompt_id="nope")


def test_mgui_evaluation_requires_an_explicit_coordinate_space(monkeypatch) -> None:
    arguments = ["evaluate", "--task", "mgui", "--manifest", "m", "--predictions", "p"]
    assert build_parser().parse_args(arguments).coordinate_space is None
    monkeypatch.setattr(sys, "argv", ["pm4bench", *arguments])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 2
