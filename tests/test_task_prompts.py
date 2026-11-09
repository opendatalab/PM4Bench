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
    build_query,
    describe_prompts,
    load_task_prompts,
    prompt_entry,
    prompt_ids,
    prompt_text,
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
PREFIXES = ("mdur-vqa-traditional", "mdur-vqa-vision-reference-text",
            "miqa-vqa-traditional")

# A documented response for every marker set, kept here rather than in the
# published resource so the resource stays free of explanatory material.
EXAMPLE_RESPONSES = {
    "mdur-vqa-vision": "Weighing the options...\n<C>",
    "msocr-ocr-vision": "<start>Spaniards Marrakesh<end>",
    "miqa-vqa-vision": "<Start of Answer>One way<End of Answer>",
    "mgui-vqa-screenshot": "<Start of Answer>[318, 312]<End of Answer>",
    "mdur-ocr-vision": '{"question": "Which is not a language?", "options": ["HTML"]}',
}


def test_released_prompt_ids_are_frozen() -> None:
    assert prompt_ids() == RELEASED


def test_schema_is_version_two_and_carries_only_data() -> None:
    resource = load_task_prompts()
    assert resource["schema_version"] == 2
    assert set(resource) == {"schema_version", "languages", "prompts"}
    allowed = {"task", "probe", "setting", "composition", "answer_markers",
               "scored_by", "assembly", "text"}
    for prompt_id in RELEASED:
        assert set(prompt_entry(prompt_id)) <= allowed, prompt_id


def test_every_prompt_covers_every_language() -> None:
    for prompt_id in RELEASED:
        text = prompt_entry(prompt_id)["text"]
        assert set(text) == set(LANGUAGES), prompt_id
        for language, value in text.items():
            assert isinstance(value, str) and value.strip(), (prompt_id, language)
            assert not value.startswith((" ", "\t", "\n")), (prompt_id, language)


def test_trailing_whitespace_is_preserved() -> None:
    """These exact endings are load-bearing for query assembly."""
    assert prompt_text("miqa-vqa-traditional", "en").endswith("Question: ")
    assert prompt_text("mdur-vqa-traditional", "en").endswith("Question:")


def test_answer_markers_appear_in_every_language() -> None:
    for prompt_id in RELEASED:
        markers = prompt_entry(prompt_id)["answer_markers"]
        assert markers, prompt_id
        for marker in markers:
            for language, value in prompt_entry(prompt_id)["text"].items():
                assert marker in value, (prompt_id, language, marker)


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
        assert ("assembly" in entry) == (entry["composition"] == "prefix")


def test_mdur_ocr_prompt_is_not_a_format_template() -> None:
    """The JSON schema it asks for contains literal braces."""
    with pytest.raises(KeyError):
        prompt_text("mdur-ocr-vision", "en").format(question="Q")


def test_prefix_prompts_carry_a_complete_assembly_rule() -> None:
    assert {p for p in RELEASED if prompt_entry(p)["composition"] == "prefix"} == set(PREFIXES)
    for prompt_id in PREFIXES:
        rule = prompt_entry(prompt_id)["assembly"]
        assert "{prompt}" in rule["template"]
        assert "separator" in rule
        if prompt_entry(prompt_id)["task"] == "mdur":
            for key in ("question_label", "options_label", "answer_prefix"):
                assert set(rule[key]) == set(LANGUAGES), (prompt_id, key)
            assert rule["option_line"] == "({letter}). {option}"
            assert rule["image_token"]["to"] == "<image>"
        else:
            assert rule["image_token"] == "<image>"
    assert prompt_entry("mdur-vqa-traditional")["assembly"]["separator"] == " "
    assert prompt_entry("mdur-vqa-vision-reference-text")["assembly"]["separator"] == ""


def test_build_query_assembles_each_composition() -> None:
    mdur = {"language": "en", "rendered_text":
            "Question: Which one?<image 1>\nOptions:\n(A)  first\n(B)  <image 2>",
            "options": ["first", ""], "question": "Which one?"}
    built = build_query("mdur-vqa-traditional", mdur)
    assert built.startswith(prompt_text("mdur-vqa-traditional", "en") + " ")
    assert "Which one?<image>" in built
    # relabeled, and the image-only option survives via rendered_text
    assert "(A). first\n(B). <image>" in built
    assert built.endswith("\n\nYour answer:")
    assert "<image 1>" not in built and "<image 2>" not in built

    assert build_query("mdur-vqa-vision-reference-text", mdur).startswith(
        prompt_text("mdur-vqa-vision-reference-text", "en") + "Which one?")

    miqa = {"language": "en", "rendered_text": "Image 1:\nImage 2:\nCompare them.\n"}
    assert build_query("miqa-vqa-traditional", miqa).endswith(
        "Image 1: <image>\nImage 2: <image>\nCompare them.")

    assert build_query("mdur-vqa-vision", mdur) == prompt_text("mdur-vqa-vision", "en")

    mgui = {"language": "en", "question": "Switch the trip type."}
    assert build_query("mgui-vqa-screenshot", mgui).endswith("Switch the trip type.")


def test_build_query_rejects_an_unusable_record() -> None:
    with pytest.raises(ValueError):
        build_query("mdur-vqa-traditional", {"rendered_text": "x"})
    with pytest.raises(ValueError):
        build_query("mdur-vqa-traditional", {"language": "en", "rendered_text": "no labels"})


def test_example_responses_parse_as_documented() -> None:
    """The contract that stops a metric change from silently lowering scores."""
    assert extract_answer_letter(EXAMPLE_RESPONSES["mdur-vqa-vision"]) == "C"
    assert extract_answer_text(EXAMPLE_RESPONSES["miqa-vqa-vision"]) == "One way"
    assert extract_coords(EXAMPLE_RESPONSES["mgui-vqa-screenshot"]) == (318.0, 312.0)
    assert msocr_score(EXAMPLE_RESPONSES["msocr-ocr-vision"],
                       [{"font_size": 40, "text": "Spaniards Marrakesh"}]) == 40.0
    assert ocr_contains(EXAMPLE_RESPONSES["mdur-ocr-vision"], "Which is not a language?")


def test_scored_by_matches_the_live_cli() -> None:
    parser = build_parser()
    for prompt_id in RELEASED:
        scored_by = prompt_entry(prompt_id)["scored_by"]
        if scored_by == "judge-miqa":
            continue
        assert scored_by.startswith("evaluate --task "), prompt_id
        task = scored_by.rsplit(" ", 1)[1]
        parsed = parser.parse_args(["evaluate", "--task", task,
                                    "--manifest", "m", "--predictions", "p"])
        assert parsed.task == task, prompt_id


def test_resource_is_canonically_serialized() -> None:
    expected = json.dumps(load_task_prompts(), ensure_ascii=False, indent=2,
                          sort_keys=True) + "\n"
    assert RESOURCE.read_text(encoding="utf-8") == expected


def test_documentation_lists_every_prompt_and_marker() -> None:
    page = Path(__file__).resolve().parents[1] / "docs" / "PROMPTS.md"
    if not page.is_file():
        pytest.skip("documentation is not part of the installed package")
    content = page.read_text(encoding="utf-8")
    for prompt_id in RELEASED:
        assert prompt_id in content, prompt_id
    for entry in load_task_prompts()["prompts"].values():
        for marker in entry["answer_markers"]:
            assert marker in content, marker


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
