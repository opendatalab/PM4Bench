"""Access to the released task prompts and their response-format contracts."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

RESOURCE = Path(__file__).with_name("prompts") / "task_prompts.json"
OPTION_LINE_RE = re.compile(r"^\(([A-Z])\)\s\s?(.*)$")


@lru_cache(maxsize=1)
def load_task_prompts() -> dict[str, Any]:
    return json.loads(RESOURCE.read_text(encoding="utf-8"))


def prompt_ids() -> tuple[str, ...]:
    return tuple(sorted(load_task_prompts()["prompts"]))


def prompt_entry(prompt_id: str) -> dict[str, Any]:
    prompts = load_task_prompts()["prompts"]
    if prompt_id not in prompts:
        raise ValueError(f"unknown prompt id {prompt_id!r}; expected one of {prompt_ids()}")
    return prompts[prompt_id]


def response_format(prompt_id: str) -> dict[str, Any]:
    entry = prompt_entry(prompt_id)
    return load_task_prompts()["response_formats"][entry["response_format"]]


def prompt_text(prompt_id: str, language: str) -> str:
    text = prompt_entry(prompt_id)["text"]
    if language not in text:
        raise ValueError(f"{prompt_id} has no {language!r} text")
    return text[language]


def _mdur_body(record: dict[str, Any], language: str, rule: dict[str, Any]) -> str:
    rendered = record["rendered_text"]
    head = f"{rule['question_label'][language]}: "
    separator = f"\n{rule['options_label'][language]}:\n"
    if not rendered.startswith(head) or separator not in rendered:
        raise ValueError("rendered_text does not have the expected MDUR layout")
    stem, _, block = rendered.partition(separator)
    stem = stem[len(head):]
    options = "\n".join(
        f"({match.group(1)}). {match.group(2)}" if (match := OPTION_LINE_RE.match(line)) else line
        for line in block.rstrip("\n").split("\n")
    )
    body = f"{stem}\n\n{options}\n\n{rule['answer_prefix'][language]}"
    rewrite = rule["image_token_rewrite"]
    for index in range(1, rewrite["n_max"] + 1):
        body = body.replace(rewrite["from"].replace("N", str(index)), rewrite["to"])
    return body


def _miqa_body(record: dict[str, Any], rule: dict[str, Any]) -> str:
    lines = []
    for line in record["rendered_text"].split("\n"):
        stripped = line.strip()
        if stripped and stripped[-1] in ":：":
            lines.append(f"{stripped} {rule['image_placeholder']}")
        else:
            lines.append(line)
    return "\n".join(lines).rstrip("\n")


def build_query(prompt_id: str, record: dict[str, Any],
                language: str | None = None) -> str:
    """Combine a released manifest row with its prompt into a ready query.

    Images are attached separately by the caller; `<image>` marks their position.
    """
    entry = prompt_entry(prompt_id)
    language = language or record.get("language")
    if not language:
        raise ValueError("language is required when the record has no language field")
    prompt = prompt_text(prompt_id, language)
    composition = entry["composition"]
    if composition == "standalone":
        return prompt
    if composition == "format_template":
        return prompt.format(question=record["question"])
    rule = entry["assembly"]
    body = (_miqa_body(record, rule) if entry["task"] == "miqa"
            else _mdur_body(record, language, rule))
    return f"{prompt}{rule['separator']}{body}"


def describe_prompts(*, prompt_id: str | None = None, task: str | None = None,
                     probe: str | None = None, language: str | None = None,
                     include_text: bool = False) -> dict[str, Any]:
    resource = load_task_prompts()
    selected = {}
    for identifier, entry in sorted(resource["prompts"].items()):
        if prompt_id and identifier != prompt_id:
            continue
        if task and entry["task"] != task:
            continue
        if probe and entry["probe"] != probe:
            continue
        described = {key: value for key, value in entry.items() if key != "text"}
        described["response_format_detail"] = resource["response_formats"][
            entry["response_format"]]
        if include_text:
            described["text"] = ({language: prompt_text(identifier, language)}
                                 if language else dict(entry["text"]))
        selected[identifier] = described
    if prompt_id and not selected:
        raise ValueError(f"unknown prompt id {prompt_id!r}; expected one of {prompt_ids()}")
    return {"schema_version": resource["schema_version"], "prompts": selected}
