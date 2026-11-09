"""Access to the released task prompts and their response-format contracts."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

RESOURCE = Path(__file__).with_name("prompts") / "task_prompts.json"


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
