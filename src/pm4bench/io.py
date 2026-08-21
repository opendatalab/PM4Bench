from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def record_key(record: dict[str, Any]) -> str:
    if record.get("id") is not None:
        return str(record["id"])
    if record.get("index") is not None:
        return str(record["index"])
    raise ValueError("Record has neither id nor index")


def load_predictions(path: str | Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        key = record_key(record)
        if key in result:
            raise ValueError(f"Duplicate prediction id: {key}")
        result[key] = record
    return result


def resolve_asset(dataset_root: str | Path, relative_path: str) -> Path:
    root = Path(dataset_root).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Asset escapes dataset root: {relative_path}") from error
    return candidate
