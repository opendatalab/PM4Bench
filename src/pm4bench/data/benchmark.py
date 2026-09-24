"""Shared data access for the four benchmark tasks; independent of rendering."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ..io import iter_jsonl, resolve_asset

RESOURCE_ROOT = Path(__file__).parent / 'resources'
LANGUAGES = ('ar', 'cs', 'en', 'hu', 'ko', 'ru', 'sr', 'th', 'vi', 'zh')
EXPECTED_COUNTS = {'mdur': 1730, 'miqa': 218, 'msocr': 100, 'mgui': 200}
TASKS = tuple(EXPECTED_COUNTS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_manifest(root: Path, task: str, language: str, *, custom: bool = False):
    if task not in TASKS or language not in LANGUAGES:
        raise ValueError(f'Unknown task/language: {task}/{language}')
    relative = f'data/{task}/{language}.jsonl'
    manifest = resolve_asset(root, relative)
    digest = sha256(manifest)
    lock = json.loads((RESOURCE_ROOT / 'inputs.lock.json').read_text(encoding='utf-8'))
    if not custom and digest != lock['manifests'].get(relative):
        raise ValueError(f'{relative}: does not match the published v2.0.0 text manifest. '
                         'Use that revision or --allow-custom-manifest for a separate experiment.')
    rows = list(iter_jsonl(manifest))
    seen = set()
    for row in rows:
        if row['task'] != task or row['language'] != language:
            raise ValueError(f'{relative}: task/language mismatch')
        key = str(row['id'])
        if not re.fullmatch(r'[A-Za-z0-9_-]+', key) or key in seen:
            raise ValueError(f'{relative}: invalid or duplicate id {key!r}')
        seen.add(key)
    if not rows:
        raise ValueError(f'{relative}: empty manifest')
    return rows, digest


def select_records(root: Path, task: str, languages: tuple[str, ...], *,
                   ids: tuple[str, ...] | None = None, limit: int | None = None,
                   custom: bool = False) -> tuple[list[dict], dict]:
    if not languages or len(languages) != len(set(languages)):
        raise ValueError('Select one or more distinct languages')
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    selected, hashes = [], {}
    for language in languages:
        rows, digest = load_manifest(root, task, language, custom=custom)
        hashes[f'data/{task}/{language}.jsonl'] = digest
        rows = [row for row in rows if ids is None or row['id'] in ids]
        selected.extend(rows[:limit] if limit else rows)
    if ids and set(ids) - {row['id'] for row in selected}:
        raise ValueError('Requested ids were not found in the selected records')
    if not selected:
        raise ValueError('No records selected')
    return selected, hashes


def image_map(row: dict, root: Path) -> dict[int, str]:
    result = {}
    for name in row['traditional_images']:
        path = resolve_asset(root, name)
        match = re.fullmatch(r'image_(\d+)\.png', path.name)
        if not match or not path.is_file():
            raise ValueError(f"{row['id']}: invalid/missing traditional image {name}")
        index = int(match[1])
        if index in result:
            raise ValueError(f"{row['id']}: duplicate image index {index}")
        result[index] = name
    if set(result) != set(range(1, len(result) + 1)):
        raise ValueError(f"{row['id']}: non-contiguous asset indices")
    return result
