from __future__ import annotations

import ast
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .io import iter_jsonl, load_predictions, record_key

DIMENSIONS = (
    "Creativity",
    "Richness",
    "Visual Perception",
    "Logical Coherence",
    "Answer Accuracy",
    "Image Relationship Understanding",
)

TRANSLATE_PROMPT = (
    "You are a language expert specialized in English and {language}. "
    "Please translate the following content into English. Please maintain the "
    "format of input, and do not output anything else other than translation. "
    "Input to be translated:\n\n{text}\n\nYour translation:"
)

JUDGE_PROMPT = (
    Path(__file__).with_name("prompts").joinpath("miqa_judge.txt").read_text(encoding="utf-8")
)


def build_judge_prompt(question: str, reference: str, response: str) -> str:
    return (
        JUDGE_PROMPT
        + "\n[Question]\n"
        + question
        + "\n\n[The Start of Reference Answer]\n"
        + reference
        + "\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n"
        + response
        + "\n[The End of Assistant's Answer]"
    )


def extract_scores(text: str) -> dict[str, int] | None:
    if not text:
        return None
    candidates = ["{" + match + "}" for match in re.findall(r"\{\{(.*?)\}\}", text, re.DOTALL)]
    candidates.extend(re.findall(r"\{[^{}]*\}", text, re.DOTALL))
    for candidate in reversed(candidates):
        try:
            value = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            normalized = _normalize_scores(value)
            if normalized:
                return normalized
    pairs = re.findall(r"['\"]?([A-Za-z][A-Za-z ]+?)['\"]?\s*[:：]\s*(\d{1,2})", text)
    return _normalize_scores(dict(pairs))


def _normalize_scores(value: dict[Any, Any]) -> dict[str, int] | None:
    canonical = {dimension.lower(): dimension for dimension in DIMENSIONS + ("Overall Score",)}
    result = {}
    for key, score in value.items():
        name = canonical.get(str(key).strip().lower())
        if not name:
            continue
        try:
            number = int(score)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= 10:
            result[name] = number
    return result if all(dimension in result for dimension in DIMENSIONS) else None


def aggregate_scores(records: list[dict[str, Any]]) -> float:
    values = [
        int(record["scores"][dimension])
        for record in records
        if record.get("scores")
        for dimension in DIMENSIONS
        if dimension in record["scores"]
    ]
    return 10.0 * sum(values) / len(values) if values else 0.0


class OpenAICompatibleChat:
    def __init__(self, model: str, api_key_env: str, base_url: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("Install pm4bench[judge] to run MIQA judging") from error
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {api_key_env} is not set")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            top_p=0.001,
            max_tokens=16384,
        )
        return response.choices[0].message.content or ""


def judge_manifest(
    manifest_path: Path,
    predictions_path: Path,
    output_path: Path,
    judge: OpenAICompatibleChat,
    translator: OpenAICompatibleChat | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    predictions = load_predictions(predictions_path)
    completed = {record_key(row) for row in iter_jsonl(output_path)} if output_path.exists() else set()
    pending = []
    for gold in iter_jsonl(manifest_path):
        key = record_key(gold)
        prediction = predictions.get(key)
        if key in completed or prediction is None:
            continue
        pending.append((key, gold, str(prediction.get("response", ""))))

    def process(item: tuple[str, dict[str, Any], str]) -> dict[str, Any]:
        key, gold, response = item
        translated = None
        judged_response = response
        if translator is not None and gold["language"] != "en":
            translated = translator.chat(
                TRANSLATE_PROMPT.format(language=gold["language"], text=response)
            )
            judged_response = translated
        judge_text = judge.chat(
            build_judge_prompt(gold["question"], gold["reference_answer"], judged_response)
        )
        return {
            "id": key,
            "language": gold["language"],
            "response": response,
            "translated_response": translated,
            "judge": judge_text,
            "scores": extract_scores(judge_text),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_handle = output_path.open("a", encoding="utf-8")
    with output_handle as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, item) for item in pending]
        for future in as_completed(futures):
            row = future.result()
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
    rows = list(iter_jsonl(output_path))
    return {"processed": len(rows), "score": aggregate_scores(rows)}
