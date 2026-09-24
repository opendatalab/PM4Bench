from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import EXPECTED_COUNTS, LANGUAGES, TASKS
from .io import iter_jsonl, load_predictions, record_key, resolve_asset
from .metrics import (
    denormalize_coords,
    extract_coords,
    mdur_correct,
    mean,
    mgui_text_correct,
    miqa_ocr_correct,
    msocr_score,
    normalize_mgui_text,
    ocr_contains,
    point_in_bbox,
)
from .miqa import OpenAICompatibleChat, aggregate_scores, judge_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pm4bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a PM4Bench dataset snapshot")
    validate.add_argument("--dataset-root", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="run a deterministic task metric")
    evaluate.add_argument(
        "--task",
        choices=("mdur", "mdur-ocr", "miqa-ocr", "msocr", "mgui", "mgui-text"),
        required=True,
    )
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument(
        "--coordinate-space", choices=("pixel", "normalized-1000"), default="normalized-1000"
    )

    judge = subparsers.add_parser("judge-miqa", help="run resumable MIQA LLM-as-judge")
    judge.add_argument("--manifest", type=Path, required=True)
    judge.add_argument("--predictions", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--judge-model", required=True)
    judge.add_argument("--translate-model")
    judge.add_argument("--base-url")
    judge.add_argument("--api-key-env", default="OPENAI_API_KEY")
    judge.add_argument("--workers", type=int, default=8)

    summarize = subparsers.add_parser("summarize-miqa", help="summarize existing MIQA judge JSONL")
    summarize.add_argument("--judgments", type=Path, required=True)

    render = subparsers.add_parser("render-mgui", help="legacy MGUI template interface")
    render.add_argument("--templates-root", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--compare-gt", type=Path)
    render.add_argument("--bbox-tolerance", type=float, default=0.2)
    render.add_argument("--report", type=Path, help="write the JSON result atomically")
    render.add_argument("--browser-executable", type=Path)
    render.add_argument(
        "--comparison-mode", choices=("strict", "structure"), default="strict"
    )
    render.add_argument("--language", action="append", choices=LANGUAGES)
    render.add_argument("--template", action="append")

    vision = subparsers.add_parser("render-vision", help="synthesize vision data from released text")
    vision.add_argument("--dataset-root", type=Path, required=True)
    vision.add_argument("--output-root", type=Path, required=True)
    vision.add_argument("--task", choices=TASKS, required=True)
    vision.add_argument("--language", action="append", choices=LANGUAGES)
    vision.add_argument("--fonts-root", type=Path)
    vision.add_argument("--id", action="append")
    vision.add_argument("--limit", type=int, help="maximum records per language")
    vision.add_argument("--audit-only", action="store_true")
    vision.add_argument("--allow-custom-manifest", action="store_true")
    vision.add_argument("--strict-glyphs", action="store_true", help="fail on missing font glyphs")
    vision.add_argument("--workers", type=int, default=1)
    vision.add_argument("--browser-executable", type=Path)
    vision.add_argument("--segoe-root", type=Path, help="MDUR: optional reference font profile")
    vision.add_argument("--styles", type=Path, help="MDUR: saved style manifest")
    vision.add_argument("--no-fit", action="store_true", help="MDUR: skip text-style fitting")
    vision.add_argument("--comparison-mode", choices=("strict", "structure"), default="structure",
                        help="MGUI: compare reference structure or geometry")
    vision.add_argument("--bbox-tolerance", type=float, default=.2, help="MGUI geometry tolerance")
    return parser


def validate_snapshot(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset_root": str(root.resolve()), "tasks": {}}
    for task, expected in EXPECTED_COUNTS.items():
        task_dir = root / "data" / task
        language_counts = {}
        for manifest in sorted(task_dir.glob("*.jsonl")):
            rows = list(iter_jsonl(manifest))
            language = manifest.stem
            if len(rows) != expected:
                raise ValueError(f"{task}/{language}: expected {expected}, got {len(rows)}")
            for row in rows:
                for field in ("image", "vision_image"):
                    if row.get(field) and not resolve_asset(root, row[field]).is_file():
                        raise FileNotFoundError(row[field])
                for field in ("traditional_images",):
                    for asset in row.get(field, []):
                        if not resolve_asset(root, asset).is_file():
                            raise FileNotFoundError(asset)
            language_counts[language] = len(rows)
        if set(language_counts) != set(LANGUAGES):
            raise ValueError(f"{task}: incomplete languages: {sorted(language_counts)}")
        result["tasks"][task] = language_counts
    return result


def evaluate_task(args: argparse.Namespace) -> dict[str, Any]:
    predictions = load_predictions(args.predictions)
    scores = []
    missing = []
    skipped = 0
    for gold in iter_jsonl(args.manifest):
        if args.task == "mgui-text" and not normalize_mgui_text(gold.get("target_text", "")):
            skipped += 1
            continue
        key = record_key(gold)
        prediction = predictions.get(key)
        if prediction is None and gold.get("index") is not None:
            prediction = predictions.get(str(gold["index"]))
        if prediction is None:
            missing.append(key)
            scores.append(0.0)
            continue
        response = str(prediction.get("response", ""))
        if args.task == "mdur":
            score = float(mdur_correct(response, gold["answer_key"]))
        elif args.task == "mdur-ocr":
            score = float(ocr_contains(response, gold["question"]))
        elif args.task == "miqa-ocr":
            score = float(miqa_ocr_correct(response, gold["rendered_text"]))
        elif args.task == "msocr":
            score = msocr_score(response, gold["lines"])
        elif args.task == "mgui":
            coords = extract_coords(response)
            if coords is None:
                score = 0.0
            else:
                if args.coordinate_space == "normalized-1000":
                    coords = denormalize_coords(coords, gold["image_size"])
                score = float(point_in_bbox(coords, gold["bbox"]))
        else:
            score = float(mgui_text_correct(response, gold["target_text"]))
        scores.append(score)
    percentage_tasks = {"mdur", "mdur-ocr", "miqa-ocr", "mgui", "mgui-text"}
    scale = 100.0 if args.task in percentage_tasks else 1.0
    return {
        "task": args.task,
        "records": len(scores),
        "skipped": skipped,
        "missing_predictions": len(missing),
        "score": mean(scores) * scale,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "validate":
        result = validate_snapshot(args.dataset_root)
    elif args.command == "evaluate":
        result = evaluate_task(args)
    elif args.command == "summarize-miqa":
        rows = list(iter_jsonl(args.judgments))
        result = {"processed": len(rows), "score": aggregate_scores(rows)}
    elif args.command == "judge-miqa":
        judge = OpenAICompatibleChat(args.judge_model, args.api_key_env, args.base_url)
        translator = (
            OpenAICompatibleChat(args.translate_model, args.api_key_env, args.base_url)
            if args.translate_model
            else None
        )
        result = judge_manifest(
            args.manifest,
            args.predictions,
            args.output,
            judge,
            translator,
            args.workers,
        )
    elif args.command == "render-vision":
        from .vision.render import render_vision

        result = render_vision(
            args.dataset_root, args.output_root, args.task,
            tuple(args.language) if args.language else LANGUAGES,
            fonts_root=args.fonts_root, ids=tuple(args.id) if args.id else None,
            limit=args.limit, audit_only=args.audit_only,
            allow_custom_manifest=args.allow_custom_manifest,
            strict_glyphs=args.strict_glyphs,
            workers=args.workers,
            browser_executable=args.browser_executable, segoe_root=args.segoe_root,
            fit=not args.no_fit, styles=args.styles,
            comparison_mode=args.comparison_mode, bbox_tolerance=args.bbox_tolerance,
        )
    else:
        import warnings

        from .mgui.render import compare_gt, render_mgui

        warnings.warn("Prefer render-vision --task mgui --dataset-root; render-mgui remains "
                      "available for template-only callers.", FutureWarning, stacklevel=1)
        result = render_mgui(
            args.templates_root,
            args.output_root,
            languages=tuple(args.language) if args.language else LANGUAGES,
            browser_executable=args.browser_executable,
            template_names=tuple(args.template) if args.template else None,
        )
        if args.compare_gt:
            result["comparison"] = compare_gt(
                args.output_root / "gt",
                args.compare_gt,
                args.bbox_tolerance,
                compare_geometry=args.comparison_mode == "strict",
            )
            result["passed"] = result["comparison"]["passed"]
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if getattr(args, "report", None):
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.report)
    print(rendered, end="")
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
