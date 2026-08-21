from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

LANGUAGES = ("ar", "cs", "en", "hu", "ko", "ru", "sr", "th", "vi", "zh")
FONT_STACK = (
    '"Noto Color Emoji", "Noto Sans", "Noto Sans Arabic", "Noto Sans SC", '
    '"Noto Sans KR", "Noto Sans Thai", "Segoe UI", Arial, sans-serif'
)
HISTORICAL_METHOD_ADDRESS = re.compile(
    r"(<built-in method (?:copy|update) of dict object at )0x[0-9a-fA-F]+(>)"
)


def normalize_historical_text(value: str) -> str:
    """Ignore only nondeterministic addresses emitted by two legacy templates."""
    return HISTORICAL_METHOD_ADDRESS.sub(r"\g<1>0x<address>\2", value)


def render_mgui(
    templates_root: Path,
    output_root: Path,
    languages: tuple[str, ...] = LANGUAGES,
    browser_executable: Path | None = None,
    template_names: tuple[str, ...] | None = None,
) -> dict[str, int]:
    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise RuntimeError("Install pm4bench[render] and run 'playwright install chromium'") from error

    environment = Environment(
        loader=FileSystemLoader(str(templates_root)),
        undefined=StrictUndefined,
        # The original generator rendered from a string with Jinja's
        # default-for-string autoescaping enabled. Preserve that behavior,
        # including escaped quotes in FONT_STACK, so released GT is reproducible.
        autoescape=True,
        keep_trailing_newline=True,
    )
    html_root = output_root / "html"
    image_root = output_root / "images"
    gt_root = output_root / "gt"
    for directory in (html_root, image_root, gt_root):
        directory.mkdir(parents=True, exist_ok=True)

    configs = sorted(templates_root.glob("*.json"))
    if template_names is not None:
        selected = set(template_names)
        configs = [path for path in configs if path.stem in selected]
        missing = selected - {path.stem for path in configs}
        if missing:
            raise FileNotFoundError(f"Unknown MGUI templates: {sorted(missing)}")
    counts = {"templates": len(configs), "renders": 0, "questions": 0}
    with sync_playwright() as playwright:
        launch_options: dict[str, Any] = {
            "headless": True,
            "args": ["--hide-scrollbars", "--disable-gpu"],
        }
        if browser_executable is not None:
            launch_options["executable_path"] = str(browser_executable)
        browser = playwright.chromium.launch(**launch_options)
        page = browser.new_page(viewport={"width": 1280, "height": 800}, device_scale_factor=1)
        for config_path in configs:
            config = json.load(config_path.open(encoding="utf-8"))
            template_name = config_path.stem
            template = environment.get_template(f"{template_name}.html.j2")
            width = int(config.get("width", 1280))
            height = int(config.get("height", 800))
            page.set_viewport_size({"width": width, "height": height})
            for language in languages:
                strings = {
                    key: translations[language]
                    for key, translations in config["strings"].items()
                }
                rendered = template.render(
                    LANG=language,
                    DIR="rtl" if language == "ar" else "ltr",
                    RTL=language == "ar",
                    FONT_STACK=FONT_STACK,
                    t=strings,
                )
                html_path = html_root / f"{template_name}_{language}.html"
                html_path.write_text(rendered, encoding="utf-8")
                page.goto("about:blank")
                page.goto(html_path.resolve().as_uri(), wait_until="load")
                page.evaluate(
                    "document.fonts && document.fonts.ready ? document.fonts.ready : true"
                )
                page.wait_for_timeout(800)
                page.evaluate(
                    "document.body.style.transform='translateZ(0)'; "
                    "void document.body.offsetHeight; "
                    "document.body.style.transform='';"
                )
                page.wait_for_timeout(300)
                image_name = f"{template_name}_{language}.png"
                page.screenshot(path=str(image_root / image_name), full_page=False)
                boxes = page.locator("[data-qid]").evaluate_all(
                    """elements => Object.fromEntries(elements.map(element => {
                      const rect = element.getBoundingClientRect();
                      if (rect.width <= 0 || rect.height <= 0) return null;
                      let text = element.innerText || element.value
                        || element.getAttribute('placeholder') || '';
                      text = String(text)
                        .replace(/[\\u0000-\\u001F\\u007F]/g, ' ')
                        .trim();
                      text = Array.from(text).slice(0, 120).join('');
                      text = text.replace(
                        /[\\uD800-\\uDBFF](?![\\uDC00-\\uDFFF])|(?:[^\\uD800-\\uDBFF]|^)[\\uDC00-\\uDFFF]/g,
                        ''
                      );
                      return [element.dataset.qid, {
                        qid: element.dataset.qid,
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height,
                        text
                      }];
                    }).filter(Boolean))"""
                )
                for box in boxes.values():
                    for key in ("x", "y", "width", "height"):
                        box[key] = round(float(box[key]), 1)
                questions = []
                for number, question in enumerate(config["questions"], start=1):
                    target = boxes[question["target_qid"]]
                    bbox = {key: target[key] for key in ("x", "y", "width", "height")}
                    questions.append({
                        "qid_in_template": number,
                        "difficulty": int(question["difficulty"]),
                        "target_qid": question["target_qid"],
                        "target_text_sample": target["text"],
                        "bbox": bbox,
                        "click_xy": [
                            round(bbox["x"] + bbox["width"] / 2, 1),
                            round(bbox["y"] + bbox["height"] / 2, 1),
                        ],
                        "question": question["prompts"][language],
                    })
                record = {
                    "template": template_name,
                    "template_title": config["title"],
                    "lang": language,
                    "image": image_name,
                    "width": width,
                    "height": height,
                    "all_qid_boxes": boxes,
                    "questions": questions,
                }
                (gt_root / f"{template_name}_{language}.json").write_text(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                counts["renders"] += 1
                counts["questions"] += len(questions)
        browser.close()
    return counts


def compare_gt(
    candidate_root: Path,
    reference_root: Path,
    tolerance: float = 0.2,
    compare_geometry: bool = True,
) -> dict[str, Any]:
    checked = 0
    maximum_delta = 0.0
    mismatches = []
    for candidate_path in sorted(candidate_root.glob("*.json")):
        reference_path = reference_root / candidate_path.name
        if not reference_path.is_file():
            mismatches.append({"file": candidate_path.name, "reason": "missing reference"})
            continue
        candidate = json.load(candidate_path.open(encoding="utf-8"))
        reference = json.load(reference_path.open(encoding="utf-8"))
        for field in ("template", "template_title", "lang", "image", "width", "height"):
            if candidate.get(field) != reference.get(field):
                mismatches.append({
                    "file": candidate_path.name,
                    "field": field,
                    "reason": "record metadata",
                })
        candidate_qids = set(candidate["all_qid_boxes"])
        reference_qids = set(reference["all_qid_boxes"])
        for qid in sorted(reference_qids - candidate_qids):
            mismatches.append({"file": candidate_path.name, "qid": qid, "reason": "missing qid"})
        for qid in sorted(candidate_qids - reference_qids):
            mismatches.append({"file": candidate_path.name, "qid": qid, "reason": "extra qid"})
        for qid, candidate_box in candidate["all_qid_boxes"].items():
            reference_box = reference["all_qid_boxes"].get(qid)
            if reference_box is None:
                continue
            if compare_geometry:
                for field in ("x", "y", "width", "height"):
                    delta = abs(float(candidate_box[field]) - float(reference_box[field]))
                    maximum_delta = max(maximum_delta, delta)
                    if delta > tolerance:
                        mismatches.append({
                            "file": candidate_path.name,
                            "qid": qid,
                            "field": field,
                            "delta": delta,
                        })
            if normalize_historical_text(candidate_box["text"]) != normalize_historical_text(
                reference_box["text"]
            ):
                mismatches.append({"file": candidate_path.name, "qid": qid, "reason": "text"})
        if len(candidate["questions"]) != len(reference["questions"]):
            mismatches.append({"file": candidate_path.name, "reason": "question count"})
        else:
            for index, (candidate_question, reference_question) in enumerate(
                zip(candidate["questions"], reference["questions"]), start=1
            ):
                for field in ("qid_in_template", "difficulty", "target_qid", "question"):
                    if candidate_question.get(field) != reference_question.get(field):
                        mismatches.append({
                            "file": candidate_path.name,
                            "question": index,
                            "field": field,
                            "reason": "question metadata",
                        })
                if normalize_historical_text(
                    candidate_question.get("target_text_sample", "")
                ) != normalize_historical_text(reference_question.get("target_text_sample", "")):
                    mismatches.append({
                        "file": candidate_path.name,
                        "question": index,
                        "field": "target_text_sample",
                        "reason": "question text",
                    })
        checked += 1
    reference_files = {path.name for path in reference_root.glob("*.json")}
    candidate_files = {path.name for path in candidate_root.glob("*.json")}
    for filename in sorted(reference_files - candidate_files):
        mismatches.append({"file": filename, "reason": "missing candidate"})
    return {
        "checked": checked,
        "comparison_mode": "strict" if compare_geometry else "structure",
        "maximum_bbox_delta": maximum_delta,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
