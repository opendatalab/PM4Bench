from __future__ import annotations

import json
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..io import resolve_asset, write_jsonl
from .inputs import FONTS, load_manifest, miqa_plan, msocr_plan, sha256


def _render_raster(arguments):
    from .raster import render_miqa, render_msocr

    plan, root, output, font = arguments
    task = plan["task"]
    relative = f"images/{task}/{plan['language']}/{plan['id']}.png"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if task == "miqa":
        render_miqa(plan, root, font, destination)
    else:
        render_msocr(plan, font, destination)
    return {
        "id": plan["id"], "language": plan["language"],
        "image": relative, "sha256": sha256(destination),
    }


def render_vision(
    dataset_root: Path,
    output_root: Path,
    task: str,
    languages: tuple[str, ...],
    *,
    fonts_root: Path | None = None,
    ids: tuple[str, ...] | None = None,
    limit: int | None = None,
    audit_only: bool = False,
    allow_custom_manifest: bool = False,
    strict_glyphs: bool = False,
    workers: int = 1,
) -> dict:
    root = dataset_root.resolve()
    output = output_root.resolve()
    if output == root or root in output.parents:
        raise ValueError("Choose an output directory outside the input dataset snapshot")
    if output.exists():
        raise FileExistsError("Output must be a new directory; existing renders are preserved")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    if workers < 1:
        raise ValueError("--workers must be positive")
    planners = {"miqa": miqa_plan, "msocr": msocr_plan}
    if task not in planners:
        raise ValueError(f"Unsupported synthesis task: {task}")
    manifest_hashes = {}
    plans = []
    found = set()
    for language in languages:
        rows, digest = load_manifest(root, task, language, custom=allow_custom_manifest)
        manifest_hashes[f"data/{task}/{language}.jsonl"] = digest
        selected = [row for row in rows if ids is None or str(row["id"]) in ids]
        if limit:
            selected = selected[:limit]
        for row in selected:
            plans.append(planners[task](row, root))
            found.add(str(row["id"]))
    if ids and set(ids) - found:
        raise ValueError(f"Requested ids were not found: {sorted(set(ids) - found)}")
    if not plans:
        raise ValueError("No records selected")
    font_paths = {}
    font_hashes = {}
    glyph_warnings = {}
    if not audit_only:
        from .raster import missing_glyphs

        if fonts_root is None:
            raise ValueError("MIQA and MSOCR require --fonts-root; see docs/VISION_SYNTHESIS.md")
        for language in languages:
            group = [plan for plan in plans if plan["language"] == language]
            if not group:
                continue
            font = resolve_asset(fonts_root, FONTS[language])
            texts = [
                block["text"] for plan in group
                for block in (plan["lines"] if task == "msocr" else plan["blocks"])
                if "text" in block
            ]
            missing = missing_glyphs(font, texts)
            if missing:
                glyph_warnings[language] = missing
                if strict_glyphs:
                    raise ValueError(f"{language}: {font.name} is missing glyphs: {missing}")
            font_paths[language] = font
            font_hashes[FONTS[language]] = sha256(font)
    output.mkdir(parents=True)
    # Save the complete checked plan, including image routing, for every row.
    write_jsonl(output / "plans.jsonl", plans)
    report = {
        "task": task, "records": len(plans), "audit_only": audit_only,
        "input_manifests_sha256": manifest_hashes, "fonts_sha256": font_hashes,
        "python": platform.python_version(), "custom_manifest": allow_custom_manifest,
        "missing_glyphs": glyph_warnings,
    }
    artifacts = []
    if not audit_only:
        import PIL
        from PIL import features

        report["pillow"] = PIL.__version__
        report["freetype"] = features.version_module("freetype2")
        report["raqm"] = features.version_feature("raqm")
        arguments = [(plan, root, output, font_paths[plan["language"]]) for plan in plans]
        if workers == 1:
            artifacts = list(map(_render_raster, arguments))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                artifacts = list(pool.map(_render_raster, arguments))
        write_jsonl(output / "images.jsonl", artifacts)
    report["images"] = len(artifacts)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
