"""One construction pipeline for every benchmark task."""
from __future__ import annotations

import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..data import LANGUAGES, select_records
from ..data.benchmark import sha256
from ..io import resolve_asset, write_jsonl
from .browser import new_output, save_json
from .options import RenderOptions
from .resources import FONTS
from .tasks import get_task


def _render_raster(arguments):
    plan, root, output, font = arguments
    task = plan['task']
    relative = f"images/{task}/{plan['language']}/{plan['id']}.png"
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    _, module, _ = get_task(task)
    if task == 'miqa':
        module.render_miqa(plan, root, font, destination)
    else:
        module.render_msocr(plan, font, destination)
    return {'id': plan['id'], 'task': task, 'language': plan['language'],
            'image': relative, 'sha256': sha256(destination)}


def _raster_batch(plans, root, output, options):
    import PIL
    from PIL import features

    from .raster import missing_glyphs

    paths, hashes, warnings = {}, {}, {}
    for language in sorted({p['language'] for p in plans}):
        group = [p for p in plans if p['language'] == language]
        font = resolve_asset(options.fonts_root, FONTS[language])
        texts = [block['text'] for p in group
                 for block in (p['lines'] if p['task'] == 'msocr' else p['blocks'])
                 if 'text' in block]
        missing = missing_glyphs(font, texts)
        if missing:
            warnings[language] = missing
            if options.strict_glyphs:
                raise ValueError(f'{language}: {font.name} is missing glyphs: {missing}')
        paths[language], hashes[FONTS[language]] = font, sha256(font)
    arguments = [(p, root, output, paths[p['language']]) for p in plans]
    if options.workers == 1:
        artifacts = list(map(_render_raster, arguments))
    else:
        with ProcessPoolExecutor(max_workers=options.workers) as pool:
            artifacts = list(pool.map(_render_raster, arguments))
    return artifacts, {'fonts_sha256': hashes, 'missing_glyphs': warnings,
                       'pillow': PIL.__version__, 'freetype': features.version_module('freetype2'),
                       'raqm': features.version_feature('raqm')}


def _check_options(task, options, audit_only):
    if options.workers < 1:
        raise ValueError('--workers must be positive')
    if options.comparison_mode not in {'strict', 'structure'} or options.bbox_tolerance < 0:
        raise ValueError('Invalid MGUI comparison settings')
    if task != 'mgui' and (options.comparison_mode != 'structure' or options.bbox_tolerance != .2):
        raise ValueError('--comparison-mode and --bbox-tolerance apply only to MGUI')
    if task in {'mgui', 'mdur'} and (options.workers != 1 or options.strict_glyphs):
        raise ValueError('Browser tasks require --workers 1 and do not support --strict-glyphs')
    if task != 'mdur' and (options.styles or options.segoe_root or not options.fit):
        raise ValueError('--styles, --segoe-root and --no-fit apply only to MDUR')
    if task in {'miqa', 'msocr'} and options.browser_executable:
        raise ValueError('--browser-executable applies only to MDUR/MGUI')
    if task == 'mgui' and options.fonts_root:
        raise ValueError('MGUI preserves its platform font stack; omit --fonts-root')
    if not audit_only and task != 'mgui' and options.fonts_root is None:
        raise ValueError(f'{task} requires --fonts-root; see docs/VISION_SYNTHESIS.md')


def render_vision(dataset_root: Path, output_root: Path, task: str,
                  languages: tuple[str, ...] = LANGUAGES, *,
                  fonts_root: Path | None = None, ids: tuple[str, ...] | None = None,
                  limit: int | None = None, audit_only: bool = False,
                  allow_custom_manifest: bool = False, strict_glyphs: bool = False,
                  workers: int = 1, browser_executable: Path | None = None,
                  segoe_root: Path | None = None, fit: bool = True,
                  styles: Path | None = None, comparison_mode: str = 'structure',
                  bbox_tolerance: float = .2, _legacy_layout: bool = False) -> dict:
    spec, module, planner = get_task(task)
    options = RenderOptions(fonts_root, browser_executable, segoe_root, fit, styles,
                            strict_glyphs, workers, comparison_mode, bbox_tolerance,
                            _legacy_layout)
    root, output = dataset_root.resolve(), output_root.resolve()
    # Fail before reading the corpus if the destination cannot be used.
    protected = [root] + [p.resolve() for p in (fonts_root, segoe_root) if p]
    if any(output == p or p in output.parents for p in protected):
        raise ValueError('Choose an output directory outside input directories')
    if output.exists():
        raise FileExistsError('Output must be a new directory; existing renders are preserved')
    _check_options(task, options, audit_only)
    rows, hashes = select_records(root, task, languages, ids=ids, limit=limit,
                                  custom=allow_custom_manifest)
    plans = [planner(row, root) for row in rows]
    output = new_output(output, *protected)
    write_jsonl(output / 'plans.jsonl', plans)
    report = {'schema_version': '1.0', 'task': task, 'backend': spec.backend,
              'records': len(plans), 'audit_only': audit_only,
              'input_manifests_sha256': hashes, 'custom_manifest': allow_custom_manifest,
              'python': platform.python_version(), 'fonts_sha256': {},
              'rendered_records': 0, 'images': 0, 'status': 'running'}
    save_json(output / 'report.json', report)
    try:
        artifacts = []
        if not audit_only:
            runner = _raster_batch if spec.backend == 'raster' else module.render_batch
            artifacts, metadata = runner(plans, root, output, options)
            report.update(metadata)
        write_jsonl(output / 'images.jsonl', artifacts)
        report.update(images=len({a['image'] for a in artifacts}),
                      rendered_records=len(artifacts),
                      status='failed' if report.get('passed') is False else 'complete')
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        save_json(output / 'report.json', report)
        raise
    save_json(output / 'report.json', report)
    return report
