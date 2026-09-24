# Benchmark vision synthesis

All four benchmark tasks use one command and Python API. They share manifest
loading, record selection, checked plans, output protection, and reporting.
Task-specific composition remains independent: MDUR/MGUI use a browser;
MIQA/MSOCR use Pillow. The published dataset layout is unchanged.

## Setup and input

```bash
python -m pip install -e '.[synthesis]'
python -m playwright install chromium
python -m pm4bench.vision.fonts --output ./fonts
```

Use the benchmark's [v2.0.0 snapshot](https://huggingface.co/datasets/songjhPKU/PM4Bench/tree/v2.0.0).
All tasks read `data/<task>/<language>.jsonl` through `pm4bench.data`.
The manifest lock in `data/resources/inputs.lock.json` covers all 40 files.
Paths are relative to one `--dataset-root`, including MGUI template/GT metadata.
No private source tree or translation service is required.

| Task | Composition and input | Options/details |
| --- | --- | --- |
| MDUR | Public transcript + routed source images, reference-led browser styling | `--fonts-root`; optional `--segoe-root`, `--styles`, `--no-fit`; [details](MDUR_SYNTHESIS.md) |
| MIQA | Ordered source images + public question/text blocks | `--fonts-root`; [raster details](RASTER_SYNTHESIS.md) |
| MSOCR | Public lines at their recorded font sizes | `--fonts-root`; [raster details](RASTER_SYNTHESIS.md) |
| MGUI | Public instructions aligned to templates and reference GT; shared pages rendered once | System font stack; `--comparison-mode`, `--bbox-tolerance`; [details](MGUI_SYNTHESIS.md) |

## One interface

```bash
# The same audit command works with mdur, miqa, msocr, or mgui; no fonts/browser needed.
pm4bench render-vision --task mdur \
  --dataset-root /path/to/PM4Bench --output-root ./audit-mdur --audit-only

# Switch --task for the desired benchmark component.
pm4bench render-vision --task miqa \
  --dataset-root /path/to/PM4Bench --fonts-root ./fonts \
  --language en --limit 2 --output-root ./rendered-miqa
```

For MGUI use `--task mgui` and omit `--fonts-root` to retain its historical
platform-font stack. MDUR supports the same selectors and accepts small pixel
differences. `--browser-executable` applies to the two browser tasks.
`--workers` and `--strict-glyphs` apply to the Pillow tasks. Unsupported
combinations fail explicitly rather than silently ignoring options.

`--language` and `--id` are repeatable. `--limit` limits **records per language**,
after ID selection; omit it for all records. MGUI records are instructions,
so two selected records may share one rendered page. An ID must occur in the
final selection. `--allow-custom-manifest` marks a separate modified-input
experiment; task-specific consistency and image-routing checks still apply.

Python callers use the same orchestrator:

```python
from pathlib import Path
from pm4bench.vision import render_vision

report = render_vision(
    Path("/path/to/PM4Bench"), Path("./audit-mgui"),
    task="mgui", languages=("en",), limit=2, audit_only=True,
)
```

## Common output contract

Each invocation requires a fresh output root outside the input directories:

```text
output/
├── plans.jsonl                  # one checked plan per selected record
├── images.jsonl                 # one artifact reference per rendered record
├── report.json                  # shared run status/counts and backend metadata
├── images/<task>/<language>/...  # deduplicated images
├── html/<task>/<language>/...    # browser tasks only
├── gt/mgui/<language>/...        # MGUI element boxes and page questions
└── styles.jsonl                 # MDUR style selection/replay
```

Every plan/artifact carries `id`, `task`, and `language`. Artifact records add
`image` and `sha256`; MGUI also adds `gt`. Paths in plans refer to the input
snapshot; artifact paths refer to the output. `images.jsonl` is empty in audit
mode. MGUI's artifact rows share image paths where the source page is shared.

Reports consistently expose `task`, `backend`, `records`, `rendered_records`,
`images` (unique image files), `audit_only`, `input_manifests_sha256`,
`custom_manifest`, and `status`. Backends add font/browser or raster versions,
styles, or GT comparisons. A run that fails after planning leaves its evidence
and a `failed` report; it never replaces an old run. MGUI comparison failures
return a nonzero CLI status. Use published benchmark images for evaluation.

## Compatibility and training data

`render-mgui --templates-root ...` and `python -m pm4bench.vision.mdur` remain
compatibility entry points. Their former image layouts/imports are retained;
new workflows should use `render-vision`. The MGUI template-only entry point
cannot provide the new manifest alignment checks without a dataset snapshot.

QGO OCR training synthesis creates new random corpora and remains a separate
workflow under `qgo/`; see [OCR_TRAIN_SYNTHESIS.md](OCR_TRAIN_SYNTHESIS.md).
It is not an additional benchmark task.
