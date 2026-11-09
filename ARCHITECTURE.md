# Architecture

PM4Bench is released as four independent artifacts:

1. **Toolkit** (this repository) — schemas, validators, deterministic metrics,
   the MIQA judge adapter, benchmark image construction, and the QGO reward and
   training recipe.
2. **Benchmark dataset** — immutable JSONL manifests plus deduplicated assets.
3. **QGO training dataset** — portable Parquet shards with embedded images.
4. **QGO model** — a standalone BF16 Transformers checkpoint.

Dataset rows use stable identifiers and paths relative to the snapshot root. No
artifact contains machine-specific paths, credentials or experiment logs.

## Repository layout

```text
src/pm4bench/
├── cli.py                  # every `pm4bench` subcommand
├── data/                   # manifest loading, record selection, integrity
│   ├── benchmark.py
│   └── inputs.lock.json    # one lock for all 40 language/task manifests
├── metrics.py              # the deterministic scoring functions
├── miqa.py                 # MIQA LLM-as-judge client and aggregation
├── io.py                   # JSONL, prediction loading, asset resolution
├── task_prompts.py         # released prompts and query assembly
├── prompts/                # the prompt resource and the MIQA judge rubric
├── vision/                 # benchmark image construction
│   ├── render.py           # shared plan → render → report orchestration
│   ├── options.py          # render options shared by the backends
│   ├── browser.py          # offline browser and font loading
│   ├── raster.py           # font, glyph and text-raster utilities
│   ├── assets.py           # resource root and per-language font files
│   ├── fonts.py            # pinned font download
│   ├── resources/          # fonts lock, backgrounds, MDUR routing, CSS
│   └── tasks/              # one planner and layout per task
│       ├── mdur.py         #   browser backend
│       ├── mdur_text.py    #   transcript parser
│       ├── mgui.py         #   manifest-driven planner
│       ├── mgui_browser.py #   template renderer and GT comparator
│       ├── miqa.py         #   raster backend
│       └── msocr.py        #   raster backend
└── qgo/                    # training-corpus synthesis, packing and reward
recipes/                    # the GRPO launcher and its parameter record
tests/                      # unit, fixture and browser-replay tests
docs/                       # task and format specifications
```

## Evaluation

`pm4bench.data` owns task and language constants, locked manifest loading and
record selection, independently of rendering. `metrics.py` holds one function
per score; `cli.py` dispatches `evaluate --task` to them. MIQA is the only task
that needs a model: `miqa.py` drives a resumable OpenAI-compatible judge, and
the endpoint and credentials always come from the caller.

The toolkit ships no inference code. `task_prompts.py` publishes the prompts
the metrics expect and `build_query` combines a prompt with a manifest row into
the full text input; producing predictions is left to the caller. See
[docs/PROMPTS.md](docs/PROMPTS.md).

## Construction

`vision.render_vision` and `pm4bench render-vision` expose one pipeline for all
four tasks. A task registry chooses the planner and backend; plans preserve
published text and asset routing. One orchestrator owns output protection,
common manifests and execution reports. Browser and raster backends stay
separate because their layout and font behavior differ.

Every output contains `plans.jsonl`, `images.jsonl` and `report.json`, with
images under `images/{task}/{language}/`. Each image-manifest row corresponds to
one selected benchmark record. MGUI deduplicates pages shared by several
questions and checks templates and GT against the selected records before
rendering. Audit mode checks all tasks without the optional rendering
dependencies. `render-mgui` renders from a template directory alone, which is
the one path that cannot perform manifest alignment.

Font downloads are pinned by source revision and checksum. Construction never
samples replacement benchmark text or calls translation services. The raster
stage accounts for actual ink extents so the final line is not clipped. MDUR
parses the public transcript, routes source images by index, and separates
plans from reference-led style fitting so saved styles can be replayed; Segoe
with Noto fallback is the closest tested profile and Noto-only also works.
Generated images are not pixel-identical to the released ones — evaluate
against the published images. See
[docs/VISION_SYNTHESIS.md](docs/VISION_SYNTHESIS.md).

## Training

QGO synthesis is separate from benchmark construction: it samples new training
text instead of consuming benchmark records. `qgo.synthesize` turns the bundled
text pool into explicit plans, renders words and derives targets from visible
boxes. `qgo.pack` validates image/GT pairs and writes portable Parquet with
saved split membership and order. Generated corpora never modify the released
training data.

The launcher in `recipes/` keeps model, data, output and reward locations
external while fixing the published training parameters. `qgo.reward`
implements the accuracy, length and repetition terms used for QGO-8B.
