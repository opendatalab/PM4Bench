# Architecture

PM4Bench v2 separates four ownership layers:

1. **GitHub toolkit** — schemas, validators, deterministic metrics, optional
   judge integration, construction code, and the QGO reward and training
   recipe.
2. **Benchmark dataset** — immutable JSONL manifests plus deduplicated assets.
3. **QGO training dataset** — portable Parquet shards with embedded images.
4. **QGO model** — a standalone BF16 Transformers checkpoint.

The static [project page](https://little-bird-vodka.github.io/PM4Bench/) is a
presentation layer over these canonical resources and the public arXiv record;
it does not duplicate released data, weights, or executable code.

No layer contains machine-specific paths, credentials, experiment logs, or
paper-result outputs. Dataset rows use stable identifiers and paths relative to
the dataset snapshot root.

## Data and construction modules

```text
src/pm4bench/
├── data/                   # four-task manifests, selection, integrity checks
│   └── resources/          # one lock for all 40 language/task manifests
├── vision/
│   ├── render.py           # shared plan → render → report orchestration
│   ├── options.py          # shared render options
│   ├── browser.py          # offline browser/font utilities
│   ├── raster.py           # font, glyph, and text-raster utilities
│   ├── resources/          # pinned fonts, backgrounds, MDUR routing/styles
│   └── tasks/              # task-specific plans and layout implementations
│       ├── mdur.py
│       ├── miqa.py
│       ├── msocr.py
│       ├── mgui.py
│       ├── _mdur_text.py
│       └── _mgui_browser.py
├── qgo/                    # training-corpus synthesis, packing, and reward
└── mgui/render.py          # compatibility facade; no separate implementation
```

`pm4bench.data` owns task/language constants, locked manifest loading, and
record selection, independently of rendering. `pm4bench.vision.render_vision`
and `pm4bench render-vision` expose the same pipeline for all four tasks.
The task registry chooses a planner and backend; plans preserve published
text and asset routing. One orchestrator owns output protection, common
manifests, and execution reports. Browser and raster implementations remain
separate because their layout and font behavior are different.

Every output contains `plans.jsonl`, `images.jsonl`, and `report.json`; images
use `images/{task}/{language}/`. Each image-manifest row corresponds to one
selected benchmark record. MGUI deduplicates pages shared by multiple questions
and checks templates/GT against the selected public records before rendering.
Audit mode checks all tasks without optional rendering dependencies.

Optional font downloads are pinned by source revision and checksum.
Construction does not sample replacement benchmark text or call translation
services. The raster stage accounts for actual ink extents to avoid clipping
the final line. See [Vision synthesis](docs/VISION_SYNTHESIS.md) for the single
user-facing construction guide and links to task-specific details.

MDUR parses the public transcript and uses index-only source-image routing.
Its offline browser renderer separates plans from reference-led style fitting;
saved styles allow replay. Segoe/Noto is the closest tested candidate, with
Noto-only rendering also available. Pixel identity is not a release requirement.

## Training and release boundaries

QGO synthesis is separate from benchmark construction: it samples new training
text instead of consuming benchmark records. `qgo.synthesize` samples the
bundled text pool into explicit plans, renders
words, and derives targets from visible boxes. `qgo.pack` validates image/GT
pairs and writes portable Parquet with saved split membership/order. New
corpora never modify the canonical training data. Shared browser utilities
handle font loading, offline operation, and output protection.

The QGO launcher keeps model, data, output, and reward locations external while
fixing the public training parameters. Its reward module implements the same
accuracy, length, and repetition terms used for QGO-8B.

The release pipeline builds immutable candidate generations on H1. A candidate
is activated only after count, schema, path, secret, checksum, and smoke-load
validation. Uploads originate from H1; the source experiment tree remains
read-only.
