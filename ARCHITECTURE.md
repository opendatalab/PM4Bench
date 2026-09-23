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

The `vision` module separates checked text/image plans from raster rendering.
MIQA and MSOCR consume the benchmark JSONL files directly. A manifest lock
identifies the released text; synthesis writes plans, generated images, and
environment/checksum reports to a separate output directory. Optional font
downloads are pinned by source revision and checksum. Construction code does
not sample replacement benchmark text or call translation services.

The QGO launcher keeps model, data, output, and reward locations external while
fixing the public training parameters. Its reward module implements the same
accuracy, length, and repetition terms used for QGO-8B.

The release pipeline builds immutable candidate generations on H1. A candidate
is activated only after count, schema, path, secret, checksum, and smoke-load
validation. Uploads originate from H1; the source experiment tree remains
read-only.
