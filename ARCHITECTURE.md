# Architecture

PM4Bench v2 separates four ownership layers:

1. **GitHub toolkit** — schemas, validators, deterministic metrics, optional
   judge integration, construction code, and QGO recipes.
2. **Benchmark dataset** — immutable JSONL manifests plus deduplicated assets.
3. **QGO training dataset** — portable Parquet shards with embedded images.
4. **QGO model** — a standalone BF16 Transformers checkpoint.

No layer contains machine-specific paths, credentials, experiment logs, or
paper-result outputs. Dataset rows use stable identifiers and paths relative to
the dataset snapshot root.

The release pipeline builds immutable candidate generations on H1. A candidate
is activated only after count, schema, path, secret, checksum, and smoke-load
validation. Uploads originate from H1; the source experiment tree remains
read-only.
