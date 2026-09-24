# Data format

All benchmark manifests are UTF-8 JSONL. Every row contains
`schema_version`, `task`, `language`, and a stable task identifier. Asset paths
are POSIX paths relative to the dataset snapshot root.

`pm4bench.data.load_manifest(root, task, language)` and `select_records(...)`
are the shared data access layer for all four tasks. Published manifest hashes
are checked by default. The [vision pipeline](VISION_SYNTHESIS.md) consumes
these records with common selectors and outputs; task-specific fields below
remain intact. This API unification does not migrate or rewrite the HF files.

## MDUR

One row represents the same question in both settings. `traditional_images`
is a list of interleaved source images; `vision_image` is the rendered page.
The row also contains `question`, `options`, `answer_key`, `answer_text`, and
the upstream `source_id`.

## MIQA

One row contains `question`, `reference_answer`, `traditional_images`, and
`vision_image`. `source_image_ids` retains the upstream MMDU identifiers for
provenance.

Read `traditional_images` in numeric `image_N.png` order. The v2.0.0 list for
`105_0` is lexicographically sorted, so `image_10.png` and `image_11.png` appear
before `image_2.png`. The vision compositor uses numeric order and retains the
source labels from `rendered_text`; these labels can be non-consecutive.

## MSOCR

One row contains a single `image`, ordered `lines` with `font_size`, and a
space-joined `ground_truth`. The 0-40 metric normalizes punctuation and markup,
then uses the font size at the first character-level recognition boundary.

## MGUI

One row represents one instruction. Two rows can reference the same screenshot.
Fields include `question`, `target_qid`, `target_text`, `bbox`, and `click_xy`.
The full element map and Jinja sources live under `metadata/mgui/` and are not
duplicated into every row.

`mgui` scores normalized or pixel click coordinates against `bbox`.
`mgui-text` is the companion identification diagnostic: it compares the
normalized visible response with `target_text` and skips targets containing no
text after decorative-symbol removal.

## Prediction format

Predictions are JSONL with `id` or `index` plus `response`. Additional fields
such as `reasoning`, latency, and model metadata are ignored by deterministic
metrics and may be retained by callers.
