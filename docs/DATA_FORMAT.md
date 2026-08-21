# Data format

All benchmark manifests are UTF-8 JSONL. Every row contains
`schema_version`, `task`, `language`, and a stable task identifier. Asset paths
are POSIX paths relative to the dataset snapshot root.

## MDUR

One row represents the same question in both settings. `traditional_images`
is a list of interleaved source images; `vision_image` is the rendered page.
The row also contains `question`, `options`, `answer_key`, `answer_text`, and
the upstream `source_id`.

## MIQA

One row contains `question`, `reference_answer`, `traditional_images`, and
`vision_image`. `source_image_ids` retains the upstream MMDU identifiers for
provenance.

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
