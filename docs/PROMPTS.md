# Task prompts and response formats

The deterministic metrics in `pm4bench.metrics` parse specific markers out of a
model response. This page is the authoritative specification of those markers,
of the prompts that produce them, and of how to combine a prompt with a
benchmark row. Without a matching prompt the metrics do not fail — they
silently score low.

Ten prompts are released, each in all ten languages — the ones used to produce
the published results. They ship as one JSON resource,
`src/pm4bench/prompts/task_prompts.json`.

## Reading the resource

```jsonc
{
  "schema_version": 2,
  "languages": ["ar", "cs", "en", "hu", "ko", "ru", "sr", "th", "vi", "zh"],
  "prompts": {
    "msocr-ocr-vision": {
      "task": "msocr",                      // mdur | miqa | msocr | mgui
      "probe": "ocr",                       // ocr transcribes, vqa answers
      "setting": "vision",                  // how the input is presented
      "composition": "standalone",          // how to turn this into a query
      "answer_markers": ["<start>", "<end>"],   // literals the prompt demands
      "scored_by": "evaluate --task msocr", // the command that scores it
      "text": { "ar": "...", "en": "...", ... }  // the prompt, ten languages
    }
  }
}
```

Only `prefix` prompts carry an extra `assembly` object; nothing else is
optional. If you just want a prompt string, read `text`. If you are writing an
evaluation harness, read `composition` first.

## Vocabulary

- **probe** — `ocr` transcribes the text in the image; `vqa` answers the
  question.
- **setting** — `traditional` supplies the question text and the source images
  separately. `vision` renders everything into one image.
  `vision_reference_text` is the `vision` image plus the reference question and
  options as text. `screenshot` is MGUI's single GUI capture.

## Released prompts

| `prompt_id` | probe | setting | `answer_markers` | parsed by |
| --- | --- | --- | --- | --- |
| `mdur-ocr-vision` | ocr | vision | `"question"` `"options"` | `metrics.ocr_contains` |
| `mdur-vqa-vision` | vqa | vision | `<X>` | `metrics.mdur_correct` |
| `mdur-vqa-traditional` | vqa | traditional | `<X>` | `metrics.mdur_correct` |
| `mdur-vqa-vision-reference-text` | vqa | vision_reference_text | `<X>` | `metrics.mdur_correct` |
| `miqa-ocr-vision` | ocr | vision | `<start>` `<end>` | `metrics.miqa_ocr_correct` |
| `miqa-vqa-vision` | vqa | vision | `<Start of Answer>` `<End of Answer>` | the MIQA judge |
| `miqa-vqa-traditional` | vqa | traditional | `<Start of Answer>` `<End of Answer>` | the MIQA judge |
| `msocr-ocr-vision` | ocr | vision | `<start>` `<end>` | `metrics.msocr_score` |
| `mgui-vqa-screenshot` | vqa | screenshot | `<Start of Answer>` `<End of Answer>` `[x, y]` | `metrics.extract_coords` + `point_in_bbox` |
| `mgui-ocr-screenshot` | ocr | screenshot | `<Start of Answer>` `<End of Answer>` | `metrics.mgui_text_correct` |

A response that satisfies each marker set looks like:

| markers | example response |
| --- | --- |
| `<X>` | `Weighing the options...` / newline / `<C>` |
| `<start>` `<end>` | `<start>Spaniards Marrakesh<end>` |
| `<Start of Answer>` `<End of Answer>` | `<Start of Answer>One way<End of Answer>` |
| ... plus `[x, y]` | `<Start of Answer>[318, 312]<End of Answer>` |
| `"question"` `"options"` | `{"question": "Which is not a language?", "options": ["HTML"]}` |

`mdur-vqa-vision-reference-text` is a control condition: it supplies the vision
image together with the reference question and options as text, so it measures
the upper bound a model would reach if it read the rendered page perfectly. The
benchmark has no separate manifest for it — use the `mdur` rows.

## Building a query

A prompt on its own is not a query. Except for the five `standalone` prompts,
it has to be combined with a row from the benchmark manifest. `build_query`
does this for all three composition kinds:

```python
from pm4bench.data import load_manifest
from pm4bench.task_prompts import build_query

records, _digest = load_manifest("/path/to/PM4Bench-snapshot", "mdur", "en")
query = build_query("mdur-vqa-traditional", records[0])
```

The returned string is the full text input. Images are **not** embedded: attach
`traditional_images` (or `vision_image` / `image`) separately in your client,
in the order the manifest lists them. `<image>` marks where each one belongs.

| `composition` | prompts | rule |
| --- | --- | --- |
| `standalone` | 5 | use `text` verbatim; the question is inside the image |
| `format_template` | 2 (MGUI) | substitute the manifest `question` into `{question}` |
| `prefix` | 3 | append a body built per the entry's `assembly` object |

**Do not call `.format()` on every prompt.** `mdur-ocr-vision` asks for a JSON
object, so its text contains literal `{` and `}` in all ten languages and
`str.format` raises `KeyError`. Only `format_template` prompts are templates.

### The `assembly` object

The three `prefix` prompts carry their rule as data, so you can reimplement it
in another language. `template` shows the order; the remaining fields supply
the pieces.

Both MDUR prompts use
`{prompt}{separator}{stem}\n\n{options}\n\n{answer_prefix}`:

- `stem` — `rendered_text` between `"{question_label}: "` and
  `"\n{options_label}:\n"`, then `<image N>` rewritten to `<image>` per
  `image_token`. Both label maps cover all ten languages.
- `options` — the option block that follows `"{options_label}:\n"` in
  `rendered_text`, each line relabeled from `(A)  text` to `option_line`. Read
  it from `rendered_text`, **not** the `options` array: the array stores
  image-only options as empty strings.
- `answer_prefix` — the localized closing line, e.g. `Your answer:` /
  `你的回答：`.
- `separator` — a space for `mdur-vqa-traditional`, empty for
  `mdur-vqa-vision-reference-text`.

`miqa-vqa-traditional` uses `{prompt}{body}`: append `" <image>"` to every line
of `rendered_text` that ends in a colon, one per entry in
`traditional_images`. Its prompt text already ends with the localized
`Question: `, so `separator` is empty.

## MGUI coordinate space

`mgui-vqa-screenshot` asks for **pixel** coordinates, and
`evaluate --task mgui` requires `--coordinate-space` explicitly — it has no
default, because the wrong value silently rescales every prediction.

```bash
pm4bench evaluate --task mgui --manifest data/mgui/en.jsonl \
  --predictions predictions.jsonl --coordinate-space pixel
```

Choose the value your model actually emits. Several families — Qwen, GLM,
Gemini, Doubao, Llama — return coordinates on a 1000x1000 grid whatever the
instruction asks for; use `--coordinate-space normalized-1000` for those. GUI
screenshots are 1280x800, which is the resolution normalized coordinates are
scaled back to.

## Metric behavior worth knowing

- `mdur_correct` delegates to `extract_answer_letter`, which has no explicit
  `<([A-J])>` branch. `<X>` is parsed by the trailing bare-letter fallback,
  which takes the **last** match in the response — a response that mentions
  another option letter after its final answer is scored wrong silently.
- `evaluate --task mdur-ocr` scores with `ocr_contains(response, question)`.
  The JSON structure the prompt requests is advisory: malformed JSON is not
  penalized and the `options` array is never checked.
- `evaluate --task miqa-ocr` compares against the instruction substring of
  `rendered_text` and requires a prediction longer than ten characters. It does
  not strip `<start>`/`<end>`; only `msocr_score` does. Both test containment,
  so the wrapper is harmless either way.
- `evaluate --task mgui-text` skips targets whose text is empty after
  decorative-symbol removal and reports them as `skipped`.
