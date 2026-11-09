# Task prompts and response formats

The deterministic metrics in `pm4bench.metrics` parse specific markers out of a
model response. This page is the authoritative specification of those markers
and of the prompts that produce them. Without a matching prompt the metrics do
not fail — they silently score low.

Ten prompts are released, each in all ten languages, exactly as used for the
reported results. They ship as a package resource and are read with
`pm4bench prompts`:

```bash
pm4bench prompts                                             # index, no prompt text
pm4bench prompts --task mdur --probe vqa                     # filter
pm4bench prompts --id mdur-vqa-vision                        # one prompt, ten languages
pm4bench prompts --id mdur-vqa-vision --language en --raw    # the bare string
```

`pm4bench.task_prompts` exposes the same data to Python callers through
`prompt_text(prompt_id, language)`, `prompt_entry`, `response_format` and
`describe_prompts`.

## Vocabulary

- **probe** — what the model is asked to do: `ocr` transcribes the text in the
  image; `vqa` answers the question.
- **setting** — how the input is presented. `traditional` supplies the question
  text and the source images separately. `vision` renders everything into one
  image. `vision_reference_text` is the `vision` image plus the reference
  question and options as text. `screenshot` is MGUI's single GUI capture.

## Released prompts

| `prompt_id` | probe | setting | response format | scored by |
| --- | --- | --- | --- | --- |
| `mdur-ocr-vision` | ocr | vision | `mdur_ocr_json` | `evaluate --task mdur-ocr` |
| `mdur-vqa-vision` | vqa | vision | `option_letter_in_angles` | `evaluate --task mdur` |
| `mdur-vqa-traditional` | vqa | traditional | `option_letter_in_angles` | `evaluate --task mdur` |
| `mdur-vqa-vision-reference-text` | vqa | vision_reference_text | `option_letter_in_angles` | `evaluate --task mdur` |
| `miqa-ocr-vision` | ocr | vision | `lowercase_start_end` | `evaluate --task miqa-ocr` |
| `miqa-vqa-vision` | vqa | vision | `answer_tag` | `judge-miqa` |
| `miqa-vqa-traditional` | vqa | traditional | `answer_tag` | `judge-miqa` |
| `msocr-ocr-vision` | ocr | vision | `lowercase_start_end` | `evaluate --task msocr` |
| `mgui-vqa-screenshot` | vqa | screenshot | `answer_tag_pixel_xy` | `evaluate --task mgui` |
| `mgui-ocr-screenshot` | ocr | screenshot | `answer_tag` | `evaluate --task mgui-text` |

`mdur-vqa-vision-reference-text` is a control condition, not a pipeline result.
It supplies the vision image together with the reference question and options,
isolating the OCR bottleneck: it is the upper bound a model would reach if it
read the rendered page perfectly. It does not consume OCR output, and the
released benchmark has no separate manifest for it.

## Response markers

Every marker below is an ASCII literal and appears verbatim in all ten
languages. The parsing functions are the authority; these are their inputs.

| response format | markers | parsed by |
| --- | --- | --- |
| `option_letter_in_angles` | `<X>` | `metrics.extract_answer_letter` |
| `lowercase_start_end` | `<start>` `<end>` | `metrics.msocr_score` |
| `answer_tag` | `<Start of Answer>` `<End of Answer>` | `metrics.extract_answer_text` |
| `answer_tag_pixel_xy` | `<Start of Answer>` `<End of Answer>` `[x, y]` | `metrics.extract_coords` |
| `mdur_ocr_json` | `"question"` `"options"` | `metrics.ocr_contains` |

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

`composition` records which rule applies, and getting it wrong is the most
likely cause of a silently wrong score:

| composition | prompts | rule |
| --- | --- | --- |
| `standalone` | 5 | use the string verbatim; the question is inside the image |
| `format_template` | 2 (MGUI) | substitute the manifest `question` into `{question}` |
| `prefix` | 3 | concatenate the body built per the entry's `assembly` block |

**Do not call `.format()` on every prompt.** `mdur-ocr-vision` asks for a JSON
object, so its text contains literal `{` and `}` in all ten languages and
`str.format` raises `KeyError`. Only `format_template` prompts are templates.

### The `assembly` block

The three `prefix` prompts carry a machine-readable `assembly` rule, so you can
reimplement it in another language if needed. Both MDUR prompts use:

```
prompt + separator + stem + "\n\n" + options + "\n\n" + answer_prefix[language]
```

- `stem` is `rendered_text` between `"<question_label>: "` and
  `"\n<options_label>:\n"`, with `<image N>` rewritten to `<image>`. The
  localized label pairs are in the block.
- `options` come from the option block of `rendered_text`, relabeled from
  `(A)  text` to `(A). text`. **Not from the `options` array** — image-only
  options appear as markers in `rendered_text` but as empty strings in the
  array, so the array silently loses them.
- `answer_prefix` is the localized closing line, e.g. `Your answer:` /
  `你的回答：`.
- `separator` is a space for `mdur-vqa-traditional` and empty for
  `mdur-vqa-vision-reference-text`.

`miqa-vqa-traditional` is simpler: append `" <image>"` to every line of
`rendered_text` that ends in a colon, then concatenate onto the prompt, whose
text already ends with the localized `Question: `.

### What is and is not claimed

The query `build_query` returns is consistent with the released images and
manifests, which is what you need to evaluate the published benchmark. It is
not claimed byte-identical to the inputs used for the paper's own runs: those
were assembled by historical scripts, and whitespace normalization plus
translation-revision differences affect a minority of non-English rows. Scores
computed this way are self-consistent and comparable across models; treat small
deviations from the published tables as expected.

The five `standalone` prompts and the two MGUI templates carry no such caveat —
each matched its corresponding frozen query exactly across all ten languages.

## MGUI coordinate space

`mgui-vqa-screenshot` asks for **pixel** coordinates. `evaluate --task mgui`
therefore requires `--coordinate-space` to be stated explicitly; it has no
default, because a default that disagrees with the released prompt would
silently rescale predictions.

```bash
pm4bench evaluate --task mgui --manifest data/mgui/en.jsonl \
  --predictions predictions.jsonl --coordinate-space pixel
```

The paper-era evaluation did not use one global setting. Model families that
emit coordinates on a 1000x1000 normalized grid regardless of the instruction
were rescaled per family — `qwen`, `glm`, `gemini`, `doubao` and `llama` were
mapped to `(1000, 1000)`; `gpt` was deliberately left unmapped pending a
reviewed rerun; every other model was read as raw pixels. GUI screenshots are
1280x800. Reproducing a published number requires matching that per-model
choice, which one flag value cannot express.

## Parsing hazards

These are properties of the current metrics, documented so they are not
mistaken for bugs introduced by a caller.

- `extract_answer_letter` has no explicit `<([A-J])>` branch. `<X>` is parsed
  only by the trailing bare-letter fallback, which takes the **last** match in
  the response. A response that mentions another option letter after its final
  answer is scored wrong without any warning. Tracked in
  [ROADMAP.md](../ROADMAP.md).
- `evaluate --task mdur-ocr` scores with `ocr_contains(response, question)`.
  The JSON structure the prompt requests is advisory: malformed JSON is not
  penalized and the `options` array is never checked.
- `evaluate --task miqa-ocr` compares against the instruction substring of
  `rendered_text` and requires a prediction longer than ten characters. It does
  not strip `<start>`/`<end>`; only `msocr_score` does. Both metrics test
  containment, so the wrapper is harmless either way.
- `evaluate --task mgui-text` skips targets whose text is empty after
  decorative-symbol removal and reports them as `skipped`.

## Verification

Each entry records the match kind against the frozen paper-era inference inputs
and the number of records checked. Totals are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).
