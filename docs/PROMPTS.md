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

## Composition

`composition` says how to turn a released string into a query. Getting this
wrong is the most likely cause of a wrong score.

- `standalone` — use the string verbatim. The image is attached separately.
- `format_template` — the string contains `{question}`; substitute the
  manifest's `question` field. Both MGUI prompts are templates.
- `prefix` — the string is only the leading instruction; the paper-era query
  appended more text. See the reproducibility limit below.

**Do not call `.format()` on every prompt.** `mdur-ocr-vision` contains a
literal JSON schema, so its text has `{` and `}` in all ten languages and
`str.format` raises `KeyError`. Only `format_template` prompts are templates.

## Reproducibility limit on the three `prefix` prompts

For `mdur-vqa-traditional`, `mdur-vqa-vision-reference-text` and
`miqa-vqa-traditional`, the released string is the instruction prefix only. The
text that followed it was assembled by historical build scripts from fields
that the published benchmark does not preserve, so **the exact paper-era query
cannot be reconstructed from the released data.** Measured against the frozen
inference inputs, rebuilding the suffix from released manifest fields reproduces

| `prompt_id` | reconstructed | total |
| --- | --- | --- |
| `mdur-vqa-traditional` | 8,477 | 17,300 |
| `mdur-vqa-vision-reference-text` | 8,477 | 17,300 |
| `miqa-vqa-traditional` | 1,917 | 2,180 |

Two causes are confirmed: released `options` arrays omit the option-level
`<image>` markers that the paper-era query carried, and whitespace adjacent to
image markers is not preserved. No assembly rule is published, because a rule
that is right for half the corpus would mislead rather than help. The
per-prompt `reproducibility` block in the resource records this machine-readably.

The seven other prompts are byte-exact or template-exact: each one matched
every corresponding frozen query across all ten languages.

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
