# MDUR vision synthesis

Use the shared [vision synthesis guide](VISION_SYNTHESIS.md) for installation,
selection, and output conventions. This page covers MDUR-specific rendering.

MDUR construction uses the recovered HTML layout and the closest tested
browser/font configuration, with reference-led style selection. Small font
and raster differences are accepted; pixel identity is not claimed. Use the
published images for comparable evaluation.

## Inputs and alignment

Use the benchmark's [v2.0.0 snapshot](https://huggingface.co/datasets/songjhPKU/PM4Bench/tree/v2.0.0),
including `data/mdur/` and `assets/mdur/{traditional,vision}/`. Content comes
from public `rendered_text`, without resampling or translation. The parser
checks the historical angle-bracket-cleanup relationship to traditional
fields, rather than using truncated traditional text to recreate vision.
Index-only image routing includes question/option images in their recorded
positions and excludes answer-explanation images. Public data remains unchanged.

```bash
pm4bench render-vision --task mdur \
  --dataset-root /path/to/PM4Bench --output-root ./mdur-audit --audit-only
```

Audit mode checks locked text manifests and source-image routing without a
browser or fonts. All ten language manifests are supported.

## Render

The closest tested English profile uses Segoe UI Light/Regular/Bold with Noto
fallback, Chromium 130.0.6723.31 (Playwright 1.48.0), disabled font hinting,
preserved label spaces, and ceiling-rounded screenshot height. Segoe is an
empirically closer candidate, not an identified historical font environment.

Segoe files are **not bundled** or relicensed. If you have appropriate access,
provide `segoeui-light.woff2`, `segoeui-regular.woff2`, `segoeui-bold.woff2` in
a separate directory. Tested files were served from Microsoft's
[font service](https://static2.sharepointonline.com/files/fabric/assets/fonts/segoeui-westeuropean/segoeui-light.woff2)
(replace `light` with `regular` or `bold`). See
[Microsoft's font information](https://learn.microsoft.com/en-us/typography/font-list/segoe-ui)
for terms. Tested hashes are in `vision/resources/reference-fonts.json`.

```bash
pm4bench render-vision --task mdur \
  --dataset-root /path/to/PM4Bench --fonts-root ./fonts \
  --segoe-root ./segoe-fonts --language en --limit 3 \
  --output-root ./mdur-rendered
```

Omit `--segoe-root` for the fully open Noto-only profile, whose English glyph
geometry differs more in our tests. Both profiles report font hashes. Repeat
`--language` or `--id` to select records; omit these and `--limit` for all
17,300 records. `--browser-executable` overrides the pinned browser.

## Style selection and replay

The legacy style table does not reliably describe the released images. For
each row, the recipe compares nine backgrounds against reference border pixels,
then performs a bounded search over size, weight, slant, shadow, and decoration
using actual browser renders. It selects the lowest measured discrepancy among
tested candidates, not a global optimum or recovered historical parameters.
No reference pixels are copied into the generated image. Start with a small
selection: fitting costs more than rendering a known style.

Replay saved `styles.jsonl` with `--styles ./mdur-rendered/styles.jsonl`, the
same record selectors, and a new output directory. The style file must cover
exactly the requested records. `--no-fit` uses the calibrated base style and
fitted background without text-style search. All modes read reference images
for comparison.

Alongside the common output files, MDUR writes `html/mdur/{language}/` and
`styles.jsonl`. Plans, styles, and reports record text hashes, image routing, selected styles,
background-fit margins, reference scores, and font/browser metadata. Small
background margins indicate ambiguity. Scores are diagnostic differences,
not task performance or percentage accuracy.

Script/network injection is disabled, inline attributes are stripped, and
math remains literal TeX. Placeholder handling follows inert browser-style
parsing. Font coverage and glyph weights can still vary between scripts.
Output must be new and outside the input snapshot.

Project code and backgrounds are Apache-2.0. MMMU-Pro provenance remains on
the [benchmark card](https://huggingface.co/datasets/songjhPKU/PM4Bench).
Bootstrap 4.5.3 CSS retains its MIT notice and `bootstrap.LICENSE`; Noto fonts
retain SIL OFL notices. No Microsoft fonts are redistributed.
