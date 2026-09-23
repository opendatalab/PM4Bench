# Vision data synthesis

The MIQA and MSOCR recipes reconstruct images from the public PM4Bench
manifests. The MIQA layout comes from the MMDU single-question compositor;
MSOCR uses the SizeBench multi-scale compositor. MGUI construction is available
through `pm4bench render-mgui` as described in the main README.

The MDUR renderer is pending recovery of a matching renderer/style pair.
Its benchmark images remain available in the dataset.

## Inputs

Download the benchmark at `v2.0.0` (commit
`525f8686cf360eabc92311ab901b3ff8e0675575`) from
[songjhPKU/PM4Bench](https://huggingface.co/datasets/songjhPKU/PM4Bench/tree/v2.0.0).
The recipes require `data/miqa/*.jsonl`, `data/msocr/*.jsonl`, and MIQA's
`assets/miqa/traditional/` images. No private text files, original upstream
downloads, TSV files, or translation API calls are needed.

| Task | Text source | Layout |
| --- | --- | --- |
| MIQA | `question` and `rendered_text`, checked against each other | 42 px text; 1,280 px text width; source images bounded to 1,200 × 700; 20 px outer padding |
| MSOCR | Ordered `lines[].text` and `lines[].font_size`, checked against `ground_truth` | 1,280 × 720; 20 lines from 40 to 2 px; horizontally centered |

MIQA retains source image numbers such as **1, 3, 4**, even though the portable
assets are named `image_1.png`, `image_2.png`, and `image_3.png`. Arabic image
labels retain the ordering in the released OCR text. The question is checked
after removing the historical `<ImageHere>` image prefix; any disagreement
with `rendered_text` stops the job. Reference answers are never drawn.
Text-block height accounts for the font's baseline offset and descenders,
preventing clipping with Arabic and other fonts whose ink extends below the
old height-only crop.

Asset filenames define their numeric order. In the v2.0.0 manifests,
`105_0` has 11 images and its `traditional_images` list is lexicographically
ordered (`1, 10, 11, 2, ...`). The compositor restores numeric order before
pairing assets with their source labels. Apply the same numeric ordering when
building traditional-setting input from that manifest.

MSOCR consumes the released lines in order. The earlier title sampler is not
part of reconstruction: rerunning it would change the evaluation text. The
renderer does not translate, resample, shuffle, or reverse those lines.

By default, the input manifest SHA-256 must match the released text. The
expected values are packaged in `vision/resources/inputs.lock.json`. Use
`--allow-custom-manifest` only for your own modified-text experiment; the same
row-level consistency checks still apply and the report marks the custom input.

## Environment and fonts

```bash
python -m pip install -e '.[vision]'
python -m pm4bench.vision.fonts --output ./fonts
```

The downloader retrieves five Noto font files and their SIL Open Font License
notices from a fixed Google Fonts commit, and verifies their SHA-256 values.
It preserves existing files and rejects a checksum mismatch. The font files
are downloaded separately from the code package.

The filenames follow the historical recipes. The pinned downloads are variable
fonts, loaded at their **Regular** instance, including Chinese and Korean fonts
whose default axis value is Thin. You can instead supply the original static
fonts under the same filenames:

- `NotoSans-Regular.ttf`: English, Czech, Hungarian, Russian, Serbian, Vietnamese
- `NotoSansArabic-Regular.ttf`: Arabic
- `NotoSansThai-Regular.ttf`: Thai
- `NotoSansKR-Regular.ttf`: Korean
- `NotoSansSC-Regular.ttf`: Chinese

Pillow is pinned to 11.3.0 and requires libraqm for text shaping. The usual
binary wheels include it. Check a custom installation with
`python -c "from PIL import features; print(features.check_feature('raqm'))"`.

Font coverage is recorded in `report.json`. The pinned Noto SC font lacks
`U+2CE2F`, which occurs in the released Chinese MSOCR text. The string is kept
unchanged and the font's missing-glyph symbol is rendered for this character.
Use `--strict-glyphs` to reject missing characters before rendering, or provide
a font covering the complete text. Font substitution changes geometry; output
reports record the exact font hashes.

## Run

First validate every input without loading fonts or rendering images:

```bash
pm4bench render-vision --task miqa \
  --dataset-root /path/to/PM4Bench-snapshot \
  --output-root ./audit-miqa --audit-only
pm4bench render-vision --task msocr \
  --dataset-root /path/to/PM4Bench-snapshot \
  --output-root ./audit-msocr --audit-only
```

Render all 2,180 MIQA images and 1,000 MSOCR images:

```bash
pm4bench render-vision --task miqa \
  --dataset-root /path/to/PM4Bench-snapshot \
  --fonts-root ./fonts --output-root ./rendered-miqa --workers 4
pm4bench render-vision --task msocr \
  --dataset-root /path/to/PM4Bench-snapshot \
  --fonts-root ./fonts --output-root ./rendered-msocr --workers 4
```

For a quick check, add `--language en --limit 2`. `--language` and `--id` can
be repeated. A limit applies separately to each selected language. Every run
requires a new output directory outside the input snapshot; failed or previous
runs are kept for inspection.

## Outputs and validation

Each run writes:

- `plans.jsonl`: exact text blocks, line sizes or image order, and text hashes
- `images/{task}/{language}/{id}.png`: generated images
- `images.jsonl`: output image paths and SHA-256 values
- `report.json`: input manifest hashes, font hashes, glyph coverage, row counts,
  and Python/Pillow/FreeType/libraqm versions

Audit-only runs produce the plans and report. Paths in plans refer to the
input snapshot; paths in `images.jsonl` refer to the generated output root.

The release checks all 2,180 MIQA and 1,000 MSOCR text records against the
public manifests, including question-prefix removal and non-consecutive image
labels, and renders all 3,180 images in the pinned Pillow environment. Tests
also cover double-digit image ordering, changed text, inconsistent ground truth, path escape,
output protection, and raster generation. Text alignment does not imply
pixel equality across font or FreeType versions. Use the public benchmark
images for comparable evaluation scores.

## Licenses

Project-owned construction code is Apache-2.0. MIQA source data retains MMDU's
CC BY-NC 4.0 terms; see the benchmark's
[component licenses](https://huggingface.co/datasets/songjhPKU/PM4Bench/blob/v2.0.0/UPSTREAM_LICENSES.md).
MSOCR is a PM4Bench-generated component. Downloaded Noto fonts retain their
SIL OFL 1.1 notices; exact upstream URLs and checksums are in
`vision/resources/fonts.lock.json`.
