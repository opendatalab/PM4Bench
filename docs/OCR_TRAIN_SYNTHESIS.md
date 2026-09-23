# QGO OCR training-data synthesis

This recipe provides multilingual text sampling, browser image generation,
visible-word annotation, and veRL Parquet packing. It is separate from the
MSOCR benchmark compositor. For the paper's exact 19,500 training and 500
validation samples, use [the published corpus](https://huggingface.co/datasets/DatasetMan/PM4Bench-QGO-Train).
These commands create a new corpus with the same construction design; they
do not replace the frozen data or promise identical historical pixels.

## Inputs and setup

The package includes the original 22,519-entry, ten-language title dictionary
and nine backgrounds. The dictionary SHA-256 is
`8f8e46bb0415b5fcc67a5c3343a010c3b0d97ab9ea4dafa860464e250d1fdd0c`.
All 20,000 released targets/prompts match the original GT; all 2,401,211
visible word blocks were found in this dictionary. No replacement benchmark
text or translation service is used.

```bash
python -m pip install -e '.[synthesis]'
python -m playwright install chromium
python -m pm4bench.vision.fonts --output ./fonts
```

The downloader pins Noto files and their SIL OFL notices. The renderer loads
Arabic, SC, KR, Thai, and general Noto faces explicitly, retaining the original
Noto-first design with an explicit Thai fallback instead of a system Thai font.
Font hashes and browser version are recorded. `--browser-executable` selects
an existing browser.

## Generate and replay

```bash
python -m pm4bench.qgo.synthesize \
  --fonts-root ./fonts --count 20000 --seed 42 --output-root ./qgo-synthetic
```

Start with `--count 12` for a smoke run. Optional `--text-pool` accepts an
object mapping source identifiers to language/text maps, with all ten languages
present. The bundled dictionary is unchanged; 41 blank translations are
excluded from new sampling. Invalid input fails rather than producing dummy words.

The recipe retains the original viewport choices, language cycling,
14/18/22/28/32 px word sizes, weight/slant variations, shadows, and backgrounds.
Only words fully within the 2 px viewport margin enter the OCR target; other
words are hidden. Vertical-center grouping orders Arabic lines right-to-left
and other lines left-to-right. Text is HTML-escaped and network requests are blocked.

Outputs: `img/`, `gt/`, `html/`, `plans.jsonl`, `images.jsonl`, `report.json`.
GT includes visible word boxes and the OCR target. The viewport is explicit,
independent of browser window outer height. Each record has a seeded random
state. Replay saved plans with the same fonts/browser:

```bash
python -m pm4bench.qgo.synthesize \
  --fonts-root ./fonts --plans ./qgo-synthetic/plans.jsonl \
  --output-root ./qgo-replayed
```

Historical multiprocess RNG scheduling and system-font fallback are not
reproduced by claiming that a seed alone regenerates the published corpus.

## Pack Parquet

```bash
python -m pm4bench.qgo.pack \
  --input-root ./qgo-synthetic --output-root ./qgo-parquet \
  --validation-size 500 --shard-size 500 --seed 42
```

For a 12-image smoke run use `--validation-size 2`. Outputs include
`data/train/*.parquet`, `data/test/*.parquet`, `split.json`, and `report.json`.
Rows carry the original OCR prompt, embedded PNG bytes, reward target, answer,
and split. Paths are portable basenames. The later GUI-mixture branch is excluded.

New-data splits sort filenames before shuffling. To reuse a split, pass
`--split-manifest ./previous-run/split.json`; its `train`/`test` lists must cover
each input image once. The bundled `qgo/resources/released_split.json` records
the published corpus's original image filenames and order. Use it only when
repacking those historical image/GT pairs, not newly sampled `ocr_*.png` files.

Commands require fresh output directories outside inputs. Missing GT, invalid
images, duplicate splits, and target/box inconsistencies fail explicitly.
Project-owned code, dictionary, and backgrounds are Apache-2.0; fonts retain
upstream licenses. Continue with [GRPO training](../recipes/qgo/README.md).
