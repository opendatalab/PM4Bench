<div align="center">

# Benchmarking and Boosting Multilingual Capabilities of LVLMs via OCR-Centric Reinforcement Learning

**PM<sup>4</sup>Bench · QGO-8B**

[![ACL Anthology](https://img.shields.io/badge/ACL-2026.emnlp--main.666-ED1C24.svg)](https://aclanthology.org/2026.emnlp-main.666/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.18484v3-B31B1B.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.18484v3)
[![Project Page](https://img.shields.io/badge/Project-Page-145C73.svg?logo=githubpages&logoColor=white)](https://little-bird-vodka.github.io/PM4Bench/)
[![Benchmark](https://img.shields.io/badge/Hugging_Face-Benchmark-FFD21E.svg)](https://huggingface.co/datasets/songjhPKU/PM4Bench)
[![Training Data](https://img.shields.io/badge/Hugging_Face-QGO_Train-FFD21E.svg)](https://huggingface.co/datasets/DatasetMan/PM4Bench-QGO-Train)
[![Model](https://img.shields.io/badge/Hugging_Face-QGO--8B-FFD21E.svg)](https://huggingface.co/DatasetMan/QGO-8B)

</div>

> [!IMPORTANT]
> **PM4Bench appears in the EMNLP 2026 Main Conference proceedings.** Read it in the [ACL Anthology](https://aclanthology.org/2026.emnlp-main.666/) ([PDF](https://aclanthology.org/2026.emnlp-main.666.pdf)) or as [arXiv v3](https://arxiv.org/abs/2503.18484v3).

PM4Bench is a strictly parallel multilingual vision-language benchmark over
10 languages and four tasks:

- **MDUR** — multi-discipline understanding and reasoning (1,730 items/language)
- **MIQA** — multi-image open-ended question answering (218 items/language)
- **MSOCR** — multi-scale OCR (100 items/language)
- **MGUI** — multilingual GUI grounding (200 items/language)

The paper uses controlled comparisons between interleaved and vision-only
input to identify OCR as a key source of cross-lingual performance gaps. QGO
then targets this bottleneck with OCR-centric GRPO on fully synthesized data.
This repository contains the clean v2 evaluation, validation, construction,
and training code for that study.

The dataset stores each image once and references it from portable JSONL
manifests. The `legacy-v1` tag preserves the earlier TSV/base64 bundle; do not
use it for new comparisons.

## What is where

```text
src/pm4bench/
├── metrics.py        the deterministic scores
├── miqa.py           the MIQA LLM-as-judge
├── task_prompts.py   the released prompts and query assembly
├── data/             manifest loading and integrity checks
├── vision/           benchmark image construction
└── qgo/              QGO training-data synthesis and reward
recipes/              the GRPO launcher
docs/                 task and format specifications
```

| If you want to | Go to |
| --- | --- |
| score predictions you already have | `pm4bench evaluate`, below |
| know what to prompt the model with | [docs/PROMPTS.md](docs/PROMPTS.md) |
| understand the manifest fields | [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) |
| regenerate the benchmark images | [docs/VISION_SYNTHESIS.md](docs/VISION_SYNTHESIS.md) |
| retrain or reproduce QGO-8B | [recipes/README.md](recipes/README.md) |
| pin versions and environments | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |
| read the module map | [ARCHITECTURE.md](ARCHITECTURE.md) |

## Installation

```bash
git clone https://github.com/opendatalab/PM4Bench.git
cd PM4Bench
python -m pip install -e .
```

Run `pm4bench --version` to confirm the release.

Install optional dependencies only when needed:

```bash
python -m pip install -e '.[judge]'   # MIQA LLM-as-judge
python -m pip install -e '.[synthesis]'  # all four vision tasks and QGO OCR synthesis
python -m pip install -e '.[dev]'     # tests and lint
```

## Dataset layout

```text
PM4Bench/
├── data/{mdur,miqa,msocr,mgui}/{language}.jsonl
├── assets/
│   ├── mdur/{traditional,vision}/...
│   ├── miqa/{traditional,vision}/...
│   ├── msocr/...
│   └── mgui/...
└── metadata/
    ├── release_inventory.json
    └── mgui/{gt,templates}/...
```

Download the snapshot from the [Hugging Face dataset](https://huggingface.co/datasets/songjhPKU/PM4Bench); the tree above is its root. The code repository does not contain the manifests or images.

```bash
pip install -U huggingface_hub
hf download songjhPKU/PM4Bench --repo-type dataset --local-dir PM4Bench-snapshot
```

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for schemas and asset resolution.

## Validate a snapshot

```bash
pm4bench validate --dataset-root /path/to/PM4Bench-snapshot
```

## Query the models under evaluation

The toolkit ships no inference code — you call your own models — but it does
ship the eight prompt definitions used to produce the published results, one per
task and input setting, each in all ten languages, and the rule for combining
each one with a benchmark row. Use them:
the metrics parse specific response markers, and a mismatched prompt is not an
error, it silently scores low.

```python
from pm4bench.data import load_manifest
from pm4bench.task_prompts import build_query

records, _digest = load_manifest("/path/to/PM4Bench-snapshot", "mdur", "en")
query = build_query("mdur-traditional", records[0])
# -> instruction + question stem + "(A). ..." options + "Your answer:"
```

`build_query` returns the full text input. Attach the row's images separately
in your client — `traditional_images` for the `traditional` setting,
`vision_image` for `vision`, `image` for MSOCR and MGUI — in the order the
manifest lists them; `<image>` marks where each belongs. Write your model's
reply to `response` and keep the row's `id`.

To inspect a prompt without writing code:

```bash
pm4bench prompts                                        # the eight released prompts
pm4bench prompts --id mgui-click --language en --raw     # one exact string
```

[docs/PROMPTS.md](docs/PROMPTS.md) specifies every response marker, which
metric reads it, the `assembly` rule, and the metric behaviors worth knowing.

## Evaluate rule-based tasks

Predictions are JSONL records containing the manifest `id` (or legacy
`index`) and a `response` string.

```json
{"id": "a01_search_en_1", "response": "(318, 312)"}
```

Every manifest record must have a prediction. If any is missing, `evaluate`
reports the missing identifiers and exits non-zero rather than scoring the gaps
as wrong — a score over partial predictions reads like a real result.

```bash
pm4bench evaluate \
  --task mgui \
  --manifest /path/to/data/mgui/en.jsonl \
  --predictions predictions/mgui_en.jsonl \
  --coordinate-space pixel
```

`--coordinate-space` has no default and is required for `--task mgui`, because
the released MGUI prompt asks for pixel coordinates.

MDUR, MSOCR, and MGUI use deterministic local metrics. MIQA follows the
paper's six-dimension LLM-as-judge protocol; the judge endpoint and model are
always supplied by the user and no provider-specific URL or credential is
stored in this repository.

MGUI also has a text-identification control, scored against the target
element's visible text:

```bash
pm4bench evaluate --task mgui-text --manifest data/mgui/en.jsonl --predictions predictions.jsonl
```

Run MIQA's resumable judge with an explicitly selected OpenAI-compatible model
and credential environment variable:

```bash
pm4bench judge-miqa \
  --manifest data/miqa/en.jsonl \
  --predictions predictions.jsonl \
  --output judgments/miqa_en.jsonl \
  --judge-model YOUR_JUDGE_MODEL
```

## Synthesize benchmark images

All four tasks use **one interface**, `pm4bench render-vision --task TASK`.
It reads the released manifests, checks their content against the `v2.0.0`
snapshot, and writes checked plans, generated images, and a provenance report
to a new directory. The same language, record-selection, and audit options
apply to MDUR, MIQA, MSOCR, and MGUI.

```bash
python -m pip install -e '.[synthesis]'
python -m playwright install chromium
python -m pm4bench.vision.fonts --output ./fonts
pm4bench render-vision \
  --task miqa \
  --dataset-root /path/to/PM4Bench-snapshot \
  --fonts-root ./fonts \
  --output-root ./rendered-miqa \
  --language en --limit 2
```

Use `--task mdur`, `msocr`, or `mgui` to select another task. MGUI uses the
platform system font stack, so omit `--fonts-root` for that task. Add
`--audit-only` to check inputs without rendering; no fonts or browser are
needed in audit mode.

[Vision synthesis](docs/VISION_SYNTHESIS.md) is the shared guide for setup,
the CLI and Python API, task-specific rendering options, and output layout.
Browser-based MDUR/MGUI and raster-based MIQA/MSOCR retain their own layout
logic behind this interface. Rendering does not overwrite the released data;
use the published images for comparable evaluation.

## QGO-8B

QGO-8B is the global-step-200 BF16 GRPO checkpoint derived from
`Qwen/Qwen3-VL-8B-Thinking`. The exact release data is provided as 19,500
training and 500 validation rows with embedded images. See
[recipes/README.md](recipes/README.md).

To generate multilingual OCR images and visible-word targets and pack training
Parquet, follow [OCR_TRAIN_SYNTHESIS.md](docs/OCR_TRAIN_SYNTHESIS.md). Original
text/background inputs and the published split inventory accompany the code.

## Reproducibility and provenance

- MDUR is derived from [MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro)
  (Apache-2.0).
- MIQA is derived from [MMDU](https://huggingface.co/datasets/laolao77/MMDU)
  (CC BY-NC 4.0).
- MSOCR and MGUI are PM4Bench-generated components.
- QGO-8B inherits the Apache-2.0 license of its Qwen base model.

Component-level provenance and licenses are documented in the dataset cards.
Project-owned code and artifacts are Apache-2.0.

## Tests

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest
```

For problems with data or evaluation, open a GitHub issue with the task,
language, sample identifier, and toolkit version. Do not include API keys or
private model outputs.

## Cite Us

```bibtex
@inproceedings{gao-etal-2026-benchmarking-boosting,
    title = "Benchmarking and Boosting Multilingual Capabilities of {LVLM}s via {OCR}-Centric Reinforcement Learning",
    author = "Gao, Junyuan  and
      Song, Jiahe  and
      Wu, Jiang  and
      Zhu, Runchuan  and
      Shenguanlin  and
      Wang, Shasha  and
      Wei, Xingjian  and
      Yang, Haote  and
      Li, Weijia  and
      Wang, Bin  and
      Wu, Lijun  and
      He, Conghui",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.666/",
    doi = "10.18653/2026.emnlp-main.666",
    pages = "15335--15365",
    ISBN = "979-8-89176-459-0"
}
```
