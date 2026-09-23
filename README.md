<div align="center">

# Benchmarking and Boosting Multilingual Capabilities of LVLMs via OCR-Centric Reinforcement Learning

**PM<sup>4</sup>Bench · QGO-8B**

[![arXiv](https://img.shields.io/badge/arXiv-2503.18484v3-B31B1B.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2503.18484v3)
[![Project Page](https://img.shields.io/badge/Project-Page-145C73.svg?logo=githubpages&logoColor=white)](https://little-bird-vodka.github.io/PM4Bench/)
[![Benchmark](https://img.shields.io/badge/Hugging_Face-Benchmark-FFD21E.svg)](https://huggingface.co/datasets/songjhPKU/PM4Bench)
[![Training Data](https://img.shields.io/badge/Hugging_Face-QGO_Train-FFD21E.svg)](https://huggingface.co/datasets/DatasetMan/PM4Bench-QGO-Train)
[![Model](https://img.shields.io/badge/Hugging_Face-QGO--8B-FFD21E.svg)](https://huggingface.co/DatasetMan/QGO-8B)

</div>

> [!IMPORTANT]
> **PM4Bench has been accepted to the EMNLP 2026 Main Conference.** The revised manuscript is available as [arXiv v3](https://arxiv.org/abs/2503.18484v3).

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

The former TSV/base64 representation is retired. The v2 dataset stores each
image once and references it from portable JSONL manifests.

## Installation

```bash
git clone https://github.com/opendatalab/PM4Bench.git
cd PM4Bench
python -m pip install -e .
```

Install optional dependencies only when needed:

```bash
python -m pip install -e '.[judge]'   # MIQA LLM-as-judge
python -m pip install -e '.[render]'  # deterministic MGUI rendering
python -m pip install -e '.[vision]'  # MIQA and MSOCR image synthesis
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

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for schemas and asset resolution.

## Validate a snapshot

```bash
pm4bench validate --dataset-root /path/to/PM4Bench-snapshot
```

## Evaluate rule-based tasks

Predictions are JSONL records containing the manifest `id` (or legacy
`index`) and a `response` string.

```json
{"id": "a01_search_en_1", "response": "(318, 312)"}
```

```bash
pm4bench evaluate \
  --task mgui \
  --manifest /path/to/data/mgui/en.jsonl \
  --predictions predictions/mgui_en.jsonl \
  --coordinate-space normalized-1000
```

MDUR, MSOCR, and MGUI use deterministic local metrics. MIQA follows the
paper's six-dimension LLM-as-judge protocol; the judge endpoint and model are
always supplied by the user and no provider-specific URL or credential is
stored in this repository.

The deterministic evaluator also exposes the paper's OCR diagnostics and the
MGUI text-identification sanity check:

```bash
pm4bench evaluate --task mdur-ocr --manifest data/mdur/en.jsonl --predictions predictions.jsonl
pm4bench evaluate --task miqa-ocr --manifest data/miqa/en.jsonl --predictions predictions.jsonl
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

The construction tools read the released dataset manifests. MIQA preserves
the question, image order, and source image labels. MSOCR renders the released
line text at its recorded font sizes. The default input checks pin these
strings to the benchmark's `v2.0.0` revision.
Text blocks expand to retain the full glyph height with the selected fonts.

```bash
python -m pip install -e '.[vision]'
python -m pm4bench.vision.fonts --output ./fonts
pm4bench render-vision \
  --task miqa \
  --dataset-root /path/to/PM4Bench-snapshot \
  --fonts-root ./fonts \
  --output-root ./rendered-miqa \
  --language en --limit 2
```

Use `--task msocr` for the multi-scale OCR renderer. Full-dataset commands,
input auditing, font requirements, and output formats are described in
[VISION_SYNTHESIS.md](docs/VISION_SYNTHESIS.md).

To reconstruct MGUI with the pinned reference renderer:

```bash
python -m playwright install chromium
pm4bench render-mgui \
  --templates-root metadata/mgui/templates \
  --output-root rendered-mgui \
  --compare-gt metadata/mgui/gt \
  --comparison-mode structure
```

## QGO-8B

QGO-8B is the global-step-200 BF16 GRPO checkpoint derived from
`Qwen/Qwen3-VL-8B-Thinking`. The exact release data is provided as 19,500
training and 500 validation rows with embedded images. See
[recipes/qgo/README.md](recipes/qgo/README.md).

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
@misc{gao2026benchmarkingboostingmultilingualcapabilities,
      title={Benchmarking and Boosting Multilingual Capabilities of LVLMs via OCR-Centric Reinforcement Learning},
      author={Junyuan Gao and Jiahe Song and Jiang Wu and Runchuan Zhu and Guanlin Shen and Shasha Wang and Xingjian Wei and Haote Yang and Weijia Li and Bin Wang and Lijun Wu and Conghui He},
      year={2026},
      eprint={2503.18484},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18484v3},
}
```
