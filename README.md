# PM4Bench

PM4Bench is a strictly parallel multilingual vision-language benchmark over
10 languages and four tasks:

- **MDUR** — multi-discipline understanding and reasoning (1,730 items/language)
- **MIQA** — multi-image open-ended question answering (218 items/language)
- **MSOCR** — multi-scale OCR (100 items/language)
- **MGUI** — multilingual GUI grounding (200 items/language)

This repository contains the clean v2 evaluation, validation, construction,
and QGO training code. Large artifacts live on Hugging Face:

- Benchmark: <https://huggingface.co/datasets/songjhPKU/PM4Bench>
- QGO training data: <https://huggingface.co/datasets/DatasetMan/PM4Bench-QGO-Train>
- QGO-8B: <https://huggingface.co/DatasetMan/QGO-8B>

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
