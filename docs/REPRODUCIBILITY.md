# Reproducibility

Pinned versions, fixed parameters and the environments the released artifacts
were produced in.

## Evaluation

Deterministic metrics have no optional dependencies and no network access. The
MSOCR implementation reproduces the published per-sample scores for all 1,000
records, with an overall mean of 8.174 before rounding.

The MIQA judge is the only model-dependent score. Historical API model versions
cannot be recreated by a present-day endpoint, so judge numbers are comparable
within one run of one model, not across model revisions.

The eight released prompts, their response markers and the assembly rule are
specified in [PROMPTS.md](PROMPTS.md).

## Benchmark image construction

MGUI templates render at a 1280 x 800 CSS viewport with device scale 1. The
renderer waits for browser fonts, captures `[data-qid]` element rectangles, and
can compare every generated rectangle and visible-text field against the
released GT with a configurable tolerance.

| | version |
| --- | --- |
| canonical released images | Jinja2 3.1.6, Selenium 4.40.0, Chrome for Testing 119.0.6045.105 |
| clean reconstruction | Jinja2 3.1.6, Playwright 1.48.0, Chromium 130.0.6723.31 |

The 1,000-page structural check requires identical files, DOM targets, visible
text, questions and metadata. Bounding boxes stay platform-font-sensitive: use
`--comparison-mode strict` to audit geometry on a controlled image, and treat
the released images and GT as canonical for evaluation.

Two templates access dictionary keys named `copy` and `update` through Jinja
attribute syntax, so their visible text contains a Python method representation
with a process-specific memory address. The comparator normalizes only that
hexadecimal address; all surrounding text and geometry remain strict.

Optional font downloads are pinned by source revision and checksum. Generated
pixels can differ from the released images when fonts or raster libraries
differ, so published evaluation uses the released images.

## QGO training

| parameter | value |
| --- | --- |
| prompts per optimization batch | 32 |
| rollouts per prompt | 8 (256 sampled trajectories per step) |
| maximum prompt / response length | 8,192 / 4,096 |
| length-reward interval | `[1000, 10000]` |
| over-length scale | 1,200 |
| repetition threshold | 0.6 |
| length-reward / length-penalty / repetition-penalty weights | 0.2 / 0.8 / 0.4 |
| accuracy / formatting weights | 0.8 / 0.2 |
| released checkpoint | global step 200, BF16 |

The training-data release preserves row order, prompt, embedded image bytes,
reward target, ability, split and auxiliary answer fields; only nested image
paths are converted from absolute paths to basenames.

Exact stochastic training replay is not promised. Recompute using these
configurations and the public base weights.

## Artifact versions

The four released artifacts are versioned independently and share version
`2.0.0`, with a machine-readable inventory recording row counts, file sizes and
checksums. The manifest lock in `data/inputs.lock.json` covers all 40
language/task files, so `load_manifest` rejects a snapshot that does not match
the published text.
