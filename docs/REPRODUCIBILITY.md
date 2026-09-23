# Reproducibility

Toolkit 2.2 adds [MDUR synthesis](MDUR_SYNTHESIS.md) and the
[QGO OCR-data pipeline](OCR_TRAIN_SYNTHESIS.md). MDUR aligns public transcripts
and images while accepting small visual differences. OCR synthesis records
per-record plans, font/browser hashes, visible-word targets, and splits.
New generated corpora are separate from unchanged published training data.

The 2.2 release checks all 17,300 MDUR transcript/image-routing records and
all 20,000 OCR targets/prompts against their source records. Browser regression
checks replay of twelve multilingual OCR samples, including image and GT
equality. MDUR visual smoke checks cover English and Chinese, supplied-style
replay, and the Noto-only fallback; a full pixel-identity test is not asserted.

Release artifacts are versioned independently but share version `2.0.0` and a
machine-readable inventory. Inventories record row counts, file sizes, and
checksums for large transformed artifacts.

Toolkit 2.1 adds the MIQA and MSOCR construction recipes in
[VISION_SYNTHESIS.md](VISION_SYNTHESIS.md). They read the frozen v2.0.0
benchmark manifests, preserve the released text and image order, and record
the font and raster-library versions. The benchmark data version is unchanged.
Generated pixels can differ from the released images when fonts or raster
libraries differ; published evaluation uses the released images.

QGO training uses 32 prompts per optimization batch and 8 rollouts per prompt,
yielding 256 sampled trajectories per step. Prompt and response limits are
8,192 and 4,096, respectively. The formatting reward uses a `[1000, 10000]`
length interval, over-length scale `1200`, repetition threshold `0.6`, and
length-reward, length-penalty, and repetition-penalty weights `0.2`, `0.8`,
and `0.4`. Accuracy and formatting are combined with weights `0.8` and `0.2`.
The released paper checkpoint is global step 200 and uses BF16 weights.

The training-data release preserves row order, prompt, embedded image bytes,
reward target, ability, split, and auxiliary answer fields. Only nested image
path strings are converted from machine-specific absolute paths to basenames.

MGUI templates are rendered at a 1280 x 800 CSS viewport with device scale 1.
The renderer waits for browser fonts, captures `[data-qid]` element rectangles,
and can compare every generated rectangle and visible-text field with the
released reference GT using a configurable floating-point tolerance.

The canonical v2 images and GT were created with Jinja2 3.1.6, Selenium 4.40.0,
and Chrome for Testing 119.0.6045.105. The clean reconstruction uses pinned
Jinja2 3.1.6, Playwright 1.48.0, and Chromium 130.0.6723.31. Its full 1,000-page
structural regression requires identical files, DOM targets, visible text,
questions, and metadata. Exact bounding boxes remain platform-font-sensitive;
use `--comparison-mode strict` to audit geometry on a controlled image, and
treat the released images and GT as canonical for evaluation.

Two historical templates access dictionary keys named `copy` and `update`
through Jinja attribute syntax, so their visible text contains a Python method
representation with a process-specific memory address. The comparator
normalizes only that hexadecimal address; all surrounding text and geometry
remain strict.

The clean MSOCR implementation is also replayed against the 1,000 QGO-8B
paper-era per-sample score records. Release validation requires zero mismatches
and reproduces the reported overall mean of 8.174 before rounding.
