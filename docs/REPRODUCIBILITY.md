# Reproducibility

Release artifacts are versioned independently but share version `2.0.0` and a
machine-readable inventory. Inventories record row counts, file sizes, and
checksums for large transformed artifacts.

QGO training uses 32 prompts per optimization batch and 8 rollouts per prompt,
yielding 256 sampled trajectories per step. The released paper checkpoint is
global step 200 and uses BF16 weights.

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
