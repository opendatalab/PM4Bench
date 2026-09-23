# Roadmap

## v2.0.0 release

- [x] Define portable deduplicated benchmark schemas.
- [x] Extract and validate all four task inventories.
- [x] Isolate the global-step-200 QGO-8B checkpoint.
- [x] Finish deterministic local evaluators and MIQA judge adapter.
- [x] Reconstruct the deterministic MGUI renderer and GT comparator.
- [x] Run the full 1,000-page structural renderer regression.
- [x] Validate the sanitized QGO Parquet release.
- [x] Add end-to-end tiny fixtures and CI.
- [x] Align the QGO reward coefficients and sanitized GRPO launcher.
- [x] Refresh the project page for the four-task/QGO release and EMNLP 2026 acceptance.
- [x] Publish the benchmark, training data, model, code, and v2.0.0 tag.
- [x] Link arXiv v3 and add cross-repository navigation and copy-ready citation blocks.
- [x] Pin every citation URL to arXiv v3 and avoid stale Hugging Face Papers metadata.

## Post-release

- [x] Publish MIQA and MSOCR vision synthesis from the released text manifests.
- [x] Add manifest locks, image-label alignment checks, and pinned font downloads.
- [ ] Recover the matching MDUR renderer/style configuration and complete its release.
- Add a prediction-format converter for common inference frameworks.
- Add resumable evaluation reports without coupling them to paper figures.
- Publish explicit compatibility tests for later Transformers/veRL releases.
- Publish a container that reproduces the historical MGUI platform-font geometry.
