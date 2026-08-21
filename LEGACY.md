# Legacy v1 release

PM4Bench v2 replaces the original TSV/base64 data bundle and experiment-tree
scripts with portable manifests, deduplicated assets, a tested Python package,
and separately versioned Hugging Face artifacts.

The original public state remains available in Git history at commit
`6562ce7`. It will also be named by the `legacy-v1` tag when v2 is published.
Do not use the legacy data or scripts for new comparisons; they predate the
current benchmark, QGO training data, and paper checkpoint.
