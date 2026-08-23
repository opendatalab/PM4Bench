# QGO-8B recipe

QGO-8B starts from `Qwen/Qwen3-VL-8B-Thinking` and uses GRPO over the released
synthetic multilingual OCR Parquet.

Canonical settings:

- optimizer learning rate: `1e-6`
- prompt batch: `32`
- rollouts per prompt: `8` (256 sampled trajectories per step)
- actor PPO mini-batch: `32`
- actor PPO micro-batch/GPU: `1`
- maximum prompt length: `8192`
- maximum response length: `4096`
- rollout temperature: `1.0`
- KL loss coefficient: `0.01`
- training GPUs: `8`
- released checkpoint: global step `200`
- reward length interval: `[1000, 10000]`
- over-length scale: `1200`
- repetition threshold: `0.6`
- length reward / length penalty / repetition penalty weights: `0.2 / 0.8 / 0.4`
- accuracy / format reward weights: `0.8 / 0.2`

`train_grpo.sh` contains no machine paths or credentials. Supply the base
model, dataset snapshot, output directory, and reward module through explicit
environment variables.

The release was smoke-tested against the available environment boundary
`veRL 0.8.0.dev0`, `Transformers 4.57.6`, and `PyTorch 2.10.0`. The exact Git
commit of the historical veRL checkout was not captured, so this recipe does
not claim a byte-for-byte environment lock. Pin a veRL revision compatible
with the arguments in `train_grpo.sh`, record it in your run metadata, and use
the released step-200 checkpoint when exact paper-weight evaluation is needed.
