#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to Qwen/Qwen3-VL-8B-Thinking or a local snapshot}"
: "${TRAIN_FILES:?Set TRAIN_FILES to the released train Parquet glob}"
: "${VAL_FILES:?Set VAL_FILES to the released validation Parquet file}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new checkpoint directory}"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.prompt_key=prompt \
  data.train_batch_size=32 \
  data.max_prompt_length=8192 \
  data.max_response_length=4096 \
  actor_rollout_ref.rollout.dtype=bfloat16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.fsdp_config.dtype=bfloat16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.default_hdfs_dir=null \
  trainer.project_name=pm4bench-qgo \
  trainer.experiment_name=qgo-8b \
  trainer.n_gpus_per_node=8 \
  trainer.logger="['console']" \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=50 \
  trainer.test_freq=10 \
  custom_reward_function.path=src/pm4bench/qgo/reward.py \
  custom_reward_function.name=qgo_reward
