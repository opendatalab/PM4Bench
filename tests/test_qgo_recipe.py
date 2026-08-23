from pathlib import Path

RECIPE = Path(__file__).parents[1] / "recipes" / "qgo" / "train_grpo.sh"


def test_qgo_recipe_uses_v1_training_parameters() -> None:
    text = RECIPE.read_text(encoding="utf-8")
    expected = (
        "data.train_batch_size=32",
        "data.max_prompt_length=8192",
        "data.max_response_length=4096",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096",
        "trainer.n_gpus_per_node=8",
        "trainer.total_epochs=1",
        "trainer.save_freq=50",
        "trainer.test_freq=10",
        "custom_reward_function.path=src/pm4bench/qgo/reward.py",
        "custom_reward_function.name=qgo_reward",
    )
    for setting in expected:
        assert setting in text


def test_qgo_recipe_excludes_ablation_and_private_configuration() -> None:
    text = RECIPE.read_text(encoding="utf-8")
    forbidden = (
        "ocr_reward_func_only_ocr",
        "WANDB_API_KEY",
        "WANDB_BASE_URL",
    )
    for value in forbidden:
        assert value not in text
