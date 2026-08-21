import math

from pm4bench.qgo.reward import qgo_reward


def test_qgo_reward_combines_accuracy_and_format() -> None:
    result = qgo_reward("ocr", r"reasoning \boxed{answer}", "answer")
    assert result["accuracy_score"] == 1.0
    assert math.isclose(
        result["score"],
        0.8 * result["accuracy_score"] + 0.2 * result["length_reward_final"],
    )


def test_qgo_reward_requires_boxed_answer() -> None:
    result = qgo_reward("ocr", "answer", "answer")
    assert result["accuracy_score"] == 0.0


def test_qgo_reward_preserves_empty_target_rule() -> None:
    result = qgo_reward("ocr", "", "")
    assert result["accuracy_score"] == 1.0
