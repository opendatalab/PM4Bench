import math

from pm4bench.qgo.reward import (
    ACCURACY_WEIGHT,
    FORMAT_REWARD_WEIGHT,
    LENGTH_PENALTY_WEIGHT,
    LENGTH_REWARD_WEIGHT,
    REPETITION_PENALTY_WEIGHT,
    REPETITION_THRESHOLD,
    format_reward,
    qgo_reward,
)


def test_qgo_reward_hyperparameters() -> None:
    assert REPETITION_THRESHOLD == 0.6
    assert LENGTH_REWARD_WEIGHT == 0.2
    assert LENGTH_PENALTY_WEIGHT == 0.8
    assert REPETITION_PENALTY_WEIGHT == 0.4
    assert ACCURACY_WEIGHT == 0.8
    assert FORMAT_REWARD_WEIGHT == 0.2


def test_qgo_reward_repetition_coefficient() -> None:
    result = format_reward("a" * 1000)
    expected_repetition = 1.0 - (1.0 / 997.0) / 0.6
    assert math.isclose(result["repetition"], expected_repetition)
    assert math.isclose(result["combined"], 0.2 - 0.4 * expected_repetition)


def test_qgo_reward_preserves_placeholder_penalty() -> None:
    result = format_reward("[Internal Processing Omitted]")
    assert result == {
        "combined": -1.2,
        "length": 0.0,
        "length_penalty": 0.0,
        "repetition": 0.0,
    }


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
