from __future__ import annotations

import math
import re
from typing import Any

from pm4bench.metrics import levenshtein_distance

REPETITION_THRESHOLD = 0.6
REPETITION_PENALTY_WEIGHT = 0.2
LENGTH_PENALTY_WEIGHT = 0.8
LENGTH_REWARD_WEIGHT = 0.2
FORMAT_REWARD_WEIGHT = 0.2
ACCURACY_WEIGHT = 0.8


def format_reward(
    solution: str, minimum_length: int = 1000, maximum_length: int = 10000
) -> dict[str, float]:
    length = len(solution)
    if "[Internal Processing Omitted]" in solution:
        return {"combined": -1.2, "length": 0.0, "length_penalty": 0.0, "repetition": 0.0}

    check_text = solution[-3000:] if length > 5000 else solution
    ngrams = [check_text[index : index + 4] for index in range(max(0, len(check_text) - 3))]
    diversity = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
    repetition = max(0.0, 1.0 - diversity / REPETITION_THRESHOLD)

    length_reward = float(minimum_length <= length <= maximum_length)
    if length < minimum_length:
        length_penalty = 1.0 - length / minimum_length
    elif length > maximum_length:
        length_penalty = min(1.0, math.log1p((length - maximum_length) / 1200.0))
    else:
        length_penalty = 0.0
    combined = (
        length_reward * LENGTH_REWARD_WEIGHT
        - length_penalty * LENGTH_PENALTY_WEIGHT
        - repetition * REPETITION_PENALTY_WEIGHT
    )
    return {
        "combined": combined,
        "length": length_reward,
        "length_penalty": length_penalty,
        "repetition": repetition,
    }


def qgo_reward(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, float | int]:
    del data_source, extra_info
    formatting = format_reward(solution_str)
    match = re.search(r"\\boxed\{(.*?)\}", solution_str, re.DOTALL)
    prediction = match.group(1).strip() if match else None
    target = ground_truth.strip()
    if not prediction and not target:
        accuracy = 1.0
    elif not prediction or not target:
        accuracy = 0.0
    else:
        denominator = max(len(prediction), len(target))
        accuracy = 1.0 - levenshtein_distance(prediction, target) / denominator
    total = ACCURACY_WEIGHT * accuracy + FORMAT_REWARD_WEIGHT * formatting["combined"]
    return {
        "score": float(total),
        "accuracy_score": float(accuracy),
        "length_reward_final": float(formatting["combined"]),
        "length_reward": float(formatting["length"]),
        "length_penalty": float(formatting["length_penalty"]),
        "repetition_penalty": float(formatting["repetition"]),
        "length": len(solution_str),
    }
