from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2 + 1
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    left_mean = mean(left_rank)
    right_mean = mean(right_rank)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_rank, right_rank))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left_rank)
        * sum((y - right_mean) ** 2 for y in right_rank)
    )
    return numerator / denominator if denominator else 0.0


def binomial_confidence_interval(k: int, m: int, z: float = 1.96):
    """Wilson interval with a fixed normal critical value (95% by default)."""
    if m == 0:
        return (0.0, 0.0)
    p = k / m
    denom = 1 + z * z / m
    center = (p + z * z / (2 * m)) / denom
    half = (z * np.sqrt(p * (1 - p) / m + z * z / (4 * m * m))) / denom
    return (max(0.0, center - half), min(1.0, center + half))
