"""Shared batching and failure-isolation helpers."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class BatchFailure(Generic[T]):
    items: list[T]
    error: Exception


def iter_bounded_batches(
    items: Sequence[T],
    *,
    max_items: int,
    max_chars: int,
    size_of: Callable[[T], int],
) -> Iterable[list[T]]:
    """Yield batches bounded by both item count and estimated input size."""
    if max_items < 1:
        raise ValueError("batch_size must be at least 1")
    if max_chars < 1:
        raise ValueError("batch_max_input_chars must be at least 1")

    batch: list[T] = []
    batch_chars = 0
    for item in items:
        item_chars = max(0, size_of(item))
        if batch and (
            len(batch) >= max_items or batch_chars + item_chars > max_chars
        ):
            yield batch
            batch = []
            batch_chars = 0

        # An individual oversized item must remain intact. If it fails, the
        # caller reports that item rather than silently truncating its content.
        batch.append(item)
        batch_chars += item_chars

    if batch:
        yield batch


def extract_with_bisection(
    items: list[T],
    extract: Callable[[list[T]], list[R]],
    *,
    should_split: Callable[[Exception], bool] = lambda _error: True,
) -> tuple[list[R], list[BatchFailure[T]], int]:
    """
    Extract a batch, recursively splitting failures to isolate bad inputs.

    Returns successful results, irreducible single-item failures, and the
    number of extraction calls attempted.
    """
    try:
        return extract(items), [], 1
    except Exception as error:
        if len(items) <= 1 or not should_split(error):
            return [], [BatchFailure(items=list(items), error=error)], 1

        midpoint = len(items) // 2
        left_results, left_failures, left_calls = extract_with_bisection(
            items[:midpoint], extract, should_split=should_split
        )
        right_results, right_failures, right_calls = extract_with_bisection(
            items[midpoint:], extract, should_split=should_split
        )
        return (
            left_results + right_results,
            left_failures + right_failures,
            1 + left_calls + right_calls,
        )
