"""Small evaluation helpers for latency, token, and cost metadata."""

from __future__ import annotations


def measure_latency_ms(start_time: float, end_time: float) -> float:
    """Convert perf-counter timestamps to milliseconds for reports."""
    return round((end_time - start_time) * 1000, 3)


def estimate_cost(
    tokens_input: int | None,
    tokens_output: int | None,
    price_per_1k_input: float | None = None,
    price_per_1k_output: float | None = None,
) -> float | None:
    """Estimate cost when token counts and prices are available.

    Phase 1 often has no token metadata, so missing values return None instead
    of inventing a cost number.
    """
    if (
        tokens_input is None
        or tokens_output is None
        or price_per_1k_input is None
        or price_per_1k_output is None
    ):
        return None

    input_cost = (tokens_input / 1000) * price_per_1k_input
    output_cost = (tokens_output / 1000) * price_per_1k_output
    return round(input_cost + output_cost, 6)
