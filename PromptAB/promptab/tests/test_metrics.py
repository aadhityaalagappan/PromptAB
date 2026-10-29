import pytest

from promptab.metrics import (
    calculate_cost,
    summarize_performance,
    paired_bootstrap_ci,
    summarize_comparison
)


def test_cost():
    pricing = {
        "input": 1.0,
        "output": 2.0
    }

    result = calculate_cost(
        1_000_000,
        500_000,
        pricing
    )

    assert result == pytest.approx(2.0)


def test_missing_pricing():
    assert calculate_cost(
        100,
        50,
        None
    ) is None


def test_performance():
    runs = [
        {
            "latency_seconds": 1.0,
            "input_tokens": 100,
            "output_tokens": 50
        },
        {
            "latency_seconds": 2.0,
            "input_tokens": 200,
            "output_tokens": 100
        }
    ]

    result = summarize_performance(
        runs,
        pricing=None
    )

    assert result["requests"] == 2
    assert result["total_input_tokens"] == 300
    assert result["total_output_tokens"] == 150
    assert result["total_tokens"] == 450
    assert result["avg_tokens_per_request"] == 225
    assert result["avg_latency_seconds"] == 1.5


def test_bootstrap_single_case():
    assert paired_bootstrap_ci([10.0]) is None


def test_paired_comparison():
    cases = [
        {
            "runs": [
                {
                    "evaluation": {
                        "response_a": {
                            "quality_score": 80
                        },
                        "response_b": {
                            "quality_score": 90
                        }
                    }
                }
            ]
        },
        {
            "runs": [
                {
                    "evaluation": {
                        "response_a": {
                            "quality_score": 70
                        },
                        "response_b": {
                            "quality_score": 85
                        }
                    }
                }
            ]
        }
    ]

    result = summarize_comparison(cases)

    assert result["case_wins_a"] == 0
    assert result["case_wins_b"] == 2
    assert result["win_rate_b_percent"] == 100
    assert (
        result["mean_quality_difference_b_minus_a"]
        == 12.5
    )