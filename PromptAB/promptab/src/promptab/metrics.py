
import random

from collections import defaultdict
from statistics import mean, stdev


def calculate_cost(
    input_tokens,
    output_tokens,
    pricing=None
):
    if (
        pricing is None
        or input_tokens is None
        or output_tokens is None
    ):
        return None

    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


def summarize_performance(runs, pricing=None):
    if not runs:
        raise ValueError("Cannot summarize empty runs.")

    usage_available = all(
        run["input_tokens"] is not None
        and run["output_tokens"] is not None
        for run in runs
    )

    input_tokens = (
        sum(run["input_tokens"] for run in runs)
        if usage_available else None
    )

    output_tokens = (
        sum(run["output_tokens"] for run in runs)
        if usage_available else None
    )

    total_tokens = (
        input_tokens + output_tokens
        if usage_available else None
    )

    return {
        "requests": len(runs),
        "avg_latency_seconds": mean(
            run["latency_seconds"] for run in runs
        ),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_request": (
            total_tokens / len(runs)
            if total_tokens is not None else None
        ),
        "estimated_cost_usd": calculate_cost(
            input_tokens,
            output_tokens,
            pricing
        )
    }


def summarize_quality(evaluations, version):
    records = [
        evaluation[version]
        for evaluation in evaluations
    ]

    scores = [
        record["quality_score"]
        for record in records
    ]

    criteria_scores = [
        record["criteria_score"]
        for record in records
        if record["criteria_score"] is not None
    ]

    dimensions = (
        "relevance",
        "instruction_following",
        "completeness",
        "clarity"
    )

    return {
        "avg_quality": round(mean(scores), 2),
        "quality_stddev": (
            round(stdev(scores), 2)
            if len(scores) > 1 else None
        ),
        "avg_criteria_score": (
            round(mean(criteria_scores), 2)
            if criteria_scores else None
        ),
        "dimensions": {
            dimension: round(
                mean(record[dimension] for record in records),
                2
            )
            for dimension in dimensions
        }
    }


def summarize_categories(cases):
    groups = defaultdict(list)

    for case in cases:
        groups[case["category"]].append(case)

    summary = {}

    for category, items in groups.items():
        scores_a = []
        scores_b = []

        for case in items:
            for run in case["runs"]:
                evaluation = run["evaluation"]

                scores_a.append(
                    evaluation["response_a"]["quality_score"]
                )
                scores_b.append(
                    evaluation["response_b"]["quality_score"]
                )

        summary[category] = {
            "cases": len(items),
            "avg_quality_a": round(mean(scores_a), 2),
            "avg_quality_b": round(mean(scores_b), 2)
        }

    return summary


def paired_differences(cases):
    """
    Calculate one difference per query.

    Average repeated runs within each query first.
    """
    differences = []

    for case in cases:
        scores_a = [
            run["evaluation"]["response_a"]["quality_score"]
            for run in case["runs"]
        ]

        scores_b = [
            run["evaluation"]["response_b"]["quality_score"]
            for run in case["runs"]
        ]

        differences.append(
            mean(scores_b) - mean(scores_a)
        )

    return differences


def paired_bootstrap_ci(
    differences,
    iterations=2000,
    seed=42
):
    if len(differences) < 2:
        return None

    rng = random.Random(seed)
    sample_count = len(differences)

    bootstrap_means = []

    for _ in range(iterations):
        sampled = rng.choices(
            differences,
            k=sample_count
        )

        bootstrap_means.append(mean(sampled))

    bootstrap_means.sort()

    lower_index = int(0.025 * iterations)
    upper_index = int(0.975 * iterations)

    return {
        "lower": round(
            bootstrap_means[lower_index], 2
        ),
        "upper": round(
            bootstrap_means[upper_index], 2
        )
    }


def summarize_comparison(cases):
    differences = paired_differences(cases)

    wins_a = sum(
        difference < -1e-9
        for difference in differences
    )

    wins_b = sum(
        difference > 1e-9
        for difference in differences
    )

    ties = len(differences) - wins_a - wins_b

    return {
        "mean_quality_difference_b_minus_a": round(
            mean(differences), 2
        ),
        "paired_confidence_interval": (
            paired_bootstrap_ci(differences)
        ),
        "case_wins_a": wins_a,
        "case_wins_b": wins_b,
        "case_ties": ties,
        "win_rate_b_percent": round(
            100 * wins_b / len(differences), 2
        ),
        "case_differences": [
            round(value, 2)
            for value in differences
        ]
    }
