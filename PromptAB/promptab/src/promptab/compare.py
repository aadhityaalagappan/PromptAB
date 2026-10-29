
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from math import isfinite

from dotenv import load_dotenv
from openai import OpenAI

from .dataset import (
    single_case,
    load_dataset,
    generate_test_cases
)
from .runner import run_prompt
from .evaluator import (
    evaluate_pair,
    suggest_prompt
)
from .metrics import (
    calculate_cost,
    summarize_performance,
    summarize_quality,
    summarize_categories,
    summarize_comparison
)
from .report import ExperimentReport


def validate_pricing(name, pricing):
    if pricing is None:
        return

    if not isinstance(pricing, dict):
        raise ValueError(f"{name} must be a dictionary.")

    for key in ("input", "output"):
        value = pricing.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
        ):
            raise ValueError(
                f"{name}.{key} must be a finite "
                "nonnegative number."
            )


def aggregate_usage(usages):
    if not usages:
        return {
            "input_tokens": 0,
            "output_tokens": 0
        }

    if any(
        usage["input_tokens"] is None
        or usage["output_tokens"] is None
        for usage in usages
    ):
        return {
            "input_tokens": None,
            "output_tokens": None
        }

    return {
        "input_tokens": sum(
            usage["input_tokens"]
            for usage in usages
        ),
        "output_tokens": sum(
            usage["output_tokens"]
            for usage in usages
        )
    }


def compare(
    prompt_a: str,
    prompt_b: str,
    query: str | None = None,
    expected: str | None = None,
    criteria: list[str] | None = None,
    dataset: str | None = None,
    task: str | None = None,
    generate_cases: int = 20,
    runs: int = 1,
    model: str = "gpt-4o-mini",
    judge_model: str = "gpt-4o-mini",
    temperature: float | None = None,
    pricing: dict | None = None,
    judge_pricing: dict | None = None,
    client=None
):
    if runs < 1:
        raise ValueError("runs must be >= 1.")

    modes = sum((
        query is not None,
        dataset is not None,
        task is not None
    ))

    if modes != 1:
        raise ValueError(
            "Provide exactly one of query, dataset, or task."
        )

    if query is None and (
        expected is not None or criteria is not None
    ):
        raise ValueError(
            "For datasets, put expected and criteria "
            "inside each test case."
        )

    validate_pricing("pricing", pricing)
    validate_pricing("judge_pricing", judge_pricing)

    load_dotenv()

    if client is None:
        client = OpenAI()

    text_a = Path(prompt_a).read_text(
        encoding="utf-8"
    )

    text_b = Path(prompt_b).read_text(
        encoding="utf-8"
    )

    if not text_a.strip() or not text_b.strip():
        raise ValueError(
            "System prompt files cannot be empty."
        )

    generation_usage = None

    if query is not None:
        cases = single_case(
            query=query,
            expected=expected,
            criteria=criteria
        )
        source = "query"

    elif dataset is not None:
        cases = load_dataset(dataset)
        source = "dataset"

    else:
        cases, generation_usage = generate_test_cases(
            client=client,
            task=task,
            count=generate_cases,
            model=model
        )
        source = "generated"

    case_results = []

    all_runs_a = []
    all_runs_b = []

    evaluations = []
    judge_usages = []

    suggestion_evidence = []

    for case_index, case in enumerate(cases, start=1):
        print(
            f"Evaluating case {case_index}/{len(cases)}"
        )

        case_runs = []

        for run_number in range(1, runs + 1):
            response_a = run_prompt(
                client=client,
                system_prompt=text_a,
                query=case["query"],
                model=model,
                temperature=temperature
            )

            response_b = run_prompt(
                client=client,
                system_prompt=text_b,
                query=case["query"],
                model=model,
                temperature=temperature
            )

            all_runs_a.append(response_a)
            all_runs_b.append(response_b)

            judged = evaluate_pair(
                client=client,
                prompt_a=text_a,
                prompt_b=text_b,
                query=case["query"],
                output_a=response_a["output"],
                output_b=response_b["output"],
                expected=case["expected"],
                criteria=case["criteria"],
                judge_model=judge_model
            )

            evaluations.append(
                judged["evaluation"]
            )

            judge_usages.append(
                judged["usage"]
            )

            case_runs.append({
                "run": run_number,
                "response_a": response_a,
                "response_b": response_b,
                "evaluation": judged["evaluation"]
            })

        case_results.append({
            "case_id": case_index,
            "query": case["query"],
            "expected": case["expected"],
            "criteria": case["criteria"],
            "category": case["category"],
            "runs": case_runs
        })

        suggestion_evidence.append({
            "query": case["query"],
            "expected": case["expected"],
            "criteria": case["criteria"],
            "category": case["category"],
            "runs": [
                {
                    "response_a": run["response_a"]["output"],
                    "response_b": run["response_b"]["output"],
                    "evaluation": run["evaluation"]
                }
                for run in case_runs
            ]
        })

    quality_a = summarize_quality(
        evaluations,
        "response_a"
    )

    quality_b = summarize_quality(
        evaluations,
        "response_b"
    )

    performance_a = summarize_performance(
        all_runs_a,
        pricing
    )

    performance_b = summarize_performance(
        all_runs_b,
        pricing
    )

    categories = summarize_categories(
        case_results
    )

    comparison = summarize_comparison(
        case_results
    )

    preferences = Counter(
        evaluation["preferred"]
        for evaluation in evaluations
    )

    comparison["judge_preferences"] = {
        "A": preferences["A"],
        "B": preferences["B"],
        "tie": preferences["tie"]
    }

    suggestion_result = suggest_prompt(
        client=client,
        prompt_a=text_a,
        prompt_b=text_b,
        evaluation_evidence=suggestion_evidence,
        judge_model=judge_model
    )

    judge_usages.append(
        suggestion_result["usage"]
    )

    judge_usage = aggregate_usage(
        judge_usages
    )

    judge_cost = calculate_cost(
        judge_usage["input_tokens"],
        judge_usage["output_tokens"],
        judge_pricing
    )

    generation_cost = None

    if generation_usage is not None:
        generation_cost = calculate_cost(
            generation_usage["input_tokens"],
            generation_usage["output_tokens"],
            pricing
        )

    costs = [
        performance_a["estimated_cost_usd"],
        performance_b["estimated_cost_usd"],
        judge_cost
    ]

    if generation_usage is not None:
        costs.append(generation_cost)

    total_cost = (
        sum(costs)
        if all(cost is not None for cost in costs)
        else None
    )

    data = {
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),

        "model": model,
        "judge_model": judge_model,
        "source": source,
        "case_count": len(cases),
        "runs_per_case": runs,

        "prompt_a": text_a,
        "prompt_b": text_b,

        "cases": case_results,

        "quality_a": quality_a,
        "quality_b": quality_b,

        "performance_a": performance_a,
        "performance_b": performance_b,

        "categories": categories,
        "comparison": comparison,

        "suggestion": suggestion_result["suggestion"],

        "judge_usage": judge_usage,
        "judge_cost_usd": judge_cost,

        "generation_usage": generation_usage,
        "generation_cost_usd": generation_cost,

        "total_estimated_cost_usd": total_cost
    }

    return ExperimentReport(data)
