from promptab.evaluator import (
    calculate_quality_score,
    calculate_criteria_score
)


def test_weighted_quality():
    evaluation = {
        "relevance": 90,
        "instruction_following": 75,
        "completeness": 70,
        "clarity": 85
    }

    assert calculate_quality_score(
        evaluation
    ) == 79.75


def test_criteria_score():
    evaluation = {
        "criteria_results": [
            {"passed": True},
            {"passed": False},
            {"passed": True},
            {"passed": False}
        ]
    }

    assert calculate_criteria_score(
        evaluation
    ) == 50.0


def test_no_criteria():
    evaluation = {
        "criteria_results": []
    }

    assert calculate_criteria_score(
        evaluation
    ) is None


def test_perfect_quality():
    evaluation = {
        "relevance": 100,
        "instruction_following": 100,
        "completeness": 100,
        "clarity": 100
    }

    assert calculate_quality_score(
        evaluation
    ) == 100.0