import json

from promptab.report import ExperimentReport


def test_suggested_prompt_property():
    report = ExperimentReport({
        "suggestion": {
            "suggested_system_prompt": (
                "You are a helpful assistant."
            )
        }
    })

    assert report.suggested_prompt == (
        "You are a helpful assistant."
    )


def test_save_suggested_prompt(tmp_path):
    report = ExperimentReport({
        "suggestion": {
            "suggested_system_prompt": (
                "You are a support assistant."
            )
        }
    })

    path = tmp_path / "v3.txt"

    report.save_suggested_prompt(path)

    assert path.read_text(
        encoding="utf-8"
    ) == "You are a support assistant."


def test_json_export(tmp_path):
    report = ExperimentReport({
        "suggestion": {
            "suggested_system_prompt": "Improved prompt"
        },
        "quality_a": {
            "avg_quality": 80
        },
        "quality_b": {
            "avg_quality": 90
        }
    })

    path = tmp_path / "results.json"

    report.to_json(path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert data["quality_a"]["avg_quality"] == 80
    assert data["quality_b"]["avg_quality"] == 90