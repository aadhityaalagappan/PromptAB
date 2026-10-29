import json

import pytest

from promptab.dataset import (
    single_case,
    load_dataset
)


def test_single_case():
    cases = single_case(
        query="My order is damaged.",
        expected="Ask for the order number.",
        criteria=["Provide next steps"]
    )

    assert len(cases) == 1
    assert cases[0]["query"] == "My order is damaged."
    assert cases[0]["expected"] == (
        "Ask for the order number."
    )
    assert cases[0]["criteria"] == [
        "Provide next steps"
    ]


def test_dataset_loading(tmp_path):
    path = tmp_path / "cases.json"

    path.write_text(
        json.dumps([
            {
                "query": "Where is my order?",
                "category": "delivery"
            }
        ]),
        encoding="utf-8"
    )

    cases = load_dataset(path)

    assert len(cases) == 1
    assert cases[0]["category"] == "delivery"
    assert cases[0]["expected"] is None


def test_empty_dataset(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        load_dataset(path)


def test_blank_query():
    with pytest.raises(ValueError):
        single_case("   ")