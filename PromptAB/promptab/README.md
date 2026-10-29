# PromptAB

A Python library for evaluating, comparing, and improving
LLM system prompts.

## Features

- Compare two system prompts using the same user query
- Support single-query and dataset evaluations
- Generate test queries automatically
- Display actual responses from both prompts
- Evaluate expected behavior and custom criteria
- Calculate weighted quality scores
- Measure token usage, latency, and estimated costs
- Evaluate repeated runs
- Analyze performance by category
- Generate an improved system prompt
- Export HTML and JSON reports

## Installation

```bash
pip install -e .
```

## Configuration

Set OPENAI_API_KEY in your environment or .env file.

## Single Query

```python
from promptab import compare

report = compare(
    prompt_a="prompts/v1.txt",
    prompt_b="prompts/v2.txt",
    query="My order arrived damaged. Can I get a refund?",
    expected="Acknowledge the issue and verify refund eligibility.",
    criteria=[
        "Acknowledge the concern",
        "Do not guarantee a refund",
        "Provide next steps"
    ]
)

report.show()
report.to_html("report.html")
report.to_json("results.json")
report.save_suggested_prompt("prompts/v3.txt")
```

## Dataset Evaluation

```python
report = compare(
    prompt_a="prompts/v1.txt",
    prompt_b="prompts/v2.txt",
    dataset="examples/cases.json",
    runs=3
)
```

## Automatic Test Generation

```python
report = compare(
    prompt_a="prompts/v1.txt",
    prompt_b="prompts/v2.txt",
    task="Customer support for an e-commerce store",
    generate_cases=20
)
```

## Limitations

Quality scores and criteria satisfaction are LLM-judge assessments.

Suggested prompts are untested candidates.

Cost estimates require current model-specific pricing.

Results apply to the evaluated queries and criteria.

Use representative, held-out test cases before adopting a new prompt.

## License

MIT