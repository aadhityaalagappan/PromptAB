
import json
from pathlib import Path

from jinja2 import Environment, BaseLoader


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<title>PromptAB Report</title>

<style>
body {
    font-family: Arial, sans-serif;
    max-width: 1150px;
    margin: 30px auto;
    padding: 20px;
    background: #f8fafc;
    color: #172033;
    line-height: 1.5;
}
h1, h2, h3 { color: #101828; }
.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 22px;
    margin: 16px 0;
}
.grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 16px;
}
table {
    width: 100%;
    border-collapse: collapse;
}
th, td {
    padding: 10px;
    border-bottom: 1px solid #e2e8f0;
    text-align: left;
}
th { background: #f1f5f9; }
pre {
    background: #f1f5f9;
    padding: 14px;
    border-radius: 8px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}
small { color: #64748b; }
.pass { color: #15803d; }
.fail { color: #b91c1c; }
@media(max-width: 700px) {
    .grid { grid-template-columns: 1fr; }
}
</style>
</head>

<body>

<h1>PromptAB Evaluation Report</h1>
<p>
Model: {{ model }} |
Judge: {{ judge_model }} |
Cases: {{ case_count }} |
Runs per case: {{ runs_per_case }}
</p>

<div class="grid">
<div class="card">
<h3>Prompt A</h3>
<h2>{{ quality_a.avg_quality }}/100</h2>
<small>Average judged quality</small>
</div>
<div class="card">
<h3>Prompt B</h3>
<h2>{{ quality_b.avg_quality }}/100</h2>
<small>Average judged quality</small>
</div>
</div>

<div class="card">
<h2>Overall Quality and Performance</h2>
<table>
<tr>
<th>Metric</th><th>Prompt A</th><th>Prompt B</th>
</tr>
<tr>
<td>Average quality</td>
<td>{{ quality_a.avg_quality }}</td>
<td>{{ quality_b.avg_quality }}</td>
</tr>
<tr>
<td>Average criteria satisfaction</td>
<td>
{{ quality_a.avg_criteria_score
if quality_a.avg_criteria_score is not none else "N/A" }}
</td>
<td>
{{ quality_b.avg_criteria_score
if quality_b.avg_criteria_score is not none else "N/A" }}
</td>
</tr>
<tr>
<td>Quality standard deviation</td>
<td>
{{ quality_a.quality_stddev
if quality_a.quality_stddev is not none else "N/A" }}
</td>
<td>
{{ quality_b.quality_stddev
if quality_b.quality_stddev is not none else "N/A" }}
</td>
</tr>
<tr>
<td>Average latency (seconds)</td>
<td>{{ "%.3f"|format(performance_a.avg_latency_seconds) }}</td>
<td>{{ "%.3f"|format(performance_b.avg_latency_seconds) }}</td>
</tr>
<tr>
<td>Total input tokens</td>
<td>
{{ performance_a.total_input_tokens
if performance_a.total_input_tokens is not none else "N/A" }}
</td>
<td>
{{ performance_b.total_input_tokens
if performance_b.total_input_tokens is not none else "N/A" }}
</td>
</tr>
<tr>
<td>Total output tokens</td>
<td>
{{ performance_a.total_output_tokens
if performance_a.total_output_tokens is not none else "N/A" }}
</td>
<td>
{{ performance_b.total_output_tokens
if performance_b.total_output_tokens is not none else "N/A" }}
</td>
</tr>
<tr>
<td>Total tokens</td>
<td>
{{ performance_a.total_tokens
if performance_a.total_tokens is not none else "N/A" }}
</td>
<td>
{{ performance_b.total_tokens
if performance_b.total_tokens is not none else "N/A" }}
</td>
</tr>
<tr>
<td>Estimated inference cost</td>
<td>
{{ "$%.6f"|format(performance_a.estimated_cost_usd)
if performance_a.estimated_cost_usd is not none else "N/A" }}
</td>
<td>
{{ "$%.6f"|format(performance_b.estimated_cost_usd)
if performance_b.estimated_cost_usd is not none else "N/A" }}
</td>
</tr>
</table>
</div>

<div class="card">
<h2>Paired Comparison</h2>
<p>
Mean quality difference (B - A):
<strong>
{{ comparison.mean_quality_difference_b_minus_a }}
</strong>
</p>
<p>
Case wins:
A={{ comparison.case_wins_a }},
B={{ comparison.case_wins_b }},
Ties={{ comparison.case_ties }}
</p>
<p>
Prompt B case win rate:
{{ comparison.win_rate_b_percent }}%
</p>

{% if comparison.paired_confidence_interval %}
<p>
Approximate paired bootstrap interval:
{{ comparison.paired_confidence_interval.lower }}
to
{{ comparison.paired_confidence_interval.upper }}
</p>
{% else %}
<p>
Confidence interval unavailable for one case.
</p>
{% endif %}

<small>
The interval is based on the evaluated cases and
LLM-judge scores, not verified universal accuracy.
</small>
</div>

<div class="card">
<h2>Performance by Category</h2>
<table>
<tr>
<th>Category</th>
<th>Cases</th>
<th>Prompt A</th>
<th>Prompt B</th>
</tr>
{% for name, item in categories.items() %}
<tr>
<td>{{ name }}</td>
<td>{{ item.cases }}</td>
<td>{{ item.avg_quality_a }}</td>
<td>{{ item.avg_quality_b }}</td>
</tr>
{% endfor %}
</table>
</div>

<h2>Individual Test Cases</h2>

{% for case in cases %}
<div class="card">
<h3>Case {{ case.case_id }} — {{ case.category }}</h3>

<p><strong>User query:</strong> {{ case.query }}</p>

{% if case.expected %}
<p><strong>Expected behavior:</strong> {{ case.expected }}</p>
{% endif %}

{% for run in case.runs %}
<h3>Run {{ run.run }}</h3>

<div class="grid">
{% for version in ["a", "b"] %}
{% set response = run["response_" ~ version] %}
{% set evaluation = run.evaluation["response_" ~ version] %}

<div>
<h3>Prompt {{ version|upper }} Output</h3>
<pre>{{ response.output }}</pre>

<p>
<strong>Quality:</strong>
{{ evaluation.quality_score }}/100
</p>

<p>
<strong>Criteria satisfaction:</strong>
{{ evaluation.criteria_score
if evaluation.criteria_score is not none else "N/A" }}
</p>

<p>
<strong>Expected alignment:</strong>
{{ evaluation.expected_alignment }}
</p>

<p><strong>Judge reasoning:</strong></p>
<p>{{ evaluation.reasoning }}</p>

<h4>Criterion-level evaluation</h4>
<ul>
{% for item in evaluation.criteria_results %}
<li>
<span class="{{ 'pass' if item.passed else 'fail' }}">
{{ "PASS" if item.passed else "FAIL" }}
</span>
— {{ item.criterion }}
<p>{{ item.explanation }}</p>
</li>
{% endfor %}
</ul>

<h4>Performance</h4>
<p>
Input tokens:
{{ response.input_tokens
if response.input_tokens is not none else "N/A" }}
</p>
<p>
Output tokens:
{{ response.output_tokens
if response.output_tokens is not none else "N/A" }}
</p>
<p>
Total tokens:
{{ response.total_tokens
if response.total_tokens is not none else "N/A" }}
</p>
<p>
Latency:
{{ "%.3f"|format(response.latency_seconds) }}s
</p>
</div>
{% endfor %}
</div>

<p>
<strong>Judge preference:</strong>
{{ run.evaluation.preferred }}
</p>
<p>{{ run.evaluation.comparison_reasoning }}</p>

<hr>
{% endfor %}
</div>
{% endfor %}

<div class="card">
<h2>Suggested Improved System Prompt</h2>
<pre>{{ suggestion.suggested_system_prompt }}</pre>

<h3>Suggested improvements</h3>
<ul>
{% for item in suggestion.improvements %}
<li>{{ item }}</li>
{% endfor %}
</ul>

<h3>Reasoning</h3>
<p>{{ suggestion.reasoning }}</p>

<small>
The suggested prompt is an untested candidate.
Evaluate it on held-out test cases before adoption.
</small>
</div>

<div class="card">
<h2>Experiment Cost</h2>
<p>
Judge input tokens:
{{ judge_usage.input_tokens
if judge_usage.input_tokens is not none else "N/A" }}
</p>
<p>
Judge output tokens:
{{ judge_usage.output_tokens
if judge_usage.output_tokens is not none else "N/A" }}
</p>
<p>
Judge estimated cost:
{{ "$%.6f"|format(judge_cost_usd)
if judge_cost_usd is not none else "N/A" }}
</p>
<p>
Generated-test cost:
{{ "$%.6f"|format(generation_cost_usd)
if generation_cost_usd is not none else "N/A" }}
</p>
<p>
<strong>Total estimated experiment cost:</strong>
{{ "$%.6f"|format(total_estimated_cost_usd)
if total_estimated_cost_usd is not none else "N/A" }}
</p>
<small>
Costs require current pricing for the selected models.
</small>
</div>

</body>
</html>
"""


class ExperimentReport:
    def __init__(self, data):
        self.data = data

    @property
    def suggested_prompt(self):
        return self.data["suggestion"][
            "suggested_system_prompt"
        ]

    def show(self):
        d = self.data

        print("\n" + "=" * 65)
        print("PROMPTAB EVALUATION REPORT")
        print("=" * 65)

        print("Model:", d["model"])
        print("Judge:", d["judge_model"])
        print("Cases:", d["case_count"])
        print("Runs per case:", d["runs_per_case"])

        for case in d["cases"]:
            print("\n" + "-" * 65)
            print("QUERY:", case["query"])
            print("CATEGORY:", case["category"])
            print("EXPECTED:", case["expected"])

            for run in case["runs"]:
                print(f"\nRUN {run['run']}")

                for version in ("a", "b"):
                    response = run[f"response_{version}"]
                    evaluation = run["evaluation"][
                        f"response_{version}"
                    ]

                    print(
                        f"\nPROMPT {version.upper()} OUTPUT:"
                    )
                    print(response["output"])

                    print(
                        "Quality:",
                        evaluation["quality_score"]
                    )
                    print(
                        "Criteria score:",
                        evaluation["criteria_score"]
                    )
                    print(
                        "Expected alignment:",
                        evaluation["expected_alignment"]
                    )

                    for item in evaluation["criteria_results"]:
                        status = (
                            "PASS" if item["passed"] else "FAIL"
                        )
                        print(
                            f"[{status}] {item['criterion']}"
                        )
                        print("  ", item["explanation"])

                    print(
                        "Input tokens:",
                        response["input_tokens"]
                    )
                    print(
                        "Output tokens:",
                        response["output_tokens"]
                    )
                    print(
                        "Total tokens:",
                        response["total_tokens"]
                    )
                    print(
                        "Latency:",
                        round(
                            response["latency_seconds"], 3
                        ),
                        "seconds"
                    )
                    print(
                        "Judge reasoning:",
                        evaluation["reasoning"]
                    )

                print(
                    "\nPreferred:",
                    run["evaluation"]["preferred"]
                )
                print(
                    "Reason:",
                    run["evaluation"]["comparison_reasoning"]
                )

        print("\n" + "=" * 65)
        print("AGGREGATE SUMMARY")
        print("=" * 65)

        for version in ("a", "b"):
            quality = d[f"quality_{version}"]
            performance = d[f"performance_{version}"]

            print(f"\nPROMPT {version.upper()}")
            print(
                "Average quality:",
                quality["avg_quality"]
            )
            print(
                "Quality std. dev.:",
                quality["quality_stddev"]
            )
            print(
                "Average criteria score:",
                quality["avg_criteria_score"]
            )
            print(
                "Total input tokens:",
                performance["total_input_tokens"]
            )
            print(
                "Total output tokens:",
                performance["total_output_tokens"]
            )
            print(
                "Total tokens:",
                performance["total_tokens"]
            )
            print(
                "Average latency:",
                round(
                    performance["avg_latency_seconds"], 3
                )
            )
            print(
                "Estimated inference cost:",
                performance["estimated_cost_usd"]
            )

        print("\nPAIRED COMPARISON")
        print(d["comparison"])

        print("\nCATEGORY PERFORMANCE")
        print(d["categories"])

        print("\nEVALUATION OVERHEAD")
        print("Judge tokens:", d["judge_usage"])
        print("Judge cost:", d["judge_cost_usd"])
        print(
            "Generation cost:",
            d["generation_cost_usd"]
        )
        print(
            "Total estimated experiment cost:",
            d["total_estimated_cost_usd"]
        )

        print("\nSUGGESTED SYSTEM PROMPT")
        print(self.suggested_prompt)

        print("\nSUGGESTION REASONING")
        print(d["suggestion"]["reasoning"])

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_text(
            json.dumps(
                self.data,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        return path

    def to_html(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        environment = Environment(
            loader=BaseLoader(),
            autoescape=True
        )

        template = environment.from_string(
            HTML_TEMPLATE
        )

        path.write_text(
            template.render(**self.data),
            encoding="utf-8"
        )

        return path

    def save_suggested_prompt(self, path):
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        path.write_text(
            self.suggested_prompt,
            encoding="utf-8"
        )

        return path
