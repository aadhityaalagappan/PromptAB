
from pathlib import Path

from promptab import compare


ROOT = Path(__file__).resolve().parents[1]

PROMPT_A = ROOT / "prompts" / "v1.txt"
PROMPT_B = ROOT / "prompts" / "v2.txt"

DATASET = ROOT / "examples" / "indian_cricket_cases.json"


# Example pricing in USD per million tokens.
# Verify current prices for the selected model.
PRICING = {
    "input": 0.15,
    "output": 0.60
}


# ---------------------------------------
# MODE 1: Single-query evaluation
# ---------------------------------------

def single_query_example():

    report = compare(
        prompt_a=str(PROMPT_A),
        prompt_b=str(PROMPT_B),

        query=(
            "Did Kapil Dev score 175 runs "
            "in the 1983 World Cup final?"
        ),

        expected=(
            "No. Kapil Dev scored an unbeaten 175 against "
            "Zimbabwe during the group stage of the 1983 "
            "World Cup. In the final against West Indies, "
            "Kapil Dev scored 15 runs from 8 balls. "
            "India won the final by 43 runs."
),

        criteria=[
            "Correct the false premise",
            "Identify Zimbabwe as the opponent",
            "Mention the 1983 Cricket World Cup",
            "State that Kapil Dev scored 15 runs in the final",
            "Distinguish the Zimbabwe match from the final",
            "Avoid incorrect or fabricated cricket statistics"
        ],

        model="gpt-4o-mini",
        judge_model="gpt-4o-mini",

        runs=1,

        pricing=PRICING,
        judge_pricing=PRICING
    )

    report.show()

    report.to_json(
        ROOT / "cricket_single_results.json"
    )

    report.to_html(
        ROOT / "cricket_single_report.html"
    )

    report.save_suggested_prompt(
        ROOT / "prompts" / "v3_single.txt"
    )


# ---------------------------------------
# MODE 2: Dataset evaluation
# ---------------------------------------

def dataset_example():

    report = compare(
        prompt_a=str(PROMPT_A),
        prompt_b=str(PROMPT_B),

        dataset=str(DATASET),

        model="gpt-4o-mini",
        judge_model="gpt-4o-mini",

        runs=3,

        pricing=PRICING,
        judge_pricing=PRICING
    )

    report.show()

    report.to_json(
        ROOT / "cricket_dataset_results.json"
    )

    report.to_html(
        ROOT / "cricket_dataset_report.html"
    )

    report.save_suggested_prompt(
        ROOT / "prompts" / "v3_dataset.txt"
    )


# ---------------------------------------
# MODE 3: Automatically generated queries
# ---------------------------------------

def generated_cases_example():

    report = compare(
        prompt_a=str(PROMPT_A),
        prompt_b=str(PROMPT_B),

        task=(
            "An Indian cricket AI assistant that answers "
            "questions about Indian cricket history, "
            "legendary players such as Sachin Tendulkar, "
            "Virat Kohli, MS Dhoni, and Kapil Dev, "
            "ICC tournaments, ODI and Test cricket, "
            "T20 Internationals, player comparisons, "
            "cricket statistics, historical matches, "
            "false premises, and uncertain information."
        ),

        generate_cases=10,
        runs=2,

        model="gpt-4o-mini",
        judge_model="gpt-4o-mini",

        pricing=PRICING,
        judge_pricing=PRICING
    )

    report.show()

    report.to_json(
        ROOT / "cricket_generated_results.json"
    )

    report.to_html(
        ROOT / "cricket_generated_report.html"
    )

    report.save_suggested_prompt(
        ROOT / "prompts" / "v3_generated.txt"
    )



if __name__ == "__main__":

    # Mode 1: Single query (4 API calls)
    single_query_example()

    # Mode 2: 10 queries × 3 runs (91 API calls)
    # dataset_example()

    # Mode 3: 10 generated queries × 2 runs
    # generated_cases_example()
