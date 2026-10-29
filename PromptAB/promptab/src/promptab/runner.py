
import time


def run_prompt(
    client,
    system_prompt,
    query,
    model,
    temperature=None
):
    params = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": query
            }
        ]
    }

    if temperature is not None:
        params["temperature"] = temperature

    start = time.perf_counter()

    response = client.chat.completions.create(
        **params
    )

    elapsed = time.perf_counter() - start
    usage = response.usage

    return {
        "output": (
            response.choices[0].message.content or ""
        ),
        "latency_seconds": elapsed,
        "input_tokens": (
            usage.prompt_tokens if usage else None
        ),
        "output_tokens": (
            usage.completion_tokens if usage else None
        ),
        "total_tokens": (
            usage.total_tokens if usage else None
        ),
        "model": response.model
    }
