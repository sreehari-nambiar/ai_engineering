import os
from typing import Any, Iterator

from huggingface_hub import InferenceClient

from src.prompt import PLANNER_SYSTEM_INSTRUCTIONS

MODEL_ID = "moonshotai/Kimi-K2-Thinking"


def _extract_content(response_obj: Any) -> str | None:
    """Extract content from a streaming chunk or complete response."""
    if hasattr(response_obj, "choices") and response_obj.choices:
        choice = response_obj.choices[0]
        if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
            return choice.delta.content
        if hasattr(choice, "message") and hasattr(choice.message, "content"):
            return choice.message.content
    return None


def _process_streaming_response(completion: Iterator[Any]) -> str:
    """Process a streaming response and return concatenated content."""
    chunks: list[str] = []
    for chunk in completion:
        content = _extract_content(chunk)
        if content:
            chunks.append(content)
            print(content, end="", flush=True)
    print()
    return "".join(chunks)


def _process_complete_response(completion: Any) -> str:
    """Process a non-streaming response and return content."""
    content = _extract_content(completion)
    if content:
        print(content)
        return content
    return ""


def generate_research_plan(query: str) -> str:
    """Generate a research plan for the given query.

    Args:
        query: The research question or topic to plan for.

    Returns:
        A string containing the generated research plan.

    Raises:
        KeyError: If HF_TOKEN environment variable is not set.
    """
    print(f"User request: {query}")

    client = InferenceClient(api_key=os.environ["HF_TOKEN"])

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    print("Generated Research Plan:")

    try:
        return _process_streaming_response(completion)
    except TypeError:
        return _process_complete_response(completion)