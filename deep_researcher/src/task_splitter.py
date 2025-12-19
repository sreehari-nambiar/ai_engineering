from huggingface_hub import InferenceClient
from .prompt import TASK_SPLITTER_SYSTEM_INSTRUCTIONS

import os
import json
from pprint import pprint
from pydantic import BaseModel, Field

MODEL_ID = "deepseek-ai/DeepSeek-V3.2-Exp"
# MODEL_ID = "moonshotai/Kimi-K2-Thinking"
# PROVIDER = "together"


class Subtask(BaseModel):
    id: str = Field(
        ...,
        description="Short identifier for the subtask (e.g. 'A', 'history', 'drivers').",
    )
    title: str = Field(
        ...,
        description="Short descriptive title of the subtask.",
    )
    description: str = Field(
        ...,
        description="Clear, detailed instructions for the sub-agent that will research this subtask.",
    )


class SubtaskList(BaseModel):
    subtasks: list[Subtask] = Field(
        ...,
        description="List of subtasks that together cover the whole research plan.",
    )


TASK_SPLITTER_JSON_SCHEMA = {
    "name": "subtaskList",
    "schema": SubtaskList.model_json_schema(),
    "strict": True,
}


def split_into_subtasks(research_plan: str, model_id: str = MODEL_ID) -> list[Subtask]:
    """Split a research plan into subtasks using an LLM.

    Args:
        research_plan: The research plan text to split.
        model_id: The model ID to use for inference.

    Returns:
        A list of Subtask objects.

    Raises:
        KeyError: If HF_TOKEN environment variable is not set.
        ValueError: If the API response cannot be parsed.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise KeyError("HF_TOKEN environment variable is not set")

    print("*** Splitting the research plan into subtasks ***")

    client = InferenceClient(api_key=hf_token)

    completion = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": TASK_SPLITTER_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": research_plan},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": TASK_SPLITTER_JSON_SCHEMA,
        },
    )

    message = completion.choices[0].message

    try:
        subtasks_data = json.loads(message.content)["subtasks"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Failed to parse subtasks from response: {e}")

    subtasks = [Subtask(**task) for task in subtasks_data]

    print("***Generated The Following Subtasks***")
    for task in subtasks:
        print(f"{task.title}")
        pprint(f"{task.description}")
        print()

    return subtasks