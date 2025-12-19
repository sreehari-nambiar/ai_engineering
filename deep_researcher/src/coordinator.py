"""Coordinator module for orchestrating deep research with sub-agents."""

import json
import os

from smolagents import InferenceClientModel, MCPClient, ToolCallingAgent, tool

from src.planner import generate_research_plan
from src.prompt import COORDINATOR_PROMPT_TEMPLATE, SUBAGENT_PROMPT_TEMPLATE
from src.task_splitter import split_into_subtasks

COORDINATOR_MODEL_ID = "MiniMaxAI/MiniMax-M1-80k"
SUBAGENT_MODEL_ID = "MiniMaxAI/MiniMax-M1-80k"


def _get_mcp_url() -> str:
    """Build the MCP URL from the Firecrawl API key."""
    api_key = os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        raise KeyError("FIRECRAWL_API_KEY environment variable is not set")
    return f"https://mcp.firecrawl.dev/{api_key}/v2/mcp"


def _get_hf_token() -> str:
    """Get the HuggingFace token from environment."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise KeyError("HF_TOKEN environment variable is not set")
    return token


def _create_model(model_id: str, api_key: str) -> InferenceClientModel:
    """Create an InferenceClientModel with the given configuration."""
    return InferenceClientModel(model_id=model_id, api_key=api_key)


def run_deep_research(query: str) -> str:
    """Run deep research on a query using a coordinator and sub-agents.

    Args:
        query: The research question or topic to investigate.

    Returns:
        A comprehensive markdown report synthesizing all sub-agent findings.

    Raises:
        KeyError: If required environment variables are not set.
    """
    print("*** Running the deep research ***")
    research_plan = generate_research_plan(query)
    subtasks = split_into_subtasks(research_plan)

    print("*** Initializing Coordinator ***")

    hf_token = _get_hf_token()
    coordinator_model = _create_model(COORDINATOR_MODEL_ID, hf_token)
    subagent_model = _create_model(SUBAGENT_MODEL_ID, hf_token)
    mcp_url = _get_mcp_url()

    with MCPClient({"url": mcp_url, "transport": "streamable-http"}) as mcp_tools:

        # ---- Initialize Subagent TOOL --------------------------------------
        @tool
        def initialize_subagent(subtask_id: str, subtask_title: str, subtask_description: str) -> str:
            """
            Spawn a dedicated research sub-agent for a single subtask.

            Args:
                subtask_id (str): The unique identifier for the subtask.
                subtask_title (str): The descriptive title of the subtask.
                subtask_description (str): Detailed instructions for the sub-agent to perform the subtask.

            The sub-agent:
            - Has access to the Firecrawl MCP tools.
            - Must perform deep research ONLY on this subtask.
            - Returns a structured markdown report with:
              - a clear heading identifying the subtask,
              - a narrative explanation,
              - bullet-point key findings,
              - explicit citations / links to sources.
            """
            print(f"Initializing Subagent for task {subtask_id}...")

            subagent = ToolCallingAgent(
                tools=mcp_tools,                # Firecrawl MCP toolkit
                model=subagent_model,
                add_base_tools=False,
                name=f"subagent_{subtask_id}",
            )

            subagent_prompt = SUBAGENT_PROMPT_TEMPLATE.format(
                user_query=query,
                research_plan=research_plan,
                subtask_id=subtask_id,
                subtask_title=subtask_title,
                subtask_description=subtask_description,
            )

            return subagent.run(subagent_prompt)
        
        # ---- Coordinator agent ---------------------------------------------
        coordinator = ToolCallingAgent(
            tools=[initialize_subagent],
            model=coordinator_model,
            add_base_tools=False,
            name="coordinator_agent",
        )

        subtasks_json = json.dumps(
            [task.model_dump() for task in subtasks],
            indent=2,
            ensure_ascii=False,
        )

        coordinator_prompt = COORDINATOR_PROMPT_TEMPLATE.format(
            user_query=query,
            research_plan=research_plan,
            subtasks_json=subtasks_json,
        )

        final_report = coordinator.run(coordinator_prompt)
        return final_report