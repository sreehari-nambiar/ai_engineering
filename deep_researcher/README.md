# Deep Researcher

AI-powered research agent that conducts comprehensive, multi-source research on any topic. Built with [smolagents](https://github.com/huggingface/smolagents) and powered by LLMs.

## How It Works

1. **Planning** - Generates a detailed research plan from your query
2. **Task Splitting** - Breaks the plan into independent subtasks
3. **Parallel Research** - Spawns sub-agents to research each subtask using web scraping (Firecrawl MCP)
4. **Synthesis** - A coordinator agent combines all findings into a comprehensive report

## Installation

```bash
cd deep_researcher

# Install dependencies with uv
uv sync

# Install dev dependencies (for testing)
uv sync --extra dev
```

## Configuration

Create a `.env` file in the `deep_researcher` directory:

```env
HF_TOKEN=your_huggingface_token
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### Required API Keys

| Variable | Description | Get it from |
|----------|-------------|-------------|
| `HF_TOKEN` | HuggingFace API token for LLM inference | [HuggingFace Settings](https://huggingface.co/settings/tokens) |
| `FIRECRAWL_API_KEY` | Firecrawl API key for web scraping | [Firecrawl](https://firecrawl.dev) |

## Usage

```bash
uv run python main.py
```

Enter your research query when prompted. The final report will be saved to `research_result.md`.

## Project Structure

```
deep_researcher/
├── main.py                 # Application entry point
├── src/
│   ├── coordinator.py      # Orchestrates research with sub-agents
│   ├── planner.py          # Generates research plans
│   ├── task_splitter.py    # Splits plans into subtasks
│   └── prompt.py           # Prompt templates
├── tests/
│   ├── test_coordinator.py
│   ├── test_planner.py
│   └── test_task_splitter.py
├── pyproject.toml
└── .env                    # Environment variables (not committed)
```

## Running Tests

```bash
uv run pytest
```
