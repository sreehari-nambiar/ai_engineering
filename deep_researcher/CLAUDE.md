# Deep Researcher

AI-powered research agent built with smolagents.

## Tech Stack

- Python 3.11+
- smolagents (with LiteLLM, MCP, OpenAI integrations)
- python-dotenv for environment management
- uv for package management

## Commands

```bash
# Install dependencies
uv sync

# Run the application
uv run python main.py
```

## Project Structure

```
deep_researcher/
├── main.py           # Entry point
├── pyproject.toml    # Project config and dependencies
├── uv.lock           # Locked dependencies
└── .env              # Environment variables (not committed)
```

## Environment Variables

Create a `.env` file with required API keys:

```
OPENAI_API_KEY=your_key_here
```

## Guidelines

- Use type hints for all functions
- Follow PEP 8 style conventions
- Keep agent logic modular and testable
