"""Entry point for the deep researcher application."""

from pathlib import Path

from dotenv import load_dotenv

from src.coordinator import run_deep_research

OUTPUT_FILE = Path("research_result.md")


def main() -> None:
    """Run the deep research application."""
    load_dotenv()

    user_query = input("Enter your research query: ").strip()
    if not user_query:
        print("No query provided. Exiting.")
        return

    result = run_deep_research(user_query)
    OUTPUT_FILE.write_text(result, encoding="utf-8")
    print(f"Research result saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
