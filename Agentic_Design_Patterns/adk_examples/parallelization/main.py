import os
import asyncio
from dotenv import load_dotenv
from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.tools import google_search
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv(override=True)
google_api_key = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash"

# --- 1. SUB AGENTS
# --- RESEARCHER SUB-AGENT 1: RENEWAL ENERGY
researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research assistant specializing in energy.
    Research the latest advancements in 'renewable energy source'.
    Use the Google Search tool provided.
    Summarize your findings concisely (1-2 sentences).
    Output *ONLY* the summary.""",
    description="Researches renewable energy sources",
    tools=[google_search],
    output_key="renewable_energy_result"
)

# --- RESEARCHER SUB-AGENT 2: ELECTRIC VEHICLE
researcher_agent_2 = LlmAgent(
    name="EVResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in transportation.
    Research the latest developments in 'electric vehicle technology'.
    Use the Google Search tool provided.
    Summarize your findings concisely (1-2 sentences).
    Output *ONLY* the summary.""",
    description="Researches electric vehicle technology",
    tools=[google_search],
    output_key="ev_technology_result"
)

# --- RESEARCHER SUB-AGENT 3: CARBON CAPTURE
researcher_agent_3 = LlmAgent(
    name="CarbonCaptureResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Research Assistant specializing in climate situations.
    Research the current state of 'carbon capture methods'.
    Use the google search tool provided.
    Summarize your key findings concisely (1-2 sentences).
    Output *ONLY* the summary.""",
    tools=[google_search],
    output_key="carbon_capture_results"
)

# --- 2. PARALLEL AGENTS - CONCURRENCY
parallel_research_agent = ParallelAgent(
    name="ParallelWebResearchAgent",
    sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
    description="Runs multiple research agents in parallel to gather information"
)

# --- 3. MERGER AGENTS - SYNTHESIZE
merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL,
    instruction="""Your are AI Agent responsible for combining research finginds into a structured report.
    Your primary task is to synthesize the following research summaries, clearly attributing findings to their source areas.
    Structure your response using headings for each topic. Ensure the report is coherent and integrates the key points smoothly.
    **Crucially: Your entire reponse MUST be grounded *exclusively* on the information provided in the 'Input Summaries' below.
    Do NOT add any external knowledge, facts or details not present in these specific summaries.**
    **Input Summaries:**
    *   **Renewable Energy:** {renewable_energy_result}
    *   **Electric Vehicles:** {ev_technology_result}
    *   **Carbon Capture:** {carbon_capture_results}
    **Output Format:**
    ## Summary of Recent Sustainable Technology Advancements
    ### Renewable Energy Findings
    (Based on RenewableEnergyResearcher's findinds)
    [Synthesize and elaborate *ONLY* on the renewable energy input summary provided above.]
    ### Electric Vechicle Findings
    (Based on EVResearcher's findinds)
    [Synthesize and elaborate *ONLY* on the EV input summary provided above.]
    ### Carbon Capture Findings
    (Based on CarbonCaptureResearcher's findinds)
    [Synthesize and elaborate *ONLY* on the carbon capture input summary provided above.]
    ### Overall Conclusion
    [Provide a brief (1-2 sentence) concluding statement that connects *ONLY* the findings presented above.]
    Output *ONLY* the structured report following this format. DO NOT include introductory or concluding phrases outside this structure,
    and strictly adhere to only using the provided input summary content""",
    description="Combines research findings from parallel agents into a structured, cited report, strictly grounded on provided inputs",
)

# --- 4. ORCHESTRATE OVERALL FLOW
sequential_pipeline_agent = SequentialAgent(
    name="ResearchAndSynthesisPipeline",
    sub_agents=[parallel_research_agent, merger_agent],
    description="Coordinates parallel research and synthesizes the results"
)
root_agent = sequential_pipeline_agent

async def main():
    runner = InMemoryRunner(agent=sequential_pipeline_agent, app_name="research_app")
    # Create a session first
    session = await runner.session_service.create_session(
        app_name="research_app",
        user_id="user1",
        session_id="session1"
    )
    async for event in runner.run_async(
        user_id="user1",
        session_id=session.id,
        new_message=types.Content(parts=[types.Part(text="Start research")])
    ):
        print(event)

if __name__ == "__main__":
    asyncio.run(main())
