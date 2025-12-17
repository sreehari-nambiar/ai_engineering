import uuid
import asyncio
from typing import Dict, Any, Optional
from google.adk.agents import Agent 
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.events import Event 

import gc
import os
from dotenv import load_dotenv
import logging

# Filter out the specific noisy error
logging.getLogger("asyncio").setLevel(logging.CRITICAL)

# Load environment variables
load_dotenv(override=True)

# specific safety check to ensure key exists
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- Define Function Tools ---
def booking_handler(request: str) -> str:
    """
    Handles booking requests for flights and hotels.
    Args:
        request: The user's request for a booking.
    Returns:
        A confirmation message that the booking was handled
    """
    print("BOOKING HANDLER WAS CALLED")
    return f"Boking action for '{request} has been simulated"

def info_handler(request: str) -> str:
    """
    Handles general information requests
    Args:
        request: The user's question
    Returns:
        A mesage indicating the information request was handled
    """
    print("INFO HANDLER WAS CALLED")
    return f"Information request for '{request}'. Result: Simulated information retrieval"

def unclear_handler(request: str) -> str:
    """
    Handles reques that could not be delegated
    """
    return f"Coordinator could not delegate request '{request}'. Plesae clarify"

# --- Create Tools from functions ---
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)

# Define specialized sub-agents equipped with their respectie tools
booking_agent = Agent(
    name = "Booker",
    model = "gemini-2.0-flash",
    description = "A specialized agent that handles all flight and hotel booking request by calling the booking tool",
    tools = [booking_tool]
)

info_agent = Agent(
    name = "Info",
    model = "gemini-2.0-flash",
    description = "A specialized agent that provides general information and aswers user questions by calling the info tool.",
    tools = [info_tool]
)

# Parent agent
coordinator = Agent(
    name = "Coordinator",
    model = "gemini-2.0-flash",
    instruction = (
        "You are the main coordinator. Your only task is to analyze the incoming user requests "
        " and delegate them to the appropriate specialized agent. Do not try to answer the user directly.\n"
        "- For any request related to booking flights or hotels, delegate to the 'Booker' agent. \n"
        "- For all other general information questions, delegate to the 'Info agent."
    ),
    description = "A coordonator that routes user request to the correct specialized agent"
)

sub_agents = [booking_agent, info_agent]

# --- Execution Logic
async def run_coordinator(runner: InMemoryRunner, request: str):
    print(f"\n --- Running the coordinator with request: {request}")
    final_result = ""
    try:
        user_id = "user_123"
        session_id = str(uuid.uuid4())
        await runner.session_service.create_session(
            app_name = runner.app_name,
            user_id = user_id,
            session_id = session_id
        )
        for event in runner.run(
            user_id = user_id,
            session_id = session_id,
            new_message = types.Content(
                role = 'user',
                parts = [types.Part(text=request)]
            ),
        ):
            # print(event)
            if event.is_final_response() and event.content:
                if hasattr(event.content, 'text') and event.content.text:
                    final_result = event.content.text
                elif event.content.parts:
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                    break
        print(f"Coordinator Final Response: {final_result}")
        return final_result
    except Exception as e:
        print(f"An error occured while processing your request: {e}")
        return f"An error occured while processing your request: {e}"

async def main():
    runner = InMemoryRunner(coordinator)

    try:
        result_a = await run_coordinator(runner, "Book me a hotel in Paris.")
        print(f"Final Output A: {result_a}")

        result_b = await run_coordinator(runner, "What is the highest mountain in the world?")
        print(f"Final Output B: {result_b}")

        result_c = await run_coordinator(runner, "Tell me a random fact.") # Should go to Info
        print(f"Final Output C: {result_c}")

        result_d = await run_coordinator(runner, "Book flights to Tokyo next month") # Should go to Booker
        print(f"Final Output D: {result_d}")
    finally:
        # 1. Delete the object holding the connection
        del runner 
        
        # 2. Force Python to clean it up NOW (triggering the .aclose call)
        gc.collect() 
        
        # 3. Give the background cleanup task a split second to finish
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())