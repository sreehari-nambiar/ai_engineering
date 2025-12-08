from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv(override=True)

# specific safety check to ensure key exists
google_api_key = os.getenv("GOOGLE_API_KEY")
try:
    llm = ChatGoogleGenerativeAI(api_key=google_api_key, model="gemini-2.5-flash", temperature=0)
    print(f"Language model initialized: {llm.model}")
except Exception as e:
    print(f"Error initializing language model: {e}")
    llm = None

# --- Define Sub-Agent Handlers ---
def booking_handler(request_text: str) -> str:
    """ Simulates the Booking agent handling a request"""
    print(f"   --> [BOOKING AGENT] processing: '{request_text}'")
    return "SUCCESS: Flight/Hotel booking simulation complete."

def info_handler(request_text: str) -> str:
    """ Simulates the Info agent handling a request"""
    print(f"   --> [INFO AGENT] processing: '{request_text}'")
    return "SUCCESS: Information retrieved."

def unclear_handler(request_text: str) -> str:
    """ Handles requests that couldn't be delegated"""
    print(f"   --> [UNCLEAR AGENT] processing: '{request_text}'")
    return "FAILURE: Could not determine intent."

# --- Define Coordinator Router Chain
# This chain decides which handler to delegate to.
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the user's request and determine which speciaist handler should process it.
        - If the request is related to booking flights or hotels, output 'booker'.
        - For all other general information questions, output 'info'.
        - If the request is unclear or doesn't fit either category, output 'unclear'.
        ONLY output one word: 'booker', 'info' or 'unclear'."""),
    ("user", "{request}")
])

if llm:
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

# Define delegation logic

# Define Branches
branches = {
    "booker": RunnablePassthrough.assign(output = lambda x : booking_handler(x['request']['request'])),
    "info" : RunnablePassthrough.assign(output = lambda x :  info_handler(x['request']['request'])),
    "unclear" : RunnablePassthrough.assign(output = lambda x : unclear_handler(x['request']['request']))
}
# 'x' here is the dictionary passed from the previous step: {"decision": "...", "original_request": "..."}
# branches = {
#     "booker": RunnableLambda(lambda x: booking_handler(x['original_request'])),
#     "info":   RunnableLambda(lambda x: info_handler(x['original_request'])),
#     "unclear": RunnableLambda(lambda x: unclear_handler(x['original_request']))
# }

# Create RunnableBranch
delegation_branch = RunnableBranch(
    (lambda x : x['decision'].strip() =="booker", branches["booker"]),
    (lambda x : x['decision'].strip() =="info", branches["info"]),
    branches["unclear"] # default branch for any other output
)
# delegation_branch = RunnableBranch(
#         (lambda x: "booker" in x['decision'].lower(), branches["booker"]),
#         (lambda x: "info" in x['decision'].lower(), branches["info"]),
#         branches["unclear"] # Default fallback
# )

# # Combine router chain and delegation brranch into single runnable
coordinator_agent = {
    "decision" : coordinator_router_chain,
    "request" : RunnablePassthrough()
} | delegation_branch | (lambda x : x['output'])
# --- Assemble the Final Chain ---
    # 1. Prepare inputs: Run the router to get the decision, but KEEP the original request
    # 2. Pass that combined data to the delegation_branch
# coordinator_agent = (
#     {
#         "decision": coordinator_router_chain,
#         "original_request": lambda x: x['request'] # Flattens input so we don't get x['request']['request']
#     }
#     | delegation_branch
# )

def main():
    # print("Hello from routing!")
    if not llm:
        print("\nSkipping execution due to the LLM Initialization failure")
        return
    
    print("--- Running with a booking request ---")
    request_a = "Book a flight to London."
    result_a = coordinator_agent.invoke({"request" : request_a})
    print(f"Final Result A: {result_a}")

    print("--- Running with a Info request ---")
    request_b = "What is the capital of Italy?"
    result_b = coordinator_agent.invoke({"request" : request_b})
    print(f"Final Result B: {result_b}")

    print("--- Running with a Unclear request ---")
    request_c = "Tell me about quantum physics."
    result_c = coordinator_agent.invoke({"request" : request_c})
    print(f"Final Result C: {result_c}")


if __name__ == "__main__":
    main()
