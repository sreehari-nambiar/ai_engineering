from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch, RunnableLambda

def main():
    try:
        # Use a standard stable model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- 1. Define Handlers (Prepared for Data) ---
    # NOTE: These now accept the whole 'input_dict' and return a 'dict'.
    # This simulates how a real tool (like an API wrapper) behaves.
    
    def booking_handler(input_dict: dict) -> dict:
        req = input_dict.get('original_request')
        # ... In the future, you would extract dates/locations here ...
        print(f"   --> [BOOKING] Logic running for: {req}")
        
        # Return structured data, not just a string
        return {
            "status": "success",
            "action": "book_flight",
            "data": {"confirmation_id": "BLK-12345", "destination": "London"},
            "final_response": "I have booked your flight to London. ID: BLK-12345"
        }

    def info_handler(input_dict: dict) -> dict:
        req = input_dict.get('original_request')
        print(f"   --> [INFO] Logic running for: {req}")
        
        return {
            "status": "success",
            "action": "search_wiki",
            "data": {"query": req, "result_count": 5},
            "final_response": f"Here is some information found regarding: {req}"
        }

    def unclear_handler(input_dict: dict) -> dict:
        return {
            "status": "error",
            "action": "ask_clarification",
            "final_response": "I am not sure what you want to do. Could you clarify?"
        }

    # --- 2. Router Chain ---
    # Same router logic as before
    coordinator_router_prompt = ChatPromptTemplate.from_messages([
        ("system", "Classify intent: 'booker', 'info', or 'unclear'."),
        ("user", "{request}")
    ])
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

    # --- 3. Branches ---
    # We pass the ENTIRE state (x) to the handler function now.
    branches = {
        "booker": RunnableLambda(booking_handler),
        "info":   RunnableLambda(info_handler),
        "unclear": RunnableLambda(unclear_handler)
    }

    delegation_branch = RunnableBranch(
        (lambda x: "booker" in x['decision'].lower(), branches["booker"]),
        (lambda x: "info" in x['decision'].lower(), branches["info"]),
        branches["unclear"]
    )

    # --- 4. Main Agent ---
    coordinator_agent = (
        {
            "decision": coordinator_router_chain,
            "original_request": lambda x: x['request'],
            # You can pass extra data here in the future:
            "user_id": lambda x: x.get('user_id', 'guest') 
        }
        | delegation_branch
    )

    # --- Execution ---
    print("\n--- Test: Complex Data Passing ---")
    
    # We simulate a request that might have extra metadata attached
    payload = {
        "request": "Book a flight to London.", 
        "user_id": "u_999"
    }
    
    result = coordinator_agent.invoke(payload)
    
    # Now we can access specific parts of the result
    print(f"Action Taken: {result.get('action')}")
    print(f"Booking ID:   {result.get('data', {}).get('confirmation_id')}")
    print(f"Message:      {result.get('final_response')}")

if __name__ == "__main__":
    main()