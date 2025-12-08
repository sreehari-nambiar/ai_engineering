import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv(override=True)
    
    # specific safety check to ensure key exists
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY not found.")
        return

    # Initialize the language model
    llm = ChatOpenAI(api_key=openai_api_key, temperature=0)

    # --- Prompt 1: Extract Information ---
    prompt_extract = ChatPromptTemplate.from_template(
        "Extract the technical specifications from the following text: \n\n{text_input}"
    )

    # --- Prompt 2: Transform to JSON ---
    # FIX: Added .from_template() here
    prompt_transform = ChatPromptTemplate.from_template(
        "Transform the following specifications into a JSON object with 'cpu', 'memory' and 'storage' as keys:\n\n{specifications}"
    )

    # --- Build the chain using LCEL ---
    extraction_chain = prompt_extract | llm | StrOutputParser()

    # The result of extraction_chain is passed as "specifications" to prompt_transform
    full_chain = (
        {"specifications": extraction_chain}
        | prompt_transform
        | llm
        | StrOutputParser()
    )

    input_text = "The new laptop model features a 3.5 GHz octa-core processor, 16GB of rAM and a 1 TB NVe SSD."

    # Invoke the chain
    final_result = full_chain.invoke({"text_input": input_text})
    print(final_result)

if __name__ == "__main__":
    main()