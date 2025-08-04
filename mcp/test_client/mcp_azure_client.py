

import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import requests

load_dotenv()

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# Azure LLM
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")


def call_mcp_tool(question: str) -> str:
    """Call the MCP server with a question and return the response."""
    try:
        response = requests.post(
            "http://localhost:8080",
            json={"input": question},
            timeout=30
        )
        response.raise_for_status()
        return response.json()["output"]
    except Exception as e:
        return f"[Error contacting MCP server]: {e}"


def call_llm(prompt: str) -> str:
    """Call the Azure LLM with a prompt and return the response."""
    result = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are an expert assistant in tourism in Spain."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=1024
    )
    return result.choices[0].message.content.strip()


if __name__ == "__main__":
    print("🤖 Simplified MCP client active.")
    while True:
        user_query = input("👤 You: ")
        if user_query.lower() in ["exit", "quit", "salir"]:
            break
        context = call_mcp_tool(user_query)
        final_answer = call_llm(f"Based on this information: {context}\n\nAnswer: {user_query}")
        print(f"🧠 Azure LLM: {final_answer}\n")
