import os
import json
from config import get_env_var
from openai import AzureOpenAI
from fastmcp import Client
import asyncio


# Configuración
AZ_API_KEY = get_env_var("AZURE_OPENAI_API_KEY")
AZ_ENDPOINT = get_env_var("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZ_DEPLOYMENT = get_env_var("AZURE_OPENAI_DEPLOYMENT")
AZ_API_VERSION = get_env_var("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

MCP_HOST = get_env_var("MCP_HOST", "127.0.0.1")
MCP_PORT = int(get_env_var("MCP_PORT", "8080"))
MCP_BASE = f"http://{MCP_HOST}:{MCP_PORT}/mcp/"

# Clientes
client_openai = AzureOpenAI(
    azure_endpoint=AZ_ENDPOINT,
    api_key=AZ_API_KEY,
    api_version=AZ_API_VERSION,
)

client_mcp = Client(MCP_BASE)


def ask_llm_for_tool(user_question: str) -> dict:
    system_prompt = (
        "You are an MCP orchestrator. When the user asks, you must choose the tool "
        "from the available ones: ['tourism_tool'] and return ONLY a JSON in the form:\n"
        '{"tool": "tourism_tool", "args": {"comunidad": "<string>", "periodo": "<string>"}}\n'
        "Where 'comunidad' is the name of the region (example 'Cataluña') and 'periodo' is the year and month in 'AAAAMM' format (example '202508').\n"
        "The values inside args must be strings. Respond ONLY with that JSON, no explanations.\n"
        "If you don't know what to answer, return an empty JSON '{}'."
    )


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]

    resp = client_openai.chat.completions.create(
        model=AZ_DEPLOYMENT,
        messages=messages,
        max_completion_tokens=400
    )

    text = resp.choices[0].message.content.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON encontrado en la respuesta del modelo:\n{text}")

    json_str = text[start : end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parseando JSON: {e}\nTexto JSON: {json_str}")

    return parsed


async def main_async():
    print("Interactive MCP client. Type your query (e.g.: 'How many tourists are expected in Cataluña in August 2025?')")
    user_q = input("Question> ").strip()
    if not user_q:
        print("You didn't type anything. Exiting.")
        return

    try:
        decision = ask_llm_for_tool(user_q)
        print("LLM decision:", decision)
    except Exception as e:
        print("Error requesting decision from LLM:", e)
        return

    tool = decision.get("tool")
    args = decision.get("args", {})

    if not tool:
        print("LLM did not return 'tool'. Exiting.")
        return

    print(f"Calling MCP tool '{tool}' with args: {args}")

    # Use async context to open MCP client
    async with client_mcp:
        try:
            result = await client_mcp.call_tool(tool, args)
            print("MCP result:")
            print(result)
        except Exception as e:
            print("Error invoking MCP tool:", e)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
