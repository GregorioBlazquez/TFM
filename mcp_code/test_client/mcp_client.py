import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from fastmcp import Client
import asyncio

load_dotenv()

# Configuración
AZ_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZ_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT").rstrip("/")
AZ_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZ_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8080"))
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
        "Eres un orquestador MCP. Cuando el usuario pregunte, debes elegir la herramienta "
        "entre las disponibles: ['tourism_tool'] y devolver únicamente un JSON con la forma:\n"
        '{"tool": "tourism_tool", "args": {"comunidad": "<string>", "periodo": "<string>"}}\n'
        "Donde 'comunidad' es el nombre de la región (ejemplo 'Cataluña') y 'periodo' el año y mes en formato 'AAAAMM' (ejemplo '202508').\n"
        "Los valores dentro de args deben ser strings. Responde SOLO con ese JSON, sin explicaciones.\n"
        "Si no sabes qué responder, devuelve un JSON vacío '{}'."
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
    print("Cliente MCP interactivo. Escribe tu consulta (ej: '¿Cuántos turistas se esperan en Cataluña en agosto de 2025?')")
    user_q = input("Pregunta> ").strip()
    if not user_q:
        print("No has escrito nada. Saliendo.")
        return

    try:
        decision = ask_llm_for_tool(user_q)
        print("Decisión LLM:", decision)
    except Exception as e:
        print("Error al pedir decisión al LLM:", e)
        return

    tool = decision.get("tool")
    args = decision.get("args", {})

    if not tool:
        print("LLM no devolvió 'tool'. Salir.")
        return

    print(f"Llamando al MCP tool '{tool}' con args: {args}")

    # Usar contexto async para abrir cliente MCP
    async with client_mcp:
        try:
            result = await client_mcp.call_tool(tool, args)
            print("Resultado MCP:")
            print(result)
        except Exception as e:
            print("Error invocando el MCP tool:", e)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
