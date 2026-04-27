import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_answer(context, query):
    prompt = f"""
You are a strict AI assistant.

RULES:
1. Answer ONLY from the given context.
2. If the answer is not clearly present in the context, reply EXACTLY:
   "No relevant data found"
3. Do NOT guess or add extra information.

Context:
{context}

Question:
{query}
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": "phi3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"].strip()