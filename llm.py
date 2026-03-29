from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_reply(history: list[dict]) -> str:
    if not history:
        return "I didn't catch that, could you say it again?"

    messages = [
        {"role": "system", "content": "You are a helpful friendly voice assistant. Keep responses short, 1 to 3 sentences. Never use bullet points or markdown."}
    ]

    for msg in history:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM failed: {e}")
        return "Something went wrong."