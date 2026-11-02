"""Groq API wrapper for entity extraction and question answering."""

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL

_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set in the project .env file")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def ask_llm(
    system: str,
    user: str,
    max_tokens: int = 1500,
    json_mode: bool = False,
) -> str:
    options = {}
    if json_mode:
        options["response_format"] = {"type": "json_object"}

    response = client().chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **options,
    )
    return response.choices[0].message.content or ""
