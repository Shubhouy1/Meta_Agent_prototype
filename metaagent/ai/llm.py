"""The single place where Gemini clients are created.

Planner, tool selector, code generator, RAG and the runtime proxy all get their
clients from here so model, timeout and retry behaviour stay consistent.
Clients are created lazily, so importing MetaAgent never requires an API key.
"""

from typing import Any, Optional

from metaagent.core.config import Settings, get_settings


class ModelNotAllowedError(ValueError):
    pass


def resolve_model(model: Optional[str], settings: Optional[Settings] = None) -> str:
    settings = settings or get_settings()
    if not model:
        return settings.default_model
    if model not in settings.available_models:
        raise ModelNotAllowedError(
            f"Model '{model}' is not enabled. Available: {', '.join(settings.available_models)}"
        )
    return model


def chat_model(model: Optional[str] = None, temperature: float = 0.0,
               settings: Optional[Settings] = None):
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = settings or get_settings()
    return ChatGoogleGenerativeAI(
        model=resolve_model(model, settings),
        temperature=temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


def embeddings(settings: Optional[Settings] = None):
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    settings = settings or get_settings()
    return GoogleGenerativeAIEmbeddings(model=settings.embedding_model)


def describe_error(error: Any) -> str:
    """Short, user-facing summary of a provider error (the raw text goes to logs/details)."""
    text = str(error)
    if "RESOURCE_EXHAUSTED" in text or " 429" in text or text.startswith("429"):
        return "Gemini quota or rate limit exceeded (HTTP 429). Wait and retry, or use another model or API key."
    if "API_KEY_INVALID" in text or "API key not valid" in text:
        return "The Gemini API key was rejected. Check GOOGLE_API_KEY in .env."
    if "PERMISSION_DENIED" in text or " 403" in text:
        return "Gemini denied the request (HTTP 403). Check that the API key may use this model."
    if "DefaultCredentialsError" in text or "GOOGLE_API_KEY" in text:
        return "No Gemini credentials found. Set GOOGLE_API_KEY in .env."
    if "DEADLINE_EXCEEDED" in text or "timed out" in text.lower() or "Timeout" in type(error).__name__:
        return "The Gemini request timed out."
    first_line = text.strip().splitlines()[0] if text.strip() else type(error).__name__
    return first_line[:300]


def response_text(content: Any) -> str:
    """Normalise an LLM message's content to plain text.

    Recent langchain-google-genai versions may return a list of content parts
    (dicts with a 'text' key) instead of a string.
    """
    if hasattr(content, "content"):
        content = content.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", content))
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type", "text") == "text" and "text" in part:
                    parts.append(part["text"])
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)
