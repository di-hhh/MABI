"""LLM client wrapper — OpenAI-compatible SDK calling DeepSeek."""
import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_client = None


def get_client() -> OpenAI:
    """Return the OpenAI-compatible client singleton."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL_OPENAI"),
        )
    return _client


def chat(
    messages: list,
    model: str = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: int = 60,
    max_retries: int = 3,
    json_mode: bool = False,
) -> str:
    """Send a chat completion request with retry/backoff.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model: Model name (defaults to DEEPSEEK_V4_FLASH from env).
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on transient failures.
        json_mode: If True, use response_format json_object for valid JSON output.

    Returns:
        The model's text response.
    """
    if model is None:
        model = os.getenv("DEEPSEEK_V4_FLASH", "deepseek-v4-pro")

    client = get_client()
    last_error = None

    kwargs = dict(
        model=model, messages=messages,
        temperature=temperature, max_tokens=max_tokens,
        timeout=timeout,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            wait = 2 ** attempt
            logger.warning("LLM call attempt %d failed: %s. Retrying in %ds...", attempt + 1, e, wait)
            time.sleep(wait)

    logger.error("LLM call failed after %d retries: %s", max_retries, last_error)
    raise last_error
