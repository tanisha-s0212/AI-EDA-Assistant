from __future__ import annotations

import os
from pathlib import Path


AGENTIC_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = AGENTIC_ROOT.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(AGENTIC_ROOT / ".env")


class Settings:
    host = os.getenv("AGENTIC_HOST", "127.0.0.1")
    port = int(os.getenv("AGENTIC_PORT", "5055"))
    log_level = os.getenv("AGENTIC_LOG_LEVEL", "info")

    primary_provider = os.getenv("LLM_PRIMARY_PROVIDER", "longcat").lower()
    fallback_providers = [
        provider.strip().lower()
        for provider in os.getenv("LLM_FALLBACK_PROVIDERS", "gemini,groq").split(",")
        if provider.strip()
    ]
    default_mode = os.getenv("LLM_DEFAULT_MODE", "fast").lower()

    longcat_api_key = os.getenv("LONGCAT_API_KEY", "")
    longcat_base_url = os.getenv("LONGCAT_BASE_URL", "https://api.longcat.chat/openai/v1").rstrip("/")
    longcat_fast_model = os.getenv("LONGCAT_FAST_MODEL", "LongCat-Flash-Chat")
    longcat_balanced_model = os.getenv("LONGCAT_BALANCED_MODEL", "LongCat-Flash-Chat")
    longcat_deep_model = os.getenv("LONGCAT_DEEP_MODEL", "LongCat-Flash-Thinking")

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    gemini_fast_model = os.getenv("GEMINI_FAST_MODEL", "gemini-2.5-flash-lite")
    gemini_balanced_model = os.getenv("GEMINI_BALANCED_MODEL", "gemini-2.5-flash")
    gemini_deep_model = os.getenv("GEMINI_DEEP_MODEL", "gemini-2.5-pro")

    groq_api_key = os.getenv("GROQ_API_KEY", "")
    groq_fallback_model = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.3-70b-versatile")
    groq_fast_model = os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_output_tokens = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "4096"))
    timeout_seconds = int(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60"))

    @classmethod
    def provider_configured(cls, provider: str) -> bool:
        if provider == "longcat":
            return bool(cls.longcat_api_key and not cls.longcat_api_key.startswith("your_"))
        if provider == "gemini":
            return bool(cls.gemini_api_key and not cls.gemini_api_key.startswith("your_"))
        if provider == "groq":
            return bool(cls.groq_api_key and not cls.groq_api_key.startswith("your_"))
        return False

    @classmethod
    def longcat_model_for_mode(cls, mode: str) -> str:
        if mode == "deep":
            return cls.longcat_deep_model
        if mode == "balanced":
            return cls.longcat_balanced_model
        return cls.longcat_fast_model

    @classmethod
    def gemini_model_for_mode(cls, mode: str) -> str:
        if mode == "deep":
            return cls.gemini_deep_model
        if mode == "balanced":
            return cls.gemini_balanced_model
        return cls.gemini_fast_model

    @classmethod
    def groq_model_for_mode(cls, mode: str) -> str:
        if mode == "fast":
            return cls.groq_fast_model
        return cls.groq_fallback_model
