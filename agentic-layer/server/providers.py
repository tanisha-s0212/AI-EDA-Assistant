from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import Settings


class ProviderError(RuntimeError):
    pass


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _gemini_model_candidates(mode: str) -> list[str]:
    if mode == "deep":
        return _unique([Settings.gemini_deep_model, Settings.gemini_balanced_model, Settings.gemini_fast_model])
    if mode == "balanced":
        return _unique([Settings.gemini_balanced_model, Settings.gemini_fast_model, Settings.gemini_deep_model])
    return _unique([Settings.gemini_fast_model, Settings.gemini_balanced_model])


def _groq_model_candidates(mode: str) -> list[str]:
    if mode == "fast":
        return _unique([Settings.groq_fast_model, Settings.groq_fallback_model])
    return _unique([Settings.groq_fallback_model, Settings.groq_fast_model])


def _post_json(url: str, payload: dict, headers: dict[str, str], timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise ProviderError(f"Provider HTTP {exc.code}: {details[:500]}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Provider request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderError("Provider request timed out.") from exc


def _call_gemini_model(messages: list[dict[str, str]], model: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={Settings.gemini_api_key}"

    system_parts = [message["content"] for message in messages if message["role"] == "system"]
    user_parts = [message["content"] for message in messages if message["role"] != "system"]
    prompt = "\n\n".join([*system_parts, *user_parts])

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": Settings.temperature,
            "maxOutputTokens": Settings.max_output_tokens,
        },
    }

    data = _post_json(url, payload, {"Content-Type": "application/json"}, Settings.timeout_seconds)
    candidates = data.get("candidates") or []
    if not candidates:
        raise ProviderError("Gemini returned no candidates.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts).strip()
    if not text:
        raise ProviderError("Gemini returned an empty response.")
    return text


def call_gemini(messages: list[dict[str, str]], mode: str) -> str:
    errors: list[str] = []
    for model in _gemini_model_candidates(mode):
        try:
            return _call_gemini_model(messages, model)
        except ProviderError as exc:
            errors.append(f"{model}: {exc}")
    raise ProviderError("Gemini model attempts failed. " + " | ".join(errors))


def _call_groq_model(messages: list[dict[str, str]], model: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": Settings.temperature,
        "max_completion_tokens": Settings.max_output_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {Settings.groq_api_key}",
    }

    data = _post_json(url, payload, headers, Settings.timeout_seconds)
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError("Groq returned no choices.")

    text = choices[0].get("message", {}).get("content", "").strip()
    if not text:
        raise ProviderError("Groq returned an empty response.")
    return text


def call_groq(messages: list[dict[str, str]], mode: str) -> str:
    errors: list[str] = []
    for model in _groq_model_candidates(mode):
        try:
            return _call_groq_model(messages, model)
        except ProviderError as exc:
            errors.append(f"{model}: {exc}")
    raise ProviderError("Groq model attempts failed. " + " | ".join(errors))


def call_provider(provider: str, messages: list[dict[str, str]], mode: str) -> str:
    if provider == "gemini":
        if not Settings.provider_configured("gemini"):
            raise ProviderError("Gemini API key is not configured.")
        return call_gemini(messages, mode)

    if provider == "groq":
        if not Settings.provider_configured("groq"):
            raise ProviderError("Groq API key is not configured.")
        return call_groq(messages, mode)

    raise ProviderError(f"Unsupported provider: {provider}")
