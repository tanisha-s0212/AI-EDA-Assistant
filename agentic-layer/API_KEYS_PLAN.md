# API Key Plan

The agentic layer uses Gemini as the primary provider and Groq as a fallback provider.

## Provider Priority

```env
LLM_PRIMARY_PROVIDER=gemini
LLM_FALLBACK_PROVIDER=groq
```

## Gemini Models

```env
GEMINI_FAST_MODEL=gemini-2.5-flash-lite
GEMINI_BALANCED_MODEL=gemini-2.5-flash
GEMINI_DEEP_MODEL=gemini-2.5-pro
```

- Primary fast/free-friendly model: `gemini-2.5-flash-lite`
- Better reasoning model: `gemini-2.5-flash`
- Deep reasoning model: `gemini-2.5-pro`

## Groq Models

```env
GROQ_FALLBACK_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant
```

- Groq fallback model: `llama-3.3-70b-versatile`
- Groq fast model: `llama-3.1-8b-instant`

## Security

Store real API keys only in:

```text
agentic-layer/.env
```

Do not commit real API keys.
