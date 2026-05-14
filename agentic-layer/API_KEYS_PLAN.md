# API Key Plan

The agentic layer uses LongCat as the primary provider, Gemini as the second priority, and Groq as the final hosted fallback provider.

## Provider Priority

```env
LLM_PRIMARY_PROVIDER=longcat
LLM_FALLBACK_PROVIDERS=gemini,groq
```

Auto mode uses:

```text
LongCat -> Gemini -> Groq -> local workspace context
```

## LongCat Models

```env
LONGCAT_BASE_URL=https://api.longcat.chat/openai/v1
LONGCAT_FAST_MODEL=LongCat-Flash-Chat
LONGCAT_BALANCED_MODEL=LongCat-Flash-Chat
LONGCAT_DEEP_MODEL=LongCat-Flash-Thinking
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
