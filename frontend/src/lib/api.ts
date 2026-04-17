import axios from 'axios';

const normalizedBaseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL || '/api').replace(/\/$/, '');
const CLIENT_SESSION_STORAGE_KEY = 'ai-eda-client-session-id';

function createClientSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }

  return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getClientSessionId(): string | null {
  if (typeof window === 'undefined') {
    return null;
  }

  const existing = window.localStorage.getItem(CLIENT_SESSION_STORAGE_KEY);
  if (existing) {
    return existing;
  }

  const nextValue = createClientSessionId();
  window.localStorage.setItem(CLIENT_SESSION_STORAGE_KEY, nextValue);
  return nextValue;
}

export const apiClient = axios.create({
  baseURL: normalizedBaseUrl,
});

apiClient.interceptors.request.use((config) => {
  const clientSessionId = getClientSessionId();
  if (clientSessionId) {
    config.headers = config.headers ?? {};
    config.headers['X-Client-Session-Id'] = clientSessionId;
  }
  return config;
});

export function getApiErrorMessage(error: unknown, fallback: string): string {
  if (axios.isAxiosError(error)) {
    const responseData = error.response?.data;
    if (typeof responseData === 'object' && responseData !== null) {
      if ('detail' in responseData && typeof responseData.detail === 'string') {
        return responseData.detail;
      }

      if ('error' in responseData && typeof responseData.error === 'string') {
        return responseData.error;
      }
    }

    return error.message || fallback;
  }

  if (error instanceof Error) {
    return error.message;
  }

  return fallback;
}

