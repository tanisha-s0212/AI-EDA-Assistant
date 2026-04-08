import axios from 'axios';

const normalizedBaseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL || '/api').replace(/\/$/, '');

export const apiClient = axios.create({
  baseURL: normalizedBaseUrl,
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

