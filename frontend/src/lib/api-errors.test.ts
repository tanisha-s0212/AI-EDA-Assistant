import axios from 'axios';
import { describe, expect, it } from 'vitest';
import { getApiErrorMessage } from './api';

describe('API error messages', () => {
  it('shows a structured readiness explanation', () => {
    const error = new axios.AxiosError('Request failed', 'ERR_BAD_REQUEST', undefined, undefined, {
      data: { detail: { message: 'Need at least 24 observed months.' } }, status: 422,
      statusText: 'Unprocessable', headers: {}, config: { headers: new axios.AxiosHeaders() },
    });
    expect(getApiErrorMessage(error, 'fallback')).toBe('Need at least 24 observed months.');
  });
  it('shows the backend structured error message', () => {
    const error = new axios.AxiosError('Request failed', 'ERR_BAD_REQUEST', undefined, undefined, {
      data: { error: { message: 'Need at least 24 observed months.' } }, status: 422,
      statusText: 'Unprocessable', headers: {}, config: { headers: new axios.AxiosHeaders() },
    });
    expect(getApiErrorMessage(error, 'fallback')).toBe('Need at least 24 observed months.');
  });
});
