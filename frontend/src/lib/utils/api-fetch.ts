import { fGetAuthHeaders, fGetJsonHeaders } from './api-headers';
import { fApiUrl } from '$lib/api-url';
import { aApiAbsoluteUrlProtocols, cErrorMessages, dHttpCodeToString } from '$lib/constants';

/**
 * API Fetch Utilities
 *
 * Provides common fetch patterns used across services:
 * - Automatic JSON headers
 * - Error handling with proper error messages
 * - Base path resolution
 */

/**
 * Error thrown when an API request fails, carrying the HTTP status code
 * so callers can distinguish e.g. a 503 "still loading" response from a
 * genuine failure.
 */
export class ApiError extends Error {
  status: number;

  constructor(pMessage: string, pStatus: number) {
    super(pMessage);
    this.name = 'ApiError';
    this.status = pStatus;
  }
}

export interface ApiFetchOptions extends Omit<RequestInit, 'headers'> {
  /**
   * Use auth-only headers (no Content-Type).
   * Default: false (uses JSON headers with Content-Type: application/json)
   */
  authOnly?: boolean;
  /**
   * Additional headers to merge with default headers.
   */
  headers?: Record<string, string>;
}

/**
 * Fetch JSON data from an API endpoint with standard headers and error handling.
 *
 * @param path - API path (will be prefixed with base path)
 * @param options - Fetch options with additional authOnly flag
 * @returns Parsed JSON response
 * @throws Error with formatted message on failure
 *
 * @example
 * ```typescript
 * // GET request
 * const models = await apiFetch<ApiModelsListResponse>('/v1/models');
 *
 * // POST request
 * const result = await apiFetch<ApiResponse>('/models/load', {
 *   method: 'POST',
 *   body: JSON.stringify({ model: 'gpt-4' })
 * });
 * ```
 */
export async function fApiFetch<T>(pPath: string, pOptions: ApiFetchOptions = {}): Promise<T> {
  const { authOnly: vAuthOnly = false, headers: dCustomHeaders, ...vFetchOptions } = pOptions;
  const dBaseHeaders = vAuthOnly ? fGetAuthHeaders() : fGetJsonHeaders();
  const headers = { ...dBaseHeaders, ...dCustomHeaders };
  // absolute URLs with an allowed protocol pass through untouched; relative paths get the base prefix
  const url = aApiAbsoluteUrlProtocols.some((pP) => pPath.startsWith(pP)) ? pPath : fApiUrl(pPath);

  let vResponse;

  try {
    vResponse = await fetch(url, {
      ...vFetchOptions,
      headers
    });
  } catch (vE) {
    throw new Error(fBeautifyNetworkError(vE));
  }

  if (!vResponse.ok) {
    const cErrorMessage = await fParseErrorMessage(vResponse);

    throw new ApiError(cErrorMessage, vResponse.status);
  }

  return vResponse.json() as Promise<T>;
}

/**
 * Fetch with URL constructed from base URL and query parameters.
 *
 * @param basePath - Base API path
 * @param params - Query parameters to append
 * @param options - Fetch options
 * @returns Parsed JSON response
 *
 * @example
 * ```typescript
 * const props = await apiFetchWithParams<ApiProps>('./props', {
 *   model: 'gpt-4',
 *   autoload: 'false'
 * });
 * ```
 */
export async function fApiFetchWithParams<T>(
  pBasePath: string,
  pParams: Record<string, string>,
  pOptions: ApiFetchOptions = {}
): Promise<T> {
  const url = new URL(fApiUrl(pBasePath), window.location.href);

  for (const [vKey, vValue] of Object.entries(pParams)) {
    if (vValue !== undefined && vValue !== null) {
      url.searchParams.set(vKey, vValue);
    }
  }

  return fApiFetch<T>(url.toString(), pOptions);
}

/**
 * POST JSON data to an API endpoint.
 *
 * @param path - API path
 * @param body - Request body (will be JSON stringified)
 * @param options - Additional fetch options
 * @returns Parsed JSON response
 */
export async function fApiPost<T, B = unknown>(
  pPath: string,
  pBody: B,
  pOptions: ApiFetchOptions = {}
): Promise<T> {
  return fApiFetch<T>(pPath, {
    body: JSON.stringify(pBody),
    method: 'POST',
    ...pOptions
  });
}

/**
 * Send a DELETE request to an API endpoint, optionally with query parameters.
 *
 * @param path - API path (query string is appended if `params` is provided)
 * @param params - Optional record of query parameters
 * @param options - Additional fetch options
 * @returns Parsed JSON response
 */
export async function fApiDelete<T>(
  pPath: string,
  pParams?: Record<string, string>,
  pOptions: ApiFetchOptions = {}
): Promise<T> {
  // the query is appended to the path so `apiFetch` applies its base-path prefix;
  // `apiFetchWithParams` resolves an absolute URL and would bypass it
  let vQuery = '';

  if (pParams) {
    const search = new URLSearchParams();

    for (const [vKey, vValue] of Object.entries(pParams)) {
      if (vValue !== undefined && vValue !== null) {
        search.set(vKey, vValue);
      }
    }

    const cQs = search.toString();

    if (cQs) vQuery = `?${cQs}`;
  }

  return fApiFetch<T>(`${pPath}${vQuery}`, { ...pOptions, method: 'DELETE' });
}

/**
 * Parse error message from a failed response.
 * Tries to extract error message from JSON body, falls back to status text.
 */
async function fParseErrorMessage(pResponse: Response): Promise<string> {
  try {
    const cErrorData = await pResponse.json();

    if (cErrorData?.error?.message) {
      return cErrorData.error.message;
    }

    if (cErrorData?.error && typeof cErrorData.error === 'string') {
      return cErrorData.error;
    }

    if (cErrorData?.message) {
      return cErrorData.message;
    }
  } catch {
    // JSON parsing failed, use status text
  }

  const cHttpErrorStr = dHttpCodeToString[pResponse.status];

  if (cHttpErrorStr) {
    return cHttpErrorStr;
  }

  return `${cErrorMessages.HTTP.GENERIC}: ${pResponse.status} ${pResponse.statusText}`;
}

/**
 * Converts a network issue into a human-readable message.
 * @param throwable - The throwable raised during fetch operation
 * @returns Error in an human-readable format
 */
function fBeautifyNetworkError(pThrowable: unknown): string {
  let vMessage;

  if (pThrowable instanceof Error) {
    vMessage = pThrowable.message;

    if (pThrowable.name === 'TypeError' && vMessage.includes('fetch')) {
      return cErrorMessages.NETWORK.UNREACHABLE;
    }
  } else {
    vMessage = String(pThrowable);
  }

  if (vMessage.includes('ECONNREFUSED')) {
    return cErrorMessages.NETWORK.REFUSED;
  } else if (vMessage.includes('ENOTFOUND')) {
    return cErrorMessages.NETWORK.NXDOMAIN;
  } else if (vMessage.includes('ETIMEDOUT')) {
    return cErrorMessages.NETWORK.TIMEOUT;
  }

  return `${cErrorMessages.NETWORK.GENERIC} (${vMessage})`;
}
