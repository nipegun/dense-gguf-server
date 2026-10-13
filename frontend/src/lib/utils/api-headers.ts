import { fRedactValue } from './redact';
import { cCorsProxy, HEADERS } from '$lib/constants';
import { MimeTypeApplication } from '$lib/enums';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';

/**
 * Get authorization headers for API requests
 * Includes Bearer token if API key is configured
 */
export function fGetAuthHeaders(): Record<string, string> {
  const cCurrentConfig = cSettingsStore.vConfig;
  const cApiKey = cCurrentConfig.apiKey?.toString().trim();

  return cApiKey ? { [HEADERS.AUTHORIZATION]: `${HEADERS.BEARER}${cApiKey}` } : {};
}

/**
 * Get standard JSON headers with optional authorization
 */
export function fGetJsonHeaders(): Record<string, string> {
  return {
    [HEADERS.CONTENT_TYPE]: MimeTypeApplication.cJson,
    ...fGetAuthHeaders()
  };
}

/**
 * Sanitize HTTP headers by redacting sensitive values.
 * Known sensitive headers (from HEADERS.REDACTED) and any extra headers
 * specified by the caller are fully redacted. Headers listed in
 * `partialRedactHeaders` are partially redacted, showing only the
 * specified number of trailing characters.
 *
 * @param headers - Headers to sanitize
 * @param extraRedactedHeaders - Additional header names to fully redact
 * @param partialRedactHeaders - Map of header name -> number of trailing chars to keep visible
 * @returns Object with header names as keys and (possibly redacted) values
 */
export function fSanitizeHeaders(
  pHeaders?: HeadersInit,
  pExtraRedactedHeaders?: Iterable<string>,
  pPartialRedactHeaders?: Map<string, number>
): Record<string, string> {
  if (!pHeaders) {
    return {};
  }

  const cNormalized = new Headers(pHeaders);
  const dSanitized: Record<string, string> = {};
  const sRedactedHeaders = new Set(
    Array.from(pExtraRedactedHeaders ?? [], (pHeader) => pHeader.toLowerCase())
  );

  for (const [vKey, vValue] of cNormalized.entries()) {
    const cNormalizedKey = vKey.toLowerCase();
    const cUnproxiedKey = cNormalizedKey.startsWith(cCorsProxy.HEADER_PREFIX)
      ? cNormalizedKey.slice(cCorsProxy.HEADER_PREFIX.length)
      : cNormalizedKey;
    const cPartialChars =
      pPartialRedactHeaders?.get(cNormalizedKey) ?? pPartialRedactHeaders?.get(cUnproxiedKey);

    if (cPartialChars !== undefined) {
      dSanitized[vKey] = fRedactValue(vValue, cPartialChars);
    } else if (
      HEADERS.REDACTED.has(cNormalizedKey) ||
      HEADERS.REDACTED.has(cUnproxiedKey) ||
      sRedactedHeaders.has(cNormalizedKey) ||
      sRedactedHeaders.has(cUnproxiedKey)
    ) {
      dSanitized[vKey] = fRedactValue(vValue);
    } else {
      dSanitized[vKey] = vValue;
    }
  }

  return dSanitized;
}
