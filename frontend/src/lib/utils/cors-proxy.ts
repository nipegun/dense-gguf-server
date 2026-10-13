/**
 * CORS Proxy utility for routing requests through dgs-server's CORS proxy.
 */

import { base } from '$app/paths';
import { cCorsProxy, cCorsProxyEndpoint } from '$lib/constants';
import { UrlProtocol } from '$lib/enums';

/**
 * Build a proxied URL that routes through dgs-server's CORS proxy.
 * @param targetUrl - The original URL to proxy
 * @returns URL pointing to the CORS proxy with target encoded
 */
export function fBuildProxiedUrl(pTargetUrl: string): URL {
  const cProxyPath = `${base}${cCorsProxyEndpoint}`;
  const cProxyUrl = new URL(cProxyPath, window.location.origin);

  cProxyUrl.searchParams.set(cCorsProxy.URL_PARAM, pTargetUrl);

  return cProxyUrl;
}

/**
 * Route a remote icon through the CORS proxy so it renders inside the UI's
 * cross-origin isolation. Inline data URLs are left alone, and with the proxy
 * off the origin URL stands in, so the caller keeps its own fallback.
 */
export function fBuildProxiedIconUrl(
  pIconUrl: string | null,
  pProxyAvailable: boolean
): string | null {
  if (!pIconUrl || !pProxyAvailable || pIconUrl.startsWith(UrlProtocol.DATA)) return pIconUrl;

  return fBuildProxiedUrl(pIconUrl).toString();
}

/**
 * Wrap original headers for proxying through the CORS proxy. This avoids issues with duplicated dgs.cpp-specific and target headers when using the CORS proxy.
 * @param headers - The original headers to be proxied to target
 * @returns List of "wrapped" headers to be sent to the CORS proxy
 */
export function fBuildProxiedHeaders(pHeaders: Record<string, string>): Record<string, string> {
  const dProxiedHeaders: Record<string, string> = {};

  for (const [vKey, vValue] of Object.entries(pHeaders)) {
    dProxiedHeaders[`${cCorsProxy.HEADER_PREFIX}${vKey}`] = vValue;
  }

  return dProxiedHeaders;
}
