import { fExtractRootDomain } from './url';
import {
  AlertTriangle,
  Code,
  Database,
  File,
  FileText,
  Image,
  Info,
  XCircle
} from '@lucide/svelte';
import {
  cCodeFileExtensionRegex,
  cDefaultResourceFilename,
  cDisplayNameSeparatorRegex,
  cExpectedThemedIconPairCount,
  cFileExtensionRegex,
  cImageFileExtensionRegex,
  sMcpAllowedIconMimeTypes,
  cMcpServerIdPrefix,
  cMimeTypePrefixes,
  cMimeTypeSubstrings,
  cPathSeparator,
  cProtocolPrefixRegex,
  cResourceTextContentSeparator,
  cTextFileExtensionRegex,
  cUriPatterns
} from '$lib/constants';
import {
  ColorMode,
  HealthCheckStatus,
  MCPLogLevel,
  MCPTransportType,
  MimeTypeText,
  UrlProtocol
} from '$lib/enums';
import type {
  HealthCheckState,
  MCPResourceContent,
  MCPResourceIcon,
  MCPResourceInfo,
  MCPServerDisplayInfo,
  MCPServerSettingsEntry
} from '$lib/types';
import type { MimeTypeUnion } from '$lib/types/common';
import type { Component } from 'svelte';

/**
 * Detects the MCP transport type from a URL.
 * WebSocket URLs (ws:// or wss://) use 'websocket', others use 'streamable_http'.
 */
export function fDetectMcpTransportFromUrl(pUrl: string): MCPTransportType {
  const cNormalized = pUrl.trim().toLowerCase();

  if (
    cNormalized.startsWith(UrlProtocol.cWebsocket) ||
    cNormalized.startsWith(UrlProtocol.cWebsocketSecure)
  ) {
    return MCPTransportType.cWebsocket;
  }

  return MCPTransportType.cStreamableHttp;
}

/**
 * Parses MCP server settings from a JSON string or array.
 * @param rawServers - The raw servers to parse
 * @returns An empty array if the input is invalid.
 */
export function fParseMcpServerSettings(pRawServers: unknown): MCPServerSettingsEntry[] {
  if (!pRawServers) return [];

  let vParsed: unknown;

  if (typeof pRawServers === 'string') {
    const cTrimmed = pRawServers.trim();

    if (!cTrimmed) return [];

    try {
      vParsed = JSON.parse(cTrimmed);
    } catch (error) {
      console.warn('[MCP] Failed to parse mcpServers JSON, ignoring value:', error);

      return [];
    }
  } else {
    vParsed = pRawServers;
  }

  if (!Array.isArray(vParsed)) return [];

  return vParsed.map((pEntry, pIndex) => {
    const url = typeof pEntry?.url === 'string' ? pEntry.url.trim() : '';
    const headers = typeof pEntry?.headers === 'string' ? pEntry.headers.trim() : undefined;
    const cId =
      typeof (pEntry as { id?: unknown })?.id === 'string' && (pEntry as { id?: string }).id?.trim()
        ? (pEntry as { id: string }).id.trim()
        : `${cMcpServerIdPrefix}-${pIndex + 1}`;

    return {
      displayName: (pEntry as { displayName?: string })?.displayName,
      enabled: Boolean((pEntry as { enabled?: unknown })?.enabled),
      headers: headers || undefined,
      id: cId,
      name: (pEntry as { name?: string })?.name,
      url,
      useProxy: Boolean((pEntry as { useProxy?: unknown })?.useProxy)
    } satisfies MCPServerSettingsEntry;
  });
}

/**
 * Get the appropriate icon component for a log level
 *
 * @param level - MCP log level
 * @returns Lucide icon component
 */
export function fGetMcpLogLevelIcon(pLevel: MCPLogLevel): Component {
  switch (pLevel) {
    case MCPLogLevel.ERROR:
      return XCircle;
    case MCPLogLevel.cWarn:
      return AlertTriangle;
    default:
      return Info;
  }
}

/**
 * Get the appropriate CSS class for a log level
 *
 * @param level - MCP log level
 * @returns Tailwind CSS class string
 */
export function fGetMcpLogLevelClass(pLevel: MCPLogLevel): string {
  switch (pLevel) {
    case MCPLogLevel.ERROR:
      return 'text-destructive';
    case MCPLogLevel.cWarn:
      return 'text-yellow-600 dark:text-yellow-500';
    default:
      return 'text-muted-foreground';
  }
}

/**
 * Check if a MIME type represents an image.
 *
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type starts with 'image/'
 */
export function fIsImageMimeType(pMimeType?: MimeTypeUnion): boolean {
  return pMimeType?.startsWith(cMimeTypePrefixes.IMAGE) ?? false;
}

/**
 * Parse a resource URI into path segments, stripping the protocol prefix.
 *
 * @param uri - The resource URI to parse
 * @returns Array of non-empty path segments
 */
export function fParseResourcePath(pUri: string): string[] {
  try {
    const cWithoutProtocol = pUri.replace(cProtocolPrefixRegex, '');

    return cWithoutProtocol.split(cPathSeparator).filter((pP) => pP.length > 0);
  } catch {
    return [pUri];
  }
}

/**
 * Convert a path part into a human-readable display name.
 * Strips file extensions and converts kebab-case/snake_case to Title Case.
 *
 * @param pathPart - The path segment to convert
 * @returns Human-readable display name
 */
export function fGetDisplayName(pPathPart: string): string {
  const cWithoutExt = pPathPart.replace(cFileExtensionRegex, '');

  return cWithoutExt
    .split(cDisplayNameSeparatorRegex)
    .map((pWord) => pWord.charAt(0).toUpperCase() + pWord.slice(1))
    .join(' ');
}

/**
 * Get the display name from a resource, extracting the last path segment from the URI.
 *
 * @param resource - The MCP resource info
 * @returns Display name string
 */
export function fGetResourceDisplayName(pResource: MCPResourceInfo): string {
  try {
    const aParts = fParseResourcePath(pResource.uri);

    return aParts[aParts.length - 1] || pResource.name || pResource.uri;
  } catch {
    return pResource.name || pResource.uri;
  }
}

/**
 * Determine if a MIME type and/or URI represents code content.
 *
 * @param mimeType - Optional MIME type string
 * @param uri - Optional URI string
 * @returns True if the content is code
 */
export function fIsCodeResource(pMimeType?: MimeTypeUnion, pUri?: string): boolean {
  const cMime = pMimeType?.toLowerCase() || '';
  const cU = pUri?.toLowerCase() || '';

  return (
    cMime.includes(cMimeTypeSubstrings.JSON) ||
    cMime.includes(cMimeTypeSubstrings.JAVASCRIPT) ||
    cMime.includes(cMimeTypeSubstrings.TYPESCRIPT) ||
    cCodeFileExtensionRegex.test(cU)
  );
}

/**
 * Determine if a MIME type and/or URI represents image content.
 *
 * @param mimeType - Optional MIME type string
 * @param uri - Optional URI string
 * @returns True if the content is an image
 */
export function fIsImageResource(pMimeType?: MimeTypeUnion, pUri?: string): boolean {
  const cMime = pMimeType?.toLowerCase() || '';
  const cU = pUri?.toLowerCase() || '';

  return cMime.startsWith(cMimeTypePrefixes.IMAGE) || cImageFileExtensionRegex.test(cU);
}

/**
 * Get the appropriate Lucide icon component for an MCP resource based on its MIME type and URI.
 *
 * @param mimeType - Optional MIME type of the resource
 * @param uri - Optional URI of the resource
 * @returns Lucide icon component
 */
export function fGetResourceIcon(pMimeType?: MimeTypeUnion, pUri?: string): Component {
  const cMime = pMimeType?.toLowerCase() || '';
  const cU = pUri?.toLowerCase() || '';

  if (cMime.startsWith(cMimeTypePrefixes.IMAGE) || cImageFileExtensionRegex.test(cU)) {
    return Image;
  }

  if (
    cMime.includes(cMimeTypeSubstrings.JSON) ||
    cMime.includes(cMimeTypeSubstrings.JAVASCRIPT) ||
    cMime.includes(cMimeTypeSubstrings.TYPESCRIPT) ||
    cCodeFileExtensionRegex.test(cU)
  ) {
    return Code;
  }

  if (cMime.includes(cMimeTypePrefixes.TEXT) || cTextFileExtensionRegex.test(cU)) {
    return FileText;
  }

  if (cU.includes(cUriPatterns.DATABASE_KEYWORD) || cU.includes(cUriPatterns.DATABASE_SCHEME)) {
    return Database;
  }

  return File;
}

/**
 * Extract text content from MCP resource content array.
 *
 * @param content - Array of MCP resource content items
 * @returns Joined text content string
 */
export function fGetResourceTextContent(pContent: MCPResourceContent[] | null | undefined): string {
  if (!pContent) return '';

  return pContent
    .filter((pC): pC is { uri: string; mimeType?: MimeTypeUnion; text: string } => 'text' in pC)
    .map((pC) => pC.text)
    .join(cResourceTextContentSeparator);
}

/**
 * Extract blob content from MCP resource content array.
 *
 * @param content - Array of MCP resource content items
 * @returns Array of blob content items
 */
export function fGetResourceBlobContent(
  pContent: MCPResourceContent[] | null | undefined
): Array<{ uri: string; mimeType?: MimeTypeUnion; blob: string }> {
  if (!pContent) return [];

  return pContent.filter(
    (pC): pC is { uri: string; mimeType?: MimeTypeUnion; blob: string } => 'blob' in pC
  );
}

/**
 * Trigger a file download from text content.
 *
 * @param text - The text content to download
 * @param mimeType - MIME type for the blob
 * @param filename - Suggested filename
 */
export function fDownloadResourceContent(
  pText: string,
  pMimeType: MimeTypeUnion = MimeTypeText.cPlain,
  pFilename: string = cDefaultResourceFilename
): void {
  const cBlob = new Blob([pText], { type: pMimeType });
  const url = URL.createObjectURL(cBlob);
  const cA = document.createElement('a');

  cA.href = url;
  cA.download = pFilename;
  document.body.appendChild(cA);
  cA.click();
  document.body.removeChild(cA);
  URL.revokeObjectURL(url);
}

/**
 * Validates that an icon URI uses a safe scheme (https: or data:).
 */
function fIsValidMcpIconUri(pSrc: string): boolean {
  try {
    if (pSrc.startsWith(UrlProtocol.DATA)) return true;

    const url = new URL(pSrc);

    return url.protocol === UrlProtocol.cHttps;
  } catch {
    return false;
  }
}

/**
 * Selects the best icon URL from an MCP icons array.
 * Follows security guidelines from the MCP specification:
 * - Only allows https: and data: URIs
 * - Filters to supported MIME types
 *
 * Selection priority:
 * 1. Icon matching the current color scheme (dark/light)
 * 2. Universal icon (no theme specified); if exactly 2, assumes [0]=light, [1]=dark
 * 3. First valid icon as last resort
 */
export function fGetMcpIconUrl(
  pIcons: MCPResourceIcon[] | undefined,
  pIsDark = false
): string | null {
  if (!pIcons?.length) return null;

  const aValidIcons = pIcons.filter((pIcon) => {
    if (!pIcon.src || !fIsValidMcpIconUri(pIcon.src)) return false;

    if (pIcon.mimeType && !sMcpAllowedIconMimeTypes.has(pIcon.mimeType)) return false;

    return true;
  });

  if (aValidIcons.length === 0) return null;

  const cPreferredTheme = pIsDark ? ColorMode.DARK : ColorMode.cLight;
  // 1. Prefer icon explicitly matching the current color scheme
  const cThemedIcon = aValidIcons.find((pIcon) => pIcon.theme === cPreferredTheme);

  if (cThemedIcon) return cThemedIcon.src;

  // 2. Handle universal icons (no theme specified)
  const aUniversalIcons = aValidIcons.filter((pIcon) => !pIcon.theme);

  if (aUniversalIcons.length === cExpectedThemedIconPairCount) {
    // Heuristic: two theme-less icons → assume [0] = light, [1] = dark
    return aUniversalIcons[pIsDark ? 1 : 0].src;
  }

  if (aUniversalIcons.length > 0) {
    return aUniversalIcons[0].src;
  }

  // 3. Last resort: use opposite-theme icon
  return aValidIcons[0].src;
}

/**
 * Construct a fallback favicon URL from the MCP server URL.
 * e.g. https://mcp.example.com/sse -> https://example.com/favicon.ico
 */
export function fGetMcpServerFaviconFallback(pServerUrl: string): string | null {
  try {
    const url = new URL(pServerUrl);
    const cRootDomain = fExtractRootDomain(url);

    if (!cRootDomain) return null;

    const cOrigin = `${url.protocol}//${cRootDomain}`;
    const aCandidates = ['favicon.ico', 'favicon.png'];

    for (const cPath of aCandidates) {
      const cFaviconUrl = `${cOrigin}/${cPath}`;

      if (fIsValidMcpIconUri(cFaviconUrl)) {
        return cFaviconUrl;
      }
    }
  } catch {
    // Invalid URL, return null
  }

  return null;
}

/**
 * Resolves the raw label for a server: user-defined display name first,
 * then server-reported title or name when the health check succeeded,
 * then the configured name (configured server settings), then URL.
 */
function fGetMcpServerBaseLabel(
  pServer: MCPServerDisplayInfo,
  pHealthState?: HealthCheckState
): string {
  if (pServer.displayName) return pServer.displayName;

  if (pHealthState?.status === HealthCheckStatus.SUCCESS)
    return (
      pHealthState.serverInfo?.title || pHealthState.serverInfo?.name || pServer.name || pServer.url
    );

  return pServer.name || pServer.url;
}

/**
 * Returns the display label for a server, suffixed with a positional
 * counter when several configured servers resolve to the same base label
 * (e.g. two endpoints of the same host reporting an identical name).
 * Numbering follows config order, so it is stable across renders.
 */
export function fGetMcpServerLabel(
  pServer: MCPServerDisplayInfo,
  pServers: MCPServerDisplayInfo[],
  pHealthChecks: Record<string, HealthCheckState>
): string {
  const cLabel = fGetMcpServerBaseLabel(pServer, pHealthChecks[pServer.id]);
  const aTwins = pServers.filter(
    (pS) => fGetMcpServerBaseLabel(pS, pHealthChecks[pS.id]) === cLabel
  );

  if (aTwins.length < 2) return cLabel;

  const cPosition = aTwins.findIndex((pS) => pS.id === pServer.id);

  return cPosition < 0 ? cLabel : `${cLabel} (${cPosition + 1})`;
}
