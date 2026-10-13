import { Globe, Zap } from '@lucide/svelte';
import { MCPTransportType } from '$lib/enums';
import { MimeTypeImage } from '$lib/enums/files.enums';
import type { ClientCapabilities, Implementation } from '$lib/types';
import type { Component } from 'svelte';

export const cDefaultClientVersion = '1.0.0';
export const cMcpClientName = 'dgs-ui-mcp';
export const cDefaultImageMimeType = MimeTypeImage.cPng;

/** MIME types considered safe for rendering MCP server icons */
export const sMcpAllowedIconMimeTypes = new Set([
  MimeTypeImage.cPng,
  MimeTypeImage.cJpeg,
  MimeTypeImage.cJpg,
  MimeTypeImage.SVG,
  MimeTypeImage.cWebp,
  MimeTypeImage.cIco,
  MimeTypeImage.cIcoMicrosoft
]);

/**
 * MCP specification version this client targets.
 * Update when the upstream MCP spec introduces a new stable version:
 * https://spec.modelcontextprotocol.io/
 */
export const cMcpProtocolVersion = '2025-06-18';

export const cDefaultMcpConfig = {
  capabilities: { tools: { listChanged: true } } as ClientCapabilities,
  clientInfo: { name: cMcpClientName, version: cDefaultClientVersion } as Implementation,
  connectionTimeoutMs: 10_000, // 10 seconds for connection establishment
  protocolVersion: cMcpProtocolVersion,
  requestTimeoutSeconds: 300 // 5 minutes for long-running tools
} as const;

export const cMcpServerIdPrefix = 'DgsUI-MCP-Server';

/** Backoff policy for reconnecting to a dropped MCP server. */
export const cMcpReconnect = {
  /** Per-attempt timeout for a single reconnection attempt before giving up and backing off. */
  ATTEMPT_TIMEOUT_MS: 15_000,
  BACKOFF_MULTIPLIER: 2,
  INITIAL_DELAY: 1000,
  MAX_DELAY: 30000
};

/** Maximum number of MCP server avatars to display in the chat form */
export const cMaxDisplayedMcpAvatars = 4;

/** Expected count when two theme-less icons represent a light/dark pair */
export const cExpectedThemedIconPairCount = 2;

/** CORS proxy connection settings */
export const cCorsProxy = {
  /** Header prefix for headers that should be forwarded by the CORS proxy */
  HEADER_PREFIX: 'x-dgs-server-proxy-header-',
  /** CORS proxy URL query parameter name */
  URL_PARAM: 'url'
} as const;

/** Human-readable labels for MCP transport types */
export const dMcpTransportLabels: Record<MCPTransportType, string> = {
  [MCPTransportType.cStreamableHttp]: 'HTTP',
  [MCPTransportType.cWebsocket]: 'WebSocket'
};

/** Icon components for MCP transport types */
export const dMcpTransportIcons: Record<MCPTransportType, Component> = {
  [MCPTransportType.cStreamableHttp]: Globe,
  [MCPTransportType.cWebsocket]: Zap
};

/** Settings entry that mirrors the server's local shell MCP server (--local-shell-mcp-server) */
export const cShellMcpServerId = 'dgs-shell';
export const cShellMcpServerName = 'local shell mcp server';
/** Tools the local shell MCP server can expose (only those of the shells it finds) */
export const aShellMcpToolNames: readonly string[] = ['bash.run', 'zsh.run', 'sh.run', 'psh.run'];
/** _meta key of tools/call that names the call, so the UI can answer the sudo password prompts of its command */
export const cShellMcpCallIdMeta = 'dgs/callId';
/** How often an executing shell call checks whether sudo is asking for a password */
export const cShellMcpAskpassPollMs = 500;

export const cShellMcpEndpoint = '/mcp/shell';
export const cShellMcpAskpassEndpoint = '/mcp/shell/askpass';
