import { fTranslate } from '$lib/i18n';
/**
 * MCPService - Stateless MCP protocol layer
 *
 * Implements the client side of the MCP spec over WebSocket, StreamableHTTP
 * transports: connect, tool/prompt/resource operations and result
 * formatting. No reactive state; consumed by mcpStore and its managers.
 */

import { Client } from '@modelcontextprotocol/sdk/client';
import {
  StreamableHTTPClientTransport,
  StreamableHTTPError
} from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { WebSocketClientTransport } from '@modelcontextprotocol/sdk/client/websocket.js';
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js';
import type {
  GetPromptResult,
  ListChangedHandlers,
  Prompt,
  Tool
} from '@modelcontextprotocol/sdk/types.js';
import {
  cCorsProxy,
  cCorsProxyEndpoint,
  cDefaultClientVersion,
  cDefaultImageMimeType,
  cDefaultMcpConfig,
  HEADERS,
  cNewline
} from '$lib/constants';
import {
  MCPConnectionPhase,
  MCPContentType,
  MCPLogLevel,
  MCPRefType,
  MCPTransportType
} from '$lib/enums';
import type {
  ClientCapabilities,
  Implementation,
  MCPConnection,
  MCPConnectionLog,
  MCPPhaseCallback,
  MCPReadResourceResult,
  MCPResource,
  MCPResourceContent,
  MCPResourceIcon,
  MCPResourceTemplate,
  MCPServerConfig,
  MCPServerInfo,
  ToolCallParams,
  ToolExecutionResult
} from '$lib/types';
import {
  fBuildProxiedHeaders,
  fBuildProxiedUrl,
  fCreateBase64DataUrl,
  fExtractJsonRpcMethods,
  fFormatDiagnosticErrorMessage,
  fGetRequestBody,
  fGetRequestMethod,
  fGetRequestUrl,
  fIsAbortError,
  type RequestBodySummary,
  fSummarizeRequestBody,
  fThrowIfAborted
} from '$lib/utils';
import { fGetAuthHeaders, fSanitizeHeaders } from '$lib/utils/api-headers';

interface ToolResultContentItem {
  type: string;
  text?: string;
  data?: string;
  mimeType?: string;
  resource?: { text?: string; blob?: string; uri?: string };
}

interface ToolCallResult {
  content?: ToolResultContentItem[];
  structuredContent?: Record<string, unknown>;
  isError?: boolean;
  _meta?: Record<string, unknown>;
}

interface DiagnosticRequestDetails {
  url: string;
  method: string;
  credentials?: RequestCredentials;
  mode?: RequestMode;
  headers: Record<string, string>;
  body: RequestBodySummary;
  jsonRpcMethods?: string[];
}

export class MCPService {
  /**
   * Execute a tool call on a connection.
   * Supports abort signal for cancellable operations (e.g., when user stops generation).
   * Formats the raw tool result into a string representation.
   *
   * @param connection - The MCP connection to execute against
   * @param params - Tool name and arguments to execute
   * @param signal - Optional AbortSignal for cancellation support
   * @returns Formatted tool execution result with content string and error flag
   * @throws {Error} If tool execution fails or is aborted
   */
  static async mCallTool(
    pConnection: MCPConnection,
    pParams: ToolCallParams,
    pSignal?: AbortSignal
  ): Promise<ToolExecutionResult> {
    fThrowIfAborted(pSignal);

    try {
      const cResult = await pConnection.client.callTool(
        { arguments: pParams.arguments, name: pParams.name, _meta: pParams._meta },
        undefined,
        { signal: pSignal, timeout: pConnection.requestTimeoutMs }
      );

      return {
        content: this.mFormatToolResult(cResult as ToolCallResult),
        isError: (cResult as ToolCallResult).isError ?? false
      };
    } catch (error) {
      if (fIsAbortError(error)) {
        throw error;
      }

      // Let session-expired errors propagate unwrapped for reconnection handling
      if (this.mIsSessionExpiredError(error)) {
        throw error;
      }

      const message = error instanceof Error ? error.message : String(error);

      throw new Error(
        `Tool "${pParams.name}" execution failed on server "${pConnection.serverName}": ${message}`,
        { cause: error instanceof Error ? error : undefined }
      );
    }
  }

  /**
   * Request completion suggestions from a server.
   * Used for autocompleting prompt arguments or resource URI templates.
   *
   * @param connection - The MCP connection to use
   * @param ref - Reference to the prompt or resource template
   * @param argument - The argument being completed (name and current value)
   * @returns Completion result with suggested values
   */
  static async mComplete(
    pConnection: MCPConnection,
    pRef: { type: MCPRefType.PROMPT; name: string } | { type: MCPRefType.RESOURCE; uri: string },
    pArgument: { name: string; value: string }
  ): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
    try {
      const cResult = await pConnection.client.complete({
        argument: pArgument,
        ref: pRef
      });

      return cResult.completion;
    } catch (error) {
      console.error(`[MCPService] Failed to get completions:`, error);

      return null;
    }
  }

  /**
   * Connect to a single MCP server with detailed phase tracking.
   *
   * Performs the full MCP connection lifecycle:
   * 1. Transport creation (with automatic fallback)
   * 2. Client initialization and capability exchange
   * 3. Tool discovery via `listTools`
   *
   * Reports progress via `onPhase` callback at each step, enabling
   * UI progress indicators during connection.
   *
   * @param serverName - Display name for the server (used in logging)
   * @param serverConfig - Server URL, transport type, proxy, and auth configuration
   * @param clientInfo - Optional client identification (defaults to app info)
   * @param capabilities - Optional client capability declaration
   * @param onPhase - Optional callback for connection phase progress updates
   * @param listChangedHandlers - Optional handlers for server-initiated list change notifications
   * @returns Full connection object with client, transport, tools, server info, and timing
   * @throws {Error} If transport creation or connection fails
   */
  static async mConnect(
    pServerName: string,
    pServerConfig: MCPServerConfig,
    pClientInfo?: Implementation,
    pCapabilities?: ClientCapabilities,
    pOnPhase?: MCPPhaseCallback,
    pListChangedHandlers?: ListChangedHandlers
  ): Promise<MCPConnection> {
    const cStartTime = performance.now();
    const cEffectiveClientInfo = pClientInfo ?? cDefaultMcpConfig.clientInfo;
    const cEffectiveCapabilities = pCapabilities ?? cDefaultMcpConfig.capabilities;

    // Phase: Creating transport
    pOnPhase?.(
      MCPConnectionPhase.cTransportCreating,
      this.mCreateLog(
        MCPConnectionPhase.cTransportCreating,
        `Creating transport for ${pServerConfig.url}`
      )
    );

    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(`[MCPService][${pServerName}] Creating transport...`);
    }

    const {
      stopPhaseLogging: fStopPhaseLogging,
      transport: vTransport,
      type: vTransportType
    } = this.mCreateTransport(pServerName, pServerConfig, (pLog) => pOnPhase?.(pLog.phase, pLog));

    // Setup WebSocket reconnection handler
    if (vTransportType === MCPTransportType.cWebsocket) {
      vTransport.onclose = () => {
        console.log(`[MCPService][${pServerName}] WebSocket closed, notifying for reconnection`);
        pOnPhase?.(
          MCPConnectionPhase.cDisconnected,
          this.mCreateLog(MCPConnectionPhase.cDisconnected, 'WebSocket connection closed')
        );
      };
    }

    // Phase: Transport ready
    pOnPhase?.(
      MCPConnectionPhase.cTransportReady,
      this.mCreateLog(MCPConnectionPhase.cTransportReady, `Transport ready (${vTransportType})`),
      { transportType: vTransportType }
    );

    const cClient = new Client(
      {
        name: cEffectiveClientInfo.name,
        version: cEffectiveClientInfo.version ?? cDefaultClientVersion
      },
      {
        capabilities: cEffectiveCapabilities,
        listChanged: pListChangedHandlers
      }
    );
    const fRuntimeErrorHandler = (pError: Error) => {
      // the SDK reports any post initialize error here, including the abort we trigger
      // ourselves on the next health check cycle, on tab unload, or on server teardown.
      // these are lifecycle aborts, not actionable errors, so we keep them out of the red console.
      // the SDK wraps the original AbortError in a generic Error like
      //   "SSE stream disconnected: AbortError: The operation was aborted."
      // which isAbortError cannot recognize by name alone, so we also pattern match on the message
      if (fIsAbortError(pError)) {
        return;
      }

      const cMsg = pError?.message ?? '';

      if (
        /SSE stream disconnected:.*AbortError/i.test(cMsg) ||
        /AbortError: .*aborted/i.test(cMsg) ||
        /stream locked by a reader/i.test(cMsg)
      ) {
        return;
      }

      console.error(`[MCPService][${pServerName}] Protocol error after initialize:`, pError);
    };

    cClient.onerror = (pError) => {
      pOnPhase?.(
        MCPConnectionPhase.ERROR,
        this.mCreateLog(
          MCPConnectionPhase.ERROR,
          `Protocol error: ${pError.message}`,
          MCPLogLevel.ERROR,
          {
            error: this.mSummarizeError(pError)
          }
        )
      );
    };

    // Phase: Initializing
    pOnPhase?.(
      MCPConnectionPhase.cInitializing,
      this.mCreateLog(MCPConnectionPhase.cInitializing, 'Sending initialize request...')
    );

    // The SDK timeout only covers the initialize request, not transport.start(),
    // which can hang forever on an unreachable host (SSE endpoint wait, WebSocket
    // handshake, proxied fetch). This race bounds the whole handshake and closes
    // the transport on expiry so the underlying fetch or socket is aborted.
    const cHandshakeTimeoutMs =
      pServerConfig.handshakeTimeoutMs ?? cDefaultMcpConfig.connectionTimeoutMs;

    try {
      let vHandshakeTimer: ReturnType<typeof setTimeout> | undefined;

      const cHandshakeDeadline = new Promise<never>((p, pReject) => {
        vHandshakeTimer = setTimeout(() => {
          void vTransport.close().catch(() => {});
          pReject(
            new Error(`Connection timed out after ${Math.round(cHandshakeTimeoutMs / 1000)}s`)
          );
        }, cHandshakeTimeoutMs);
      });

      try {
        await Promise.race([
          cClient.connect(vTransport, { timeout: cHandshakeTimeoutMs }),
          cHandshakeDeadline
        ]);
      } finally {
        clearTimeout(vHandshakeTimer);
      }

      // Transport diagnostics are only for the initial handshake, not long-lived traffic.
      fStopPhaseLogging();
      cClient.onerror = fRuntimeErrorHandler;
    } catch (error) {
      cClient.onerror = fRuntimeErrorHandler;
      const url =
        (pServerConfig.useProxy ?? false)
          ? fBuildProxiedUrl(pServerConfig.url)
          : new URL(pServerConfig.url);

      pOnPhase?.(
        MCPConnectionPhase.ERROR,
        this.mCreateLog(
          MCPConnectionPhase.ERROR,
          `Connection failed during initialize: ${
            error instanceof Error ? error.message : String(error)
          }`,
          MCPLogLevel.ERROR,
          {
            browser: this.mGetBrowserContext(url, pServerConfig.useProxy ?? false),
            config: {
              configuredUrl: pServerConfig.url,
              credentials: pServerConfig.credentials,
              effectiveUrl: url.href,
              headers: fSanitizeHeaders(
                pServerConfig.headers,
                Object.keys(pServerConfig.headers ?? {}),
                HEADERS.PARTIAL_REDACT
              ),
              serverName: pServerName,
              transportType: vTransportType,
              useProxy: pServerConfig.useProxy ?? false
            },
            error: this.mSummarizeError(error),
            hints: this.mGetConnectionHints(url, pServerConfig, error)
          }
        )
      );

      throw error;
    }

    const cServerVersion = cClient.getServerVersion();
    const cServerCapabilities = cClient.getServerCapabilities();
    const cInstructions = cClient.getInstructions();
    const cServerInfo = this.mExtractServerInfo(cServerVersion);

    // Phase: Capabilities exchanged
    pOnPhase?.(
      MCPConnectionPhase.cCapabilitiesExchanged,
      this.mCreateLog(
        MCPConnectionPhase.cCapabilitiesExchanged,
        'Capabilities exchanged successfully',
        MCPLogLevel.INFO,
        {
          serverCapabilities: cServerCapabilities,
          serverInfo: cServerInfo
        }
      ),
      {
        clientCapabilities: cEffectiveCapabilities,
        instructions: cInstructions,
        serverCapabilities: cServerCapabilities,
        serverInfo: cServerInfo
      }
    );

    // Phase: Listing tools
    pOnPhase?.(
      MCPConnectionPhase.cListingTools,
      this.mCreateLog(MCPConnectionPhase.cListingTools, 'Listing available tools...')
    );

    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(`[MCPService][${pServerName}] Connected, listing tools...`);
    }

    const aTools = await this.mListTools({
      client: cClient,
      connectionTimeMs: 0,
      requestTimeoutMs:
        pServerConfig.requestTimeoutMs ?? cDefaultMcpConfig.requestTimeoutSeconds * 1000,
      serverName: pServerName,
      tools: [],
      transport: vTransport,
      transportType: vTransportType
    });
    const cConnectionTimeMs = Math.round(performance.now() - cStartTime);

    // Phase: Connected
    pOnPhase?.(
      MCPConnectionPhase.cConnected,
      this.mCreateLog(
        MCPConnectionPhase.cConnected,
        `Connection established with ${aTools.length} tools (${cConnectionTimeMs}ms)`
      )
    );

    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(
        `[MCPService][${pServerName}] Initialization complete with ${aTools.length} tools in ${cConnectionTimeMs}ms`
      );
    }

    return {
      client: cClient,
      clientCapabilities: cEffectiveCapabilities,
      connectionTimeMs: cConnectionTimeMs,
      instructions: cInstructions,
      protocolVersion: cDefaultMcpConfig.protocolVersion,
      requestTimeoutMs:
        pServerConfig.requestTimeoutMs ?? cDefaultMcpConfig.requestTimeoutSeconds * 1000,
      serverCapabilities: cServerCapabilities,
      serverInfo: cServerInfo,
      serverName: pServerName,
      tools: aTools,
      transport: vTransport,
      transportType: vTransportType
    };
  }

  /**
   * Create transport based on server configuration.
   * Supports WebSocket, StreamableHTTP (modern), transports.
   * When `useProxy` is enabled, routes HTTP requests through dgs-server's CORS proxy.
   *
   * **Fallback Order:**
   * 1. WebSocket — if explicitly configured (no CORS proxy support)
   * 2. StreamableHTTP — default for HTTP connections
   *
   * @param config - Server configuration with url, transport type, proxy, and auth settings
   * @returns Object containing the created transport and the transport type used
   * @throws {Error} If url is missing, WebSocket + proxy combination, or all transports fail
   */
  static mCreateTransport(
    pServerName: string,
    pConfig: MCPServerConfig,
    pOnLog?: (pLog: MCPConnectionLog) => void
  ): {
    transport: Transport;
    type: MCPTransportType;
    stopPhaseLogging: () => void;
  } {
    if (!pConfig.url) {
      throw new Error(fTranslate('text_44a9a196bae7'));
    }

    if (
      pConfig.transport !== undefined &&
      pConfig.transport !== MCPTransportType.cStreamableHttp &&
      pConfig.transport !== MCPTransportType.cWebsocket
    ) {
      throw new Error(fTranslate('dgs_bb0df66d6924'));
    }

    const cUseProxy = pConfig.useProxy ?? false;
    const cRequestInit: RequestInit = {};

    if (pConfig.headers) {
      cRequestInit.headers = pConfig.useProxy
        ? fBuildProxiedHeaders(pConfig.headers)
        : pConfig.headers;
    }

    if (cUseProxy) {
      cRequestInit.headers = {
        ...fGetAuthHeaders(),
        ...(cRequestInit.headers as Record<string, string>)
      };
    }

    if (
      !cUseProxy &&
      typeof window !== 'undefined' &&
      pConfig.transport !== MCPTransportType.cWebsocket &&
      new URL(pConfig.url).origin === window.location.origin
    ) {
      cRequestInit.headers = {
        ...fGetAuthHeaders(),
        ...(cRequestInit.headers as Record<string, string>)
      };
    }
    if (pConfig.credentials) {
      cRequestInit.credentials = pConfig.credentials;
    }

    if (pConfig.transport === MCPTransportType.cWebsocket) {
      if (cUseProxy) {
        throw new Error(fTranslate('text_79c12c60af8a'));
      }

      const url = new URL(pConfig.url);

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log(`[MCPService] Creating WebSocket transport for ${url.href}`);
      }

      return {
        stopPhaseLogging: () => {},
        transport: new WebSocketClientTransport(url),
        type: MCPTransportType.cWebsocket
      };
    }

    const url = cUseProxy ? fBuildProxiedUrl(pConfig.url) : new URL(pConfig.url);
    const { disable: fStopPhaseLogging, fetch: fDiagnosticFetch } = this.mCreateDiagnosticFetch(
      pServerName,
      pConfig,
      cRequestInit,
      url,
      cUseProxy,
      pOnLog
    );

    if (cUseProxy && import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(`[MCPService] Using CORS proxy for ${pConfig.url} -> ${url.href}`);
    }

    try {
      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log(`[MCPService] Creating StreamableHTTP transport for ${url.href}`);
      }

      return {
        stopPhaseLogging: fStopPhaseLogging,
        transport: new StreamableHTTPClientTransport(url, {
          fetch: fDiagnosticFetch,
          requestInit: cRequestInit
        }),
        type: MCPTransportType.cStreamableHttp
      };
    } catch (vHttpError) {
      fStopPhaseLogging();
      throw vHttpError;
    }
  }

  /**
   * Disconnect from a server.
   * Clears the `onclose` handler to prevent reconnection attempts on voluntary disconnect.
   *
   * @param connection - The active MCP connection to close
   */
  static async mDisconnect(pConnection: MCPConnection): Promise<void> {
    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log(`[MCPService][${pConnection.serverName}] Disconnecting...`);
    }

    try {
      // Terminate the session first for streamable-http transports to cleanly
      // close streams, matching the inspector's disconnect flow.
      if (pConnection.transport instanceof StreamableHTTPClientTransport) {
        await pConnection.transport.terminateSession();
      }

      // Clear error handlers before closing to prevent noise from expected
      // abort errors during shutdown. The inspector avoids this entirely
      // by not setting onerror, but since we use it for protocol logging,
      // we must clear it before disconnect.
      pConnection.client.onerror = undefined;

      if (pConnection.transport.onclose) {
        pConnection.transport.onclose = undefined;
      }

      await pConnection.client.close();
    } catch (error) {
      console.warn(`[MCPService][${pConnection.serverName}] Error during disconnect:`, error);
    }
  }

  /**
   * Get a specific prompt with arguments.
   * Unlike list operations, this throws on failure since the caller explicitly
   * requested a specific prompt and needs to handle the error.
   *
   * @param connection - The MCP connection to use
   * @param name - The prompt name to retrieve
   * @param args - Optional key-value arguments to pass to the prompt
   * @returns The prompt result with messages and metadata
   * @throws {Error} If the prompt retrieval fails
   */
  static async mGetPrompt(
    pConnection: MCPConnection,
    pName: string,
    pArgs?: Record<string, string>
  ): Promise<GetPromptResult> {
    try {
      return await pConnection.client.getPrompt({ arguments: pArgs, name: pName });
    } catch (error) {
      console.error(`[MCPService][${pConnection.serverName}] Failed to get prompt:`, error);

      throw error;
    }
  }

  /**
   * Detect if an error indicates an expired/invalidated MCP session.
   * Per MCP spec 2025-11-25: HTTP 404 means session invalidated, client MUST
   * discard its session ID and start a new session with a fresh initialize request.
   *
   * @param error - The caught error to inspect
   * @returns true if the error is a StreamableHTTP 404 (session not found)
   */
  static mIsSessionExpiredError(pError: unknown): boolean {
    return pError instanceof StreamableHTTPError && pError.code === 404;
  }

  /**
   * List all resources from a connection (handles pagination automatically).
   * @param connection - The MCP connection to use
   * @returns Array of all available resources
   */
  static async mListAllResources(pConnection: MCPConnection): Promise<MCPResource[]> {
    return this.mPaginate(
      pConnection,
      (pCursor) => this.mListResources(pConnection, pCursor),
      (pResult) => pResult.resources
    );
  }

  /**
   * List all resource templates from a connection (handles pagination automatically).
   * @param connection - The MCP connection to use
   * @returns Array of all available resource templates
   */
  static async mListAllResourceTemplates(
    pConnection: MCPConnection
  ): Promise<MCPResourceTemplate[]> {
    return this.mPaginate(
      pConnection,
      (pCursor) => this.mListResourceTemplates(pConnection, pCursor),
      (pResult) => pResult.resourceTemplates
    );
  }

  /**
   * List prompts from a connection.
   * Silently returns empty array on failure (logged as warning).
   *
   * @param connection - The MCP connection to query
   * @returns Array of available prompts, or empty array on error
   */
  static async mListPrompts(pConnection: MCPConnection): Promise<Prompt[]> {
    try {
      const cResult = await pConnection.client.listPrompts();

      return cResult.prompts ?? [];
    } catch (error) {
      // Let session-expired errors propagate for reconnection handling
      if (this.mIsSessionExpiredError(error)) {
        throw error;
      }

      console.warn(`[MCPService][${pConnection.serverName}] Failed to list prompts:`, error);

      return [];
    }
  }

  /**
   * List resources from a connection.
   * @param connection - The MCP connection to use
   * @param cursor - Optional pagination cursor
   * @returns Array of available resources and optional next cursor
   */
  static async mListResources(
    pConnection: MCPConnection,
    pCursor?: string
  ): Promise<{ resources: MCPResource[]; nextCursor?: string }> {
    try {
      const cResult = await pConnection.client.listResources(
        pCursor ? { cursor: pCursor } : undefined
      );

      return {
        nextCursor: cResult.nextCursor,
        resources: (cResult.resources ?? []) as MCPResource[]
      };
    } catch (error) {
      if (this.mIsSessionExpiredError(error)) {
        throw error;
      }

      console.warn(`[MCPService][${pConnection.serverName}] Failed to list resources:`, error);

      return { resources: [] };
    }
  }

  /**
   * List resource templates from a connection.
   * @param connection - The MCP connection to use
   * @param cursor - Optional pagination cursor
   * @returns Array of available resource templates and optional next cursor
   */
  static async mListResourceTemplates(
    pConnection: MCPConnection,
    pCursor?: string
  ): Promise<{ resourceTemplates: MCPResourceTemplate[]; nextCursor?: string }> {
    try {
      const cResult = await pConnection.client.listResourceTemplates(
        pCursor ? { cursor: pCursor } : undefined
      );

      return {
        nextCursor: cResult.nextCursor,
        resourceTemplates: (cResult.resourceTemplates ?? []) as MCPResourceTemplate[]
      };
    } catch (error) {
      if (this.mIsSessionExpiredError(error)) {
        throw error;
      }

      console.warn(
        `[MCPService][${pConnection.serverName}] Failed to list resource templates:`,
        error
      );

      return { resourceTemplates: [] };
    }
  }

  /**
   * List tools from a connection.
   * Silently returns empty array on failure (logged as warning).
   *
   * @param connection - The MCP connection to query
   * @returns Array of available tools, or empty array on error
   */
  static async mListTools(pConnection: MCPConnection): Promise<Tool[]> {
    try {
      const cResult = await pConnection.client.listTools();

      return cResult.tools ?? [];
    } catch (error) {
      // Let session-expired errors propagate for reconnection handling
      if (this.mIsSessionExpiredError(error)) {
        throw error;
      }

      console.warn(`[MCPService][${pConnection.serverName}] Failed to list tools:`, error);

      return [];
    }
  }

  /**
   * Read the contents of a resource.
   * @param connection - The MCP connection to use
   * @param uri - The URI of the resource to read
   * @returns The resource contents
   */
  static async mReadResource(
    pConnection: MCPConnection,
    pUri: string
  ): Promise<MCPReadResourceResult> {
    try {
      const cResult = await pConnection.client.readResource({ uri: pUri });

      return {
        _meta: cResult._meta,
        contents: (cResult.contents ?? []) as MCPResourceContent[]
      };
    } catch (error) {
      console.error(`[MCPService][${pConnection.serverName}] Failed to read resource:`, error);

      throw error;
    }
  }

  /**
   * Subscribe to updates for a resource.
   * The server will send notifications/resources/updated when the resource changes.
   * @param connection - The MCP connection to use
   * @param uri - The URI of the resource to subscribe to
   */
  static async mSubscribeResource(pConnection: MCPConnection, pUri: string): Promise<void> {
    try {
      await pConnection.client.subscribeResource({ uri: pUri });

      console.log(`[MCPService][${pConnection.serverName}] Subscribed to resource: ${pUri}`);
    } catch (error) {
      console.error(
        `[MCPService][${pConnection.serverName}] Failed to subscribe to resource:`,
        error
      );

      throw error;
    }
  }

  /**
   * Check if a connection supports resources.
   * Per MCP spec: presence of the `resources` key (even as empty object `{}`) indicates support.
   * Empty object means resources are supported but no sub-features (subscribe, listChanged).
   *
   * @param connection - The MCP connection to check
   * @returns Whether the server declares the resources capability
   */
  static mSupportsResources(pConnection: MCPConnection): boolean {
    // Per MCP spec: "Servers that support resources MUST declare the resources capability"
    // The presence of the key indicates support, even if it's an empty object
    return pConnection.serverCapabilities?.resources !== undefined;
  }

  /**
   * Check if a connection supports resource subscriptions.
   * @param connection - The MCP connection to check
   * @returns Whether the server supports resource subscriptions
   */
  static mSupportsResourceSubscriptions(pConnection: MCPConnection): boolean {
    return !!pConnection.serverCapabilities?.resources?.subscribe;
  }

  /**
   * Unsubscribe from updates for a resource.
   * @param connection - The MCP connection to use
   * @param uri - The URI of the resource to unsubscribe from
   */
  static async mUnsubscribeResource(pConnection: MCPConnection, pUri: string): Promise<void> {
    try {
      await pConnection.client.unsubscribeResource({ uri: pUri });

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log(`[MCPService][${pConnection.serverName}] Unsubscribed from resource: ${pUri}`);
      }
    } catch (error) {
      console.error(
        `[MCPService][${pConnection.serverName}] Failed to unsubscribe from resource:`,
        error
      );

      throw error;
    }
  }

  private static mAddRequestHeaders(
    pRequestHeaders: Headers,
    pHeaders: HeadersInit,
    pUseProxy: boolean
  ) {
    for (const [vKey, vValue] of new Headers(pHeaders).entries()) {
      const cProxiedKey =
        pUseProxy && !vKey.toLowerCase().startsWith(cCorsProxy.HEADER_PREFIX)
          ? `${cCorsProxy.HEADER_PREFIX}${vKey}`
          : vKey;

      pRequestHeaders.set(cProxiedKey, vValue);
    }
  }

  private static mCreateDiagnosticFetch(
    pServerName: string,
    pConfig: MCPServerConfig,
    pBaseInit: RequestInit,
    pTargetUrl: URL,
    pUseProxy: boolean,
    pOnLog?: (pLog: MCPConnectionLog) => void
  ): {
    fetch: typeof fetch;
    disable: () => void;
  } {
    let vEnabled = true;

    const fLogIfEnabled = (pLog: MCPConnectionLog) => {
      if (vEnabled) {
        pOnLog?.(pLog);
      }
    };

    return {
      disable: () => {
        vEnabled = false;
      },
      fetch: async (pInput, pInit) => {
        if (pUseProxy && typeof window !== 'undefined') {
          let vRequestUrlStr = '';

          if (typeof pInput === 'string') {
            vRequestUrlStr = pInput;
          } else if (pInput instanceof URL) {
            vRequestUrlStr = pInput.href;
          }

          if (vRequestUrlStr) {
            const cParsedRequestUrl = new URL(vRequestUrlStr, window.location.origin);

            if (
              cParsedRequestUrl.origin === window.location.origin &&
              !cParsedRequestUrl.pathname.includes(cCorsProxyEndpoint)
            ) {
              const cOriginalConfigUrl = new URL(pConfig.url);
              const cRealTargetUrl = new URL(
                cParsedRequestUrl.pathname + cParsedRequestUrl.search,
                cOriginalConfigUrl.origin
              );
              const cProxiedUrl = fBuildProxiedUrl(cRealTargetUrl.href);

              if (typeof pInput === 'string') {
                pInput = cProxiedUrl.href;
              } else if (pInput instanceof URL) {
                pInput = cProxiedUrl;
              }
            }
          }
        }

        const cStartedAt = performance.now();
        const cRequestHeaders = new Headers(pBaseInit.headers);

        if (typeof Request !== 'undefined' && pInput instanceof Request) {
          this.mAddRequestHeaders(cRequestHeaders, pInput.headers, pUseProxy);
        }

        if (pInit?.headers) {
          this.mAddRequestHeaders(cRequestHeaders, pInit.headers, pUseProxy);
        }

        const cRequest = this.mCreateDiagnosticRequestDetails(
          pInput,
          pInit,
          pBaseInit,
          cRequestHeaders,
          Object.keys(pConfig.headers ?? {})
        );
        const { method: vMethod, url: vUrl } = cRequest;

        fLogIfEnabled(
          this.mCreateLog(
            MCPConnectionPhase.cInitializing,
            `HTTP ${vMethod} ${vUrl}`,
            MCPLogLevel.INFO,
            {
              request: cRequest,
              serverName: pServerName
            }
          )
        );

        if (vMethod === 'DELETE' && vUrl.includes(cCorsProxyEndpoint)) {
          const cResponse = new Response(null, { status: 200, statusText: 'OK' });

          fLogIfEnabled(
            this.mCreateLog(
              MCPConnectionPhase.cInitializing,
              `HTTP 200 ${vMethod} ${vUrl} (fake response)`,
              MCPLogLevel.INFO,
              {
                response: {
                  durationMs: 0,
                  isFake: true,
                  status: cResponse.status,
                  statusText: cResponse.statusText,
                  url: vUrl
                }
              }
            )
          );

          // fake response, bypass real fetch()
          return cResponse;
        }

        try {
          const cResponse = await fetch(pInput, {
            ...pBaseInit,
            ...pInit,
            headers: cRequestHeaders
          });
          const cDurationMs = Math.round(performance.now() - cStartedAt);

          fLogIfEnabled(
            this.mCreateLog(
              MCPConnectionPhase.cInitializing,
              `HTTP ${cResponse.status} ${vMethod} ${vUrl} (${cDurationMs}ms)`,
              cResponse.ok ? MCPLogLevel.INFO : MCPLogLevel.cWarn,
              {
                response: {
                  durationMs: cDurationMs,
                  headers: fSanitizeHeaders(cResponse.headers, undefined, HEADERS.PARTIAL_REDACT),
                  status: cResponse.status,
                  statusText: cResponse.statusText,
                  url: vUrl
                }
              }
            )
          );

          return cResponse;
        } catch (error) {
          const cDurationMs = Math.round(performance.now() - cStartedAt);

          fLogIfEnabled(
            this.mCreateLog(
              MCPConnectionPhase.ERROR,
              `HTTP ${vMethod} ${vUrl} failed: ${fFormatDiagnosticErrorMessage(error)}`,
              MCPLogLevel.ERROR,
              {
                browser: this.mGetBrowserContext(pTargetUrl, pUseProxy),
                durationMs: cDurationMs,
                error: this.mSummarizeError(error),
                hints: this.mGetConnectionHints(pTargetUrl, pConfig, error),
                request: cRequest,
                serverName: pServerName
              }
            )
          );

          throw error;
        }
      }
    };
  }

  private static mCreateDiagnosticRequestDetails(
    pInput: RequestInfo | URL,
    pInit: RequestInit | undefined,
    pBaseInit: RequestInit,
    pRequestHeaders: Headers,
    pExtraRedactedHeaders?: Iterable<string>
  ): DiagnosticRequestDetails {
    const cBody = fGetRequestBody(pInput, pInit);
    const cDetails: DiagnosticRequestDetails = {
      body: fSummarizeRequestBody(cBody),
      credentials: pInit?.credentials ?? pBaseInit.credentials,
      headers: fSanitizeHeaders(pRequestHeaders, pExtraRedactedHeaders, HEADERS.PARTIAL_REDACT),
      method: fGetRequestMethod(pInput, pInit, pBaseInit).toUpperCase(),
      mode: pInit?.mode ?? pBaseInit.mode,
      url: fGetRequestUrl(pInput)
    };
    const cJsonRpcMethods = fExtractJsonRpcMethods(cBody);

    if (cJsonRpcMethods) {
      cDetails.jsonRpcMethods = cJsonRpcMethods;
    }

    return cDetails;
  }

  /**
   * Create a connection log entry for phase tracking.
   *
   * @param phase - The connection phase this log belongs to
   * @param message - Human-readable log message
   * @param level - Log severity level (default: INFO)
   * @param details - Optional structured details for debugging
   * @returns Formatted connection log entry
   */
  private static mCreateLog(
    pPhase: MCPConnectionPhase,
    pMessage: string,
    pLevel: MCPLogLevel = MCPLogLevel.INFO,
    pDetails?: unknown
  ): MCPConnectionLog {
    return {
      details: pDetails,
      level: pLevel,
      message: pMessage,
      phase: pPhase,
      timestamp: new Date()
    };
  }

  /**
   * Extract server info from SDK Implementation type.
   * Normalizes the SDK's server version response into our MCPServerInfo type.
   *
   * @param impl - Raw Implementation object from MCP SDK
   * @returns Normalized server info or undefined if input is empty
   */
  private static mExtractServerInfo(pImpl: Implementation | undefined): MCPServerInfo | undefined {
    if (!pImpl) {
      return undefined;
    }

    return {
      description: pImpl.description,
      icons: pImpl.icons?.map((pIcon: MCPResourceIcon) => ({
        mimeType: pIcon.mimeType,
        sizes: pIcon.sizes,
        src: pIcon.src,
        theme: pIcon.theme
      })),
      name: pImpl.name,
      title: pImpl.title,
      version: pImpl.version,
      websiteUrl: pImpl.websiteUrl
    };
  }

  private static mFormatSingleContent(pContent: ToolResultContentItem): string {
    if (pContent.type === MCPContentType.TEXT && pContent.text) {
      return pContent.text;
    }

    if (pContent.type === MCPContentType.IMAGE && pContent.data) {
      return fCreateBase64DataUrl(pContent.mimeType ?? cDefaultImageMimeType, pContent.data);
    }

    if (pContent.type === MCPContentType.RESOURCE && pContent.resource) {
      const resource = pContent.resource;

      if (resource.text) return resource.text;

      if (resource.blob) return resource.blob;

      return JSON.stringify(resource);
    }

    if (pContent.data && pContent.mimeType) {
      return fCreateBase64DataUrl(pContent.mimeType, pContent.data);
    }

    return JSON.stringify(pContent);
  }

  /**
   * Format tool result content items to a single string.
   * Handles text, image (base64 data URL), and embedded resource content types.
   *
   * @param result - Raw tool call result from MCP SDK
   * @returns Concatenated string representation of all content items
   */
  private static mFormatToolResult(pResult: ToolCallResult): string {
    const cContent = pResult.content;

    if (!Array.isArray(cContent)) return '';

    const cFormatted = cContent
      .map((pItem) => this.mFormatSingleContent(pItem))
      .filter(Boolean)
      .join(cNewline);

    if (cFormatted !== '') {
      return cFormatted;
    }

    if (pResult.structuredContent && typeof pResult.structuredContent === 'object') {
      return JSON.stringify(pResult.structuredContent);
    }

    return '';
  }

  private static mGetBrowserContext(
    pTargetUrl: URL,
    pUseProxy: boolean
  ): Record<string, unknown> | undefined {
    if (typeof window === 'undefined') {
      return undefined;
    }

    return {
      isSecureContext: window.isSecureContext,
      location: window.location.href,
      origin: window.location.origin,
      protocol: window.location.protocol,
      sameOrigin: window.location.origin === pTargetUrl.origin,
      targetOrigin: pTargetUrl.origin,
      targetProtocol: pTargetUrl.protocol,
      useProxy: pUseProxy
    };
  }

  private static mGetConnectionHints(
    pTargetUrl: URL,
    pConfig: MCPServerConfig,
    pError: unknown
  ): string[] {
    const aHints: string[] = [];
    const message = pError instanceof Error ? pError.message : String(pError);
    const aHeaderNames = Object.keys(pConfig.headers ?? {});

    if (typeof window !== 'undefined') {
      if (
        window.location.protocol === 'https:' &&
        pTargetUrl.protocol === 'http:' &&
        !pConfig.useProxy
      ) {
        aHints.push(
          'The page is running over HTTPS but the MCP server is HTTP. Browsers often block this as mixed content; enable the proxy or use HTTPS/WSS for the MCP server.'
        );
      }

      if (window.location.origin !== pTargetUrl.origin && !pConfig.useProxy) {
        aHints.push(
          'This is a cross-origin browser request. If the server is reachable from curl or Node but not from the browser, missing CORS headers are the most likely cause.'
        );
      }
    }

    if (aHeaderNames.length > 0) {
      aHints.push(
        `Custom request headers are configured (${aHeaderNames.join(', ')}). That triggers a CORS preflight, so the server must allow OPTIONS and include the matching Access-Control-Allow-Headers response.`
      );
    }

    if (pConfig.credentials && pConfig.credentials !== 'omit') {
      aHints.push(
        'Credentials are enabled for this connection. Cross-origin credentialed requests need Access-Control-Allow-Credentials: true and cannot use a wildcard Access-Control-Allow-Origin.'
      );
    }

    if (message.includes('Failed to fetch')) {
      aHints.push(
        '"Failed to fetch" is a browser-level network failure. Common causes are CORS rejection, mixed-content blocking, certificate/TLS errors, DNS failures, or nothing listening on the target port.'
      );
    }

    return aHints;
  }

  /**
   * Walk a cursor-paginated MCP list endpoint, collecting every page.
   */
  private static async mPaginate<T, R extends { nextCursor?: string }>(
    pConnection: MCPConnection,
    pFetchPage: (pCursor?: string) => Promise<R>,
    pExtract: (pResult: R) => T[]
  ): Promise<T[]> {
    const aAll: T[] = [];

    let vCursor: string | undefined;

    do {
      const cResult = await pFetchPage(vCursor);

      aAll.push(...pExtract(cResult));
      vCursor = cResult.nextCursor;
    } while (vCursor);

    return aAll;
  }

  private static mSummarizeError(pError: unknown): Record<string, unknown> {
    if (pError instanceof Error) {
      return {
        cause:
          pError.cause instanceof Error
            ? { message: pError.cause.message, name: pError.cause.name }
            : pError.cause,
        message: pError.message,
        name: pError.name,
        stack: pError.stack?.split('\n').slice(0, 6).join('\n')
      };
    }

    return { value: String(pError) };
  }
}
