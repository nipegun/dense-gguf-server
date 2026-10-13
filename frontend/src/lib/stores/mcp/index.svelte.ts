import { SvelteMap } from 'svelte/reactivity';
import { fApiUrl } from '$lib/api-url';
import {
  cShellMcpServerId,
  cShellMcpServerName,
  cShellMcpEndpoint,
  cShellMcpCallIdMeta
} from '$lib/constants';
import type { LocalShellActiveCall } from '$lib/types';
/**
 * mcpStore - MCP host: server connections and tool operations
 *
 * Implements the MCP "Host" role, coordinating multiple server connections
 * and exposing a unified tool interface: lifecycle, name-conflict detection
 * and automatic tool-to-server routing. Owns connection state and raw
 * `Tool[]` per server; the OpenAI-compatible wire format is built in
 * toolsStore. Composes the health-check manager; uses MCPService for the
 * protocol layer.
 */

import type { ListChangedHandlers } from '@modelcontextprotocol/sdk/types.js';
import { browser } from '$app/environment';
import { cSettingsKeys } from '$lib/constants';
import { cCache, cDefaultMcpConfig, cMcpReconnect, cMcpServerIdPrefix } from '$lib/constants';
import { ColorMode, HealthCheckStatus, MCPConnectionPhase, MCPRefType } from '$lib/enums';
import { MCPService } from '$lib/services/mcp.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { MCPHealthCheckManager, type McpHealthHost } from '$lib/stores/mcp/health.svelte';
import { cMcpResourceStore } from '$lib/stores/mcp/resources.svelte';
import { cServerStore } from '$lib/stores/server.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import type {
  GetPromptResult,
  HealthCheckParams,
  HealthCheckState,
  MCPClientConfig,
  MCPConnection,
  MCPPromptInfo,
  MCPResourceAttachment,
  MCPResourceContent,
  MCPServerConfig,
  MCPServerDisplayInfo,
  MCPServerSettingsEntry,
  MCPToolCall,
  ServerStatus,
  Tool,
  ToolExecutionResult
} from '$lib/types';
import type { DatabaseMessageExtraMcpResource } from '$lib/types/database';
import type { SettingsConfigType } from '$lib/types/settings';
import {
  fBuildProxiedIconUrl,
  fDetectMcpTransportFromUrl,
  fGetMcpIconUrl,
  fGetMcpServerFaviconFallback,
  fGetMcpServerLabel,
  fParseMcpServerSettings,
  fUuid
} from '$lib/utils';
import { mode } from 'mode-watcher';

class MCPStore implements McpHealthHost {
  private dShellCalls = new SvelteMap<string, LocalShellActiveCall>();
  mActiveShellCalls(): LocalShellActiveCall[] {
    return [...this.dShellCalls.values()];
  }

  private _error = $state<string | null>(null);
  private vIsInitializing = $state(false);
  private vToolCount = $state(0);
  private vActiveFlowCount = 0;

  private vConfigSignature: string | null = null;
  private aConnectedServers = $state<string[]>([]);
  private dConnections = new Map<string, MCPConnection>();
  // health checks: per-server connectivity probes with optional promotion to active connections
  private vHealth = new MCPHealthCheckManager(this);
  private vInitPromise: Promise<boolean> | null = null;
  private sReconnectingServers = new Set<string>(); // Guard against concurrent reconnections
  private dServerConfigs = new Map<string, MCPServerConfig>(); // Store configs for reconnection
  private vServersCache: { raw: unknown; servers: MCPServerSettingsEntry[] } | null = null;
  private dToolsIndex = new Map<string, string>();

  get availableTools(): string[] {
    return Array.from(this.dToolsIndex.keys());
  }

  get connectedServerCount(): number {
    return this.aConnectedServers.length;
  }

  get connectedServerNames(): string[] {
    return this.aConnectedServers;
  }

  get error(): string | null {
    return this._error;
  }

  get isEnabled(): boolean {
    const cMcpConfig = this.mBuildMcpClientConfig(cSettingsStore.vConfig);

    return (
      cMcpConfig !== null && cMcpConfig !== undefined && Object.keys(cMcpConfig.servers).length > 0
    );
  }

  get isInitialized(): boolean {
    return this.dConnections.size > 0;
  }

  get isInitializing(): boolean {
    return this.vIsInitializing;
  }

  get isProxyAvailable(): boolean {
    return cServerStore.props?.cors_proxy_enabled ?? false;
  }

  /** Resource state, composed here so consumers have a single MCP scope. */
  get resources() {
    return cMcpResourceStore;
  }

  get toolCount(): number {
    return this.vToolCount;
  }

  mAcquireConnection(): void {
    this.vActiveFlowCount++;
  }

  mAddServer(
    pServerData: Omit<MCPServerSettingsEntry, 'id'> & { id?: string }
  ): MCPServerSettingsEntry {
    const aServers = this.mGetServers();
    const cNewServer: MCPServerSettingsEntry = {
      displayName: pServerData.displayName,
      enabled: pServerData.enabled,
      headers: pServerData.headers?.trim() || undefined,
      id: pServerData.id || (fUuid() ?? `server-${Date.now()}`),
      name: pServerData.name,
      url: pServerData.url.trim(),
      useProxy: pServerData.useProxy
    };

    cSettingsStore.mUpdateConfig(
      cSettingsKeys.MCP_SERVERS,
      JSON.stringify([...aServers, cNewServer])
    );

    return cNewServer;
  }

  /**
   * Add a resource as attachment to chat context.
   * Automatically fetches content if not cached.
   */
  async mAttachResource(pUri: string): Promise<MCPResourceAttachment | null> {
    const cResourceInfo = cMcpResourceStore.mFindResourceByUri(pUri);

    if (!cResourceInfo) {
      console.error(`[MCPStore] Resource not found: ${pUri}`);

      return null;
    }

    if (cMcpResourceStore.mIsAttached(pUri)) {
      return null;
    }

    const attachment = cMcpResourceStore.mAddAttachment(cResourceInfo);

    try {
      const cContent = await this.mReadResource(pUri);

      if (cContent) {
        cMcpResourceStore.mUpdateAttachmentContent(attachment.id, cContent);
      } else {
        cMcpResourceStore.mUpdateAttachmentError(attachment.id, 'Failed to read resource');
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);

      cMcpResourceStore.mUpdateAttachmentError(attachment.id, message);
    }

    return cMcpResourceStore.mGetAttachment(attachment.id) ?? null;
  }

  /**
   * Auto-reconnect to a server with exponential backoff.
   * Continues indefinitely until successful.
   *
   * Race-condition safety: when the phase callback fires a DISCONNECTED event
   * while we are still inside this function (e.g., the server drops right after
   * a successful connect()), a naive inner `autoReconnect()` call would be
   * swallowed by the `reconnectingServers` guard, leaving the server
   * permanently disconnected once the outer call exits. We solve this by
   * deferring the new reconnection via the `needsReconnect` flag: the flag is
   * set inside the phase callback and honoured in the `finally` block after
   * the guard entry has been removed.
   */
  async mAutoReconnect(pServerName: string): Promise<void> {
    // Guard against concurrent reconnections
    if (this.sReconnectingServers.has(pServerName)) {
      console.log(`[MCPStore][${pServerName}] Reconnection already in progress, skipping`);

      return;
    }

    const cServerConfig = this.dServerConfigs.get(pServerName);

    if (!cServerConfig) {
      console.error(`[MCPStore] No config found for ${pServerName}, cannot reconnect`);

      return;
    }

    this.sReconnectingServers.add(pServerName);
    let vBackoff = cMcpReconnect.INITIAL_DELAY;
    // Flag set by the phase callback when a DISCONNECTED event fires while
    // reconnectingServers still holds this server (see JSDoc above).
    let vNeedsReconnect = false;

    try {
      while (true) {
        await new Promise((pResolve) => setTimeout(pResolve, vBackoff));

        console.log(`[MCPStore][${pServerName}] Auto-reconnecting...`);

        try {
          // Per-attempt timeout: reject if the server doesn't respond in time,
          // then fall through to backoff logic as with any other failure.
          const cTimeoutPromise = new Promise<never>((p, pReject) =>
            setTimeout(
              () =>
                pReject(
                  new Error(
                    `Reconnect attempt timed out after ${cMcpReconnect.ATTEMPT_TIMEOUT_MS}ms`
                  )
                ),
              cMcpReconnect.ATTEMPT_TIMEOUT_MS
            )
          );

          vNeedsReconnect = false;
          const cListChangedHandlers = this.mCreateListChangedHandlers(pServerName);
          const cConnectPromise = MCPService.mConnect(
            pServerName,
            cServerConfig,
            cDefaultMcpConfig.clientInfo,
            cDefaultMcpConfig.capabilities,
            (pPhase) => {
              if (pPhase === MCPConnectionPhase.cDisconnected) {
                if (this.sReconnectingServers.has(pServerName)) {
                  // Reconnect loop is active; defer to after it exits.
                  vNeedsReconnect = true;
                } else {
                  console.log(
                    `[MCPStore][${pServerName}] Connection lost, restarting auto-reconnect`
                  );
                  this.mAutoReconnect(pServerName);
                }
              }
            },
            cListChangedHandlers
          );
          const cConnection = await Promise.race([cConnectPromise, cTimeoutPromise]);

          this.dConnections.set(pServerName, cConnection);

          // Rebuild tool index for this server
          this.mIndexServerTools(pServerName, cConnection.tools);

          console.log(`[MCPStore][${pServerName}] Reconnected successfully`);

          break;
        } catch (error) {
          console.warn(`[MCPStore][${pServerName}] Reconnection failed:`, error);
          vBackoff = Math.min(vBackoff * cMcpReconnect.BACKOFF_MULTIPLIER, cMcpReconnect.MAX_DELAY);
        }
      }
    } finally {
      this.sReconnectingServers.delete(pServerName);

      // If the phase callback signalled a disconnect while this function held
      // the guard, kick off a fresh reconnect now that the guard is released.
      if (vNeedsReconnect) {
        console.log(
          `[MCPStore][${pServerName}] Deferred disconnect detected, restarting auto-reconnect`
        );
        this.mAutoReconnect(pServerName);
      }
    }
  }

  mClearError(): void {
    this._error = null;
  }

  mClearHealthCheck(pServerId: string): void {
    this.vHealth.mClear(pServerId);
  }

  /**
   * Clear all resource attachments.
   */
  mClearResourceAttachments(): void {
    cMcpResourceStore.mClearAttachments();
  }

  /**
   * Convert current resource attachments to DatabaseMessageExtra[] and clear them.
   * Called during message send to persist resources with the user message.
   */
  mConsumeResourceAttachmentsAsExtras(): DatabaseMessageExtraMcpResource[] {
    const aExtras = cMcpResourceStore.mToMessageExtras();

    if (aExtras.length > 0) {
      cMcpResourceStore.mClearAttachments();
    }

    return aExtras;
  }

  /**
   * Initialize every settings-enabled server. Policy filtering happens at tool
   * collection time, so switching conversation policies never re-initializes.
   */
  async mEnsureInitialized(): Promise<boolean> {
    if (!browser) {
      return false;
    }

    const cMcpConfig = this.mBuildMcpClientConfig(cSettingsStore.vConfig);
    const cSignature = cMcpConfig ? JSON.stringify(cMcpConfig) : null;

    if (!cSignature) {
      await this.mShutdown();

      return false;
    }

    if (this.isInitialized && this.vConfigSignature === cSignature) {
      return true;
    }

    if (this.vInitPromise && this.vConfigSignature === cSignature) {
      return this.vInitPromise;
    }

    if (this.dConnections.size > 0 || this.vInitPromise) await this.mShutdown();

    return this.mInitialize(cSignature, cMcpConfig!);
  }

  async mExecuteTool(pToolCall: MCPToolCall, pSignal?: AbortSignal): Promise<ToolExecutionResult> {
    return this.mExecuteToolByName(
      pToolCall.function.name,
      this.mParseToolArguments(pToolCall.function.arguments),
      pSignal
    );
  }

  async mExecuteToolByName(
    pToolName: string,
    pArgs: Record<string, unknown>,
    pSignal?: AbortSignal
  ): Promise<ToolExecutionResult> {
    const cServerName = this.dToolsIndex.get(pToolName);

    if (!cServerName) throw new Error(`Unknown tool: ${pToolName}`);

    const cConnection = this.dConnections.get(cServerName);

    if (!cConnection) throw new Error(`Server "${cServerName}" is not connected`);

    const vCallId = cServerName === cShellMcpServerId ? fUuid() : undefined;
    const dMeta = vCallId ? { [cShellMcpCallIdMeta]: vCallId } : undefined;
    if (vCallId)
      this.dShellCalls.set(vCallId, {
        id: vCallId,
        toolName: pToolName,
        command: typeof pArgs.command === 'string' ? pArgs.command : ''
      });
    try {
      return await MCPService.mCallTool(
        cConnection,
        { arguments: pArgs, name: pToolName, _meta: dMeta },
        pSignal
      );
    } catch (error) {
      if (MCPService.mIsSessionExpiredError(error)) {
        await this.mReconnectServer(cServerName);

        const cNewConnection = this.dConnections.get(cServerName);

        if (!cNewConnection) throw new Error(`Failed to reconnect to "${cServerName}"`);

        return await MCPService.mCallTool(
          cNewConnection,
          { arguments: pArgs, name: pToolName, _meta: dMeta },
          pSignal
        );
      }

      throw error;
    } finally {
      if (vCallId) this.dShellCalls.delete(vCallId);
    }
  }

  /**
   * Fetch resources from all connected servers that support them.
   * Updates mcpResourceStore with the results.
   * @param forceRefresh - If true, bypass cache and fetch fresh data
   */
  async mFetchAllResources(pForceRefresh: boolean = false): Promise<void> {
    const aServersWithResources = this.mGetServersWithResources();

    if (aServersWithResources.length === 0) {
      return;
    }

    // Check if we have cached resources and they're recent (unless force refresh)
    if (!pForceRefresh) {
      const cAllServersCached = aServersWithResources.every((pServerName) => {
        const cServerRes = cMcpResourceStore.mGetServerResources(pServerName);

        if (!cServerRes || !cServerRes.lastFetched) {
          return false;
        }

        // Cache is valid for 5 minutes
        const cAge = Date.now() - cServerRes.lastFetched.getTime();

        return cAge < cCache.DEFAULT_TTL_MS;
      });

      if (cAllServersCached) {
        console.log('[MCPStore] Using cached resources');

        return;
      }
    }

    cMcpResourceStore.mSetLoading(true);

    try {
      await Promise.all(
        aServersWithResources.map((pServerName) => this.mFetchServerResources(pServerName))
      );
    } finally {
      cMcpResourceStore.mSetLoading(false);
    }
  }

  /**
   * Fetch resources from a specific server.
   * Updates mcpResourceStore with the results.
   */
  async mFetchServerResources(pServerName: string): Promise<void> {
    const cConnection = this.dConnections.get(pServerName);

    if (!cConnection) {
      console.warn(`[MCPStore] No connection found for server: ${pServerName}`);

      return;
    }

    if (!MCPService.mSupportsResources(cConnection)) {
      return;
    }

    cMcpResourceStore.mSetServerLoading(pServerName, true);

    try {
      const [aResources, aTemplates] = await Promise.all([
        MCPService.mListAllResources(cConnection),
        MCPService.mListAllResourceTemplates(cConnection)
      ]);

      cMcpResourceStore.mSetServerResources(pServerName, aResources, aTemplates);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);

      cMcpResourceStore.mSetServerError(pServerName, message);
      console.error(`[MCPStore][${pServerName}] Failed to fetch resources:`, error);
    }
  }

  /**
   * Resolve which configured MCP server owns a given tool name. Looks at
   * active connections first (fast path), then falls back to per-server
   * health-check data so server-side MCP proxies (where dgs-server
   * executes MCP tools but the browser does not hold a direct connection)
   * still resolve tool names to their owning server.
   */
  mFindServerForTool(pToolName: string): string | undefined {
    const cFromIndex = this.dToolsIndex.get(pToolName);

    if (cFromIndex) return cFromIndex;

    for (const server of this.mGetServers()) {
      const cHealth = this.vHealth.checks[server.id];

      if (!cHealth || cHealth.status !== HealthCheckStatus.SUCCESS) continue;

      if (cHealth.tools.some((pTool) => pTool.name === pToolName)) {
        return server.id;
      }
    }

    return undefined;
  }
  mGetActiveFlowCount(): number {
    return this.vActiveFlowCount;
  }

  async mGetAllPrompts(): Promise<MCPPromptInfo[]> {
    const aResults: MCPPromptInfo[] = [];

    for (const [vServerName, vConnection] of this.dConnections) {
      if (!vConnection.serverCapabilities?.prompts) continue;

      const aPrompts = await MCPService.mListPrompts(vConnection);

      for (const prompt of aPrompts) {
        aResults.push({
          arguments: prompt.arguments?.map((pArg) => ({
            description: pArg.description,
            name: pArg.name,
            required: pArg.required
          })),
          description: prompt.description,
          name: prompt.name,
          serverName: vServerName,
          title: prompt.title
        });
      }
    }

    return aResults;
  }

  /**
   * Get all active MCP connections.
   * @returns Map of server names to connections
   */
  mGetConnections(): Map<string, MCPConnection> {
    return this.dConnections;
  }

  /**
   * Check if a server already has an active connection that can be reused.
   * Returns the existing connection if available.
   */
  mGetExistingConnection(pServerId: string): MCPConnection | undefined {
    return this.dConnections.get(pServerId);
  }

  /**
   * Get server instructions from health check results (for display before active connection).
   * Useful for showing instructions in settings UI.
   */
  mGetHealthCheckInstructions(): Array<{
    serverId: string;
    serverTitle?: string;
    instructions: string;
  }> {
    const aResults: Array<{ serverId: string; serverTitle?: string; instructions: string }> = [];

    for (const [vServerId, vState] of Object.entries(this.vHealth.checks)) {
      if (vState.status === HealthCheckStatus.SUCCESS && vState.instructions) {
        aResults.push({
          instructions: vState.instructions,
          serverId: vServerId,
          serverTitle: vState.serverInfo?.title || vState.serverInfo?.name
        });
      }
    }

    return aResults;
  }

  /**
   * Health checks live in MCPHealthCheckManager; these delegate so
   * consumers keep a single entry point.
   */
  mGetHealthCheckState(pServerId: string): HealthCheckState {
    return this.vHealth.mGetState(pServerId);
  }

  async mGetPrompt(
    pServerName: string,
    pPromptName: string,
    pArgs?: Record<string, string>
  ): Promise<GetPromptResult> {
    const cConnection = this.dConnections.get(pServerName);

    if (!cConnection)
      throw new Error(`Server "${pServerName}" not found for prompt "${pPromptName}"`);

    return MCPService.mGetPrompt(cConnection, pPromptName, pArgs);
  }

  async mGetPromptCompletions(
    pServerName: string,
    pPromptName: string,
    pArgumentName: string,
    pArgumentValue: string
  ): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
    const cConnection = this.dConnections.get(pServerName);

    if (!cConnection) {
      console.warn(`[MCPStore] Server "${pServerName}" is not connected`);

      return null;
    }

    if (!cConnection.serverCapabilities?.completions) {
      return null;
    }

    return MCPService.mComplete(
      cConnection,
      { name: pPromptName, type: MCPRefType.PROMPT },
      { name: pArgumentName, value: pArgumentValue }
    );
  }

  /**
   * Request timeout in milliseconds, read live from the global setting
   * so a change in Settings applies to every server immediately.
   */
  mGetRequestTimeoutMs(): number {
    const cSeconds =
      Number(cSettingsStore.vConfig.mcpRequestTimeoutSeconds) ||
      cDefaultMcpConfig.requestTimeoutSeconds;

    return Math.round(cSeconds * 1000);
  }

  /**
   * Get completions for a resource template argument.
   * Uses the MCP Completion API with ref/resource.
   */
  async mGetResourceCompletions(
    pServerName: string,
    pUriTemplate: string,
    pArgumentName: string,
    pArgumentValue: string
  ): Promise<{ values: string[]; total?: number; hasMore?: boolean } | null> {
    const cConnection = this.dConnections.get(pServerName);

    if (!cConnection) {
      console.warn(`[MCPStore] Server "${pServerName}" is not connected`);

      return null;
    }

    if (!cConnection.serverCapabilities?.completions) {
      return null;
    }

    return MCPService.mComplete(
      cConnection,
      { type: MCPRefType.RESOURCE, uri: pUriTemplate },
      { name: pArgumentName, value: pArgumentValue }
    );
  }

  /**
   * Get formatted resource context for chat.
   */
  mGetResourceContextForChat(): string {
    return cMcpResourceStore.mFormatAttachmentsForContext();
  }

  mSyncShellServer(pOffered: boolean): void {
    if (!browser) return;

    const vExisting = this.mGetServerById(cShellMcpServerId);
    if (!pOffered) {
      if (vExisting) this.mRemoveServer(cShellMcpServerId);
      return;
    }

    const vUrl = new URL(fApiUrl(cShellMcpEndpoint), window.location.origin).href;
    if (!vExisting) {
      this.mAddServer({
        id: cShellMcpServerId,
        enabled: true,
        url: vUrl,
        name: cShellMcpServerName,
        useProxy: false
      });
      return;
    }

    if (vExisting.url !== vUrl || vExisting.useProxy) {
      this.mUpdateServer(cShellMcpServerId, {
        url: vUrl,
        name: cShellMcpServerName,
        useProxy: false
      });
      this.mClearHealthCheck(cShellMcpServerId);
    } else if (vExisting.name !== cShellMcpServerName) {
      this.mUpdateServer(cShellMcpServerId, { name: cShellMcpServerName });
    }
  }

  mGetServerById(pServerId: string): MCPServerSettingsEntry | undefined {
    return this.mGetServers().find((pS) => pS.id === pServerId);
  }

  /**
   * Get display name for an MCP server by its ID.
   * Falls back to the server ID if server is not found.
   */
  mGetServerDisplayName(pServerId: string): string {
    const server = this.mGetServerById(pServerId);

    return server ? this.mGetServerLabel(server) : pServerId;
  }

  /**
   * Get icon URL for an MCP server by its ID.
   * Returns the best icon from the MCP server's `icons` array
   * (see MCP spec: spec.modelcontextprotocol.io).
   * Returns null if no icon is available.
   */
  mGetServerFavicon(pServerId: string): string | null {
    const server = this.mGetServerById(pServerId);

    if (!server) {
      return null;
    }

    const cIsDark = mode.current === ColorMode.DARK;
    const cHealthState = this.vHealth.mGetState(pServerId);

    if (cHealthState.status === HealthCheckStatus.SUCCESS && cHealthState.serverInfo?.icons) {
      const cMcpIconUrl = fGetMcpIconUrl(cHealthState.serverInfo.icons, cIsDark);

      if (cMcpIconUrl) {
        return this.mToRenderableIconUrl(cMcpIconUrl);
      }
    }

    return this.mToRenderableIconUrl(fGetMcpServerFaviconFallback(server.url));
  }

  /**
   * Resolve the favicon URL for an MCP server by one of its tool names.
   * Returns `null` if the tool is not provided by any configured MCP server,
   * or if the owning server has no icon to show.
   * Pair with {@link getServerFavicon} for direct server-id lookup.
   */
  mGetServerFaviconForTool(pToolName: string | undefined): string | null {
    if (!pToolName) return null;

    const cServerId = this.mFindServerForTool(pToolName);

    if (!cServerId) return null;

    return this.mGetServerFavicon(cServerId);
  }

  /**
   * Get aggregated server instructions from all connected servers.
   * Returns an array of { serverName, serverTitle, instructions } objects.
   */
  mGetServerInstructions(): Array<{
    serverName: string;
    serverTitle?: string;
    instructions: string;
  }> {
    const aResults: Array<{ serverName: string; serverTitle?: string; instructions: string }> = [];

    for (const [vServerName, vConnection] of this.dConnections) {
      if (vConnection.instructions) {
        aResults.push({
          instructions: vConnection.instructions,
          serverName: vServerName,
          serverTitle: vConnection.serverInfo?.title || vConnection.serverInfo?.name
        });
      }
    }

    return aResults;
  }

  mGetServerLabel(pServer: MCPServerDisplayInfo): string {
    return fGetMcpServerLabel(pServer, this.mGetServers(), this.vHealth.checks);
  }

  mGetServers(): MCPServerSettingsEntry[] {
    const cRaw = cSettingsStore.vConfig.mcpServers;

    // cache the parse: the config string rarely changes and getServers is
    // called from hot paths (per-tool display lookups, capability checks)
    if (this.vServersCache && this.vServersCache.raw === cRaw) {
      return this.vServersCache.servers;
    }

    const aServers = fParseMcpServerSettings(cRaw);

    this.vServersCache = { raw: cRaw, servers: aServers };

    return aServers;
  }

  mGetServersStatus(): ServerStatus[] {
    const aStatuses: ServerStatus[] = [];

    for (const [vName, vConnection] of this.dConnections) {
      aStatuses.push({
        error: undefined,
        isConnected: true,
        name: vName,
        toolCount: vConnection.tools.length
      });
    }

    return aStatuses;
  }

  /**
   * Get list of enabled servers that support resources.
   * Checks active connections first, then health check state as fallback.
   */
  mGetServersWithResources(): string[] {
    const sEnabledServerIds = new Set(
      this.mGetServers()
        .filter((pS) => pS.enabled)
        .map((pS) => pS.id)
    );
    const aServers: string[] = [];

    for (const [vName, vConnection] of this.dConnections) {
      if (!sEnabledServerIds.has(vName)) continue;

      if (MCPService.mSupportsResources(vConnection) && !aServers.includes(vName)) {
        aServers.push(vName);
      }
    }

    // Also check health check states for servers not yet connected
    for (const [vServerId, vState] of Object.entries(this.vHealth.checks)) {
      if (!sEnabledServerIds.has(vServerId)) continue;

      if (
        !aServers.includes(vServerId) &&
        vState.status === HealthCheckStatus.SUCCESS &&
        vState.capabilities?.server?.resources !== undefined
      ) {
        aServers.push(vServerId);
      }
    }

    return aServers;
  }

  mGetToolNames(): string[] {
    return Array.from(this.dToolsIndex.keys());
  }

  mGetToolServer(pToolName: string): string | undefined {
    return this.dToolsIndex.get(pToolName);
  }

  mHasAvailableServers(): boolean {
    return fParseMcpServerSettings(cSettingsStore.vConfig.mcpServers).some(
      (pS) => pS.enabled && pS.url.trim()
    );
  }

  mHasEnabledServers(): boolean {
    return Boolean(this.mBuildMcpClientConfig(cSettingsStore.vConfig));
  }

  /**
   * Check if any connected server has instructions.
   */
  mHasServerInstructions(): boolean {
    for (const cConnection of this.dConnections.values()) {
      if (cConnection.instructions) {
        return true;
      }
    }

    return false;
  }

  mHasTool(pToolName: string): boolean {
    return this.dToolsIndex.has(pToolName);
  }

  /**
   * Promote a health check connection to an active connection.
   * This avoids the need to reconnect when the server is needed for agentic flows.
   */
  mPromoteHealthCheckToConnection(pServerId: string, pConnection: MCPConnection): void {
    this.mIndexServerTools(pServerId, pConnection.tools);

    this.dConnections.set(pServerId, pConnection);

    this.mUpdateState({
      connectedServers: Array.from(this.dConnections.keys()),
      toolCount: this.dToolsIndex.size
    });
  }

  /**
   * Read resource content from a server.
   * Caches the result in mcpResourceStore.
   */
  async mReadResource(pUri: string): Promise<MCPResourceContent[] | null> {
    const cCached = cMcpResourceStore.mGetCachedContent(pUri);

    if (cCached) {
      return cCached.content;
    }

    // Find which server has this resource
    const cServerName = cMcpResourceStore.mFindServerForUri(pUri);

    if (!cServerName) {
      console.error(`[MCPStore] No server found for resource URI: ${pUri}`);

      return null;
    }

    const cConnection = this.dConnections.get(cServerName);

    if (!cConnection) {
      console.error(`[MCPStore] No connection found for server: ${cServerName}`);

      return null;
    }

    try {
      const cResult = await MCPService.mReadResource(cConnection, pUri);
      const cResourceInfo = cMcpResourceStore.mFindResourceByUri(pUri);

      if (cResourceInfo) {
        cMcpResourceStore.mCacheResourceContent(cResourceInfo, cResult.contents);
      }

      return cResult.contents;
    } catch (error) {
      console.error(`[MCPStore] Failed to read resource ${pUri}:`, error);

      return null;
    }
  }

  /**
   * Read a resource by an arbitrary URI (e.g., one expanded from a template).
   * Unlike readResource(), this does not require the URI to be in the resources list.
   */
  async mReadResourceByUri(
    pServerName: string,
    pUri: string
  ): Promise<MCPResourceContent[] | null> {
    const cConnection = this.dConnections.get(pServerName);

    if (!cConnection) {
      console.error(`[MCPStore] No connection found for server: ${pServerName}`);

      return null;
    }

    try {
      const cResult = await MCPService.mReadResource(cConnection, pUri);

      return cResult.contents;
    } catch (error) {
      console.error(`[MCPStore] Failed to read resource ${pUri}:`, error);

      return null;
    }
  }

  /** Store a server config so auto-reconnect can rebuild the session. */
  mRegisterServerConfig(pName: string, pConfig: MCPServerConfig): void {
    this.dServerConfigs.set(pName, pConfig);
  }

  /**
   * Release a connection reference.
   * By default, keeps connections alive for reuse (shutdownIfUnused=false).
   * MCP spec encourages long-lived sessions to avoid reconnection overhead.
   */
  async mReleaseConnection(pShutdownIfUnused = false): Promise<void> {
    this.vActiveFlowCount = Math.max(0, this.vActiveFlowCount - 1);

    if (pShutdownIfUnused && this.vActiveFlowCount === 0) {
      await this.mShutdown();
    }
  }

  /**
   * Drop a connection without disconnecting, e.g. when a health check finds
   * it stale and recreates it.
   */
  mRemoveConnection(pServerId: string): void {
    this.dConnections.delete(pServerId);
  }

  /**
   * Remove a resource attachment from chat context.
   */
  mRemoveResourceAttachment(pAttachmentId: string): void {
    cMcpResourceStore.mRemoveAttachment(pAttachmentId);
  }

  mRemoveServer(pId: string): void {
    const aServers = this.mGetServers();

    cSettingsStore.mUpdateConfig(
      cSettingsKeys.MCP_SERVERS,
      JSON.stringify(aServers.filter((pS) => pS.id !== pId))
    );
    this.mClearHealthCheck(pId);
  }

  async mRunHealthCheck(pServer: HealthCheckParams, pPromoteToActive = false): Promise<void> {
    return this.vHealth.mRun(pServer, pPromoteToActive);
  }

  async mRunHealthChecksForServers(
    pServers: {
      id: string;
      enabled: boolean;
      url: string;
      headers?: string;
    }[],
    pSkipIfChecked = true,
    pPromoteToActive = false
  ): Promise<void> {
    return this.vHealth.mRunForServers(pServers, pSkipIfChecked, pPromoteToActive);
  }

  async mShutdown(): Promise<void> {
    if (this.vInitPromise) {
      await this.vInitPromise.catch(() => {});
      this.vInitPromise = null;
    }

    if (this.dConnections.size === 0) {
      return;
    }

    await Promise.all(
      Array.from(this.dConnections.values()).map((pConn) =>
        MCPService.mDisconnect(pConn).catch((pError) =>
          console.warn(`[MCPStore] Error disconnecting ${pConn.serverName}:`, pError)
        )
      )
    );

    this.dConnections.clear();
    this.dToolsIndex.clear();
    this.dServerConfigs.clear();
    this.vConfigSignature = null;
    this.mUpdateState({
      connectedServers: [],
      error: null,
      isInitializing: false,
      toolCount: 0
    });
  }

  /**
   * Subscribe to resource updates.
   */
  async mSubscribeToResource(pUri: string): Promise<boolean> {
    const cServerName = cMcpResourceStore.mFindServerForUri(pUri);

    if (!cServerName) {
      console.error(`[MCPStore] No server found for resource URI: ${pUri}`);

      return false;
    }

    const cConnection = this.dConnections.get(cServerName);

    if (!cConnection) {
      console.error(`[MCPStore] No connection found for server: ${cServerName}`);

      return false;
    }

    if (!MCPService.mSupportsResourceSubscriptions(cConnection)) {
      return false;
    }

    try {
      await MCPService.mSubscribeResource(cConnection, pUri);
      cMcpResourceStore.mAddSubscription(pUri, cServerName);

      return true;
    } catch (error) {
      console.error(`[MCPStore] Failed to subscribe to resource ${pUri}:`, error);

      return false;
    }
  }

  /**
   * Unsubscribe from resource updates.
   */
  async mUnsubscribeFromResource(pUri: string): Promise<boolean> {
    const cServerName = cMcpResourceStore.mFindServerForUri(pUri);

    if (!cServerName) {
      console.error(`[MCPStore] No server found for resource URI: ${pUri}`);

      return false;
    }

    const cConnection = this.dConnections.get(cServerName);

    if (!cConnection) {
      console.error(`[MCPStore] No connection found for server: ${cServerName}`);

      return false;
    }

    try {
      await MCPService.mUnsubscribeResource(cConnection, pUri);
      cMcpResourceStore.mRemoveSubscription(pUri);

      return true;
    } catch (error) {
      console.error(`[MCPStore] Failed to unsubscribe from resource ${pUri}:`, error);

      return false;
    }
  }

  mUpdateServer(pId: string, pUpdates: Partial<MCPServerSettingsEntry>): void {
    const aServers = this.mGetServers();

    cSettingsStore.mUpdateConfig(
      cSettingsKeys.MCP_SERVERS,
      JSON.stringify(
        aServers.map((pServer) => (pServer.id === pId ? { ...pServer, ...pUpdates } : pServer))
      )
    );
  }

  /**
   * Builds MCP client configuration from settings.
   */
  private mBuildMcpClientConfig(pCfg: SettingsConfigType): MCPClientConfig | undefined {
    const aRawServers = fParseMcpServerSettings(pCfg.mcpServers);

    if (!aRawServers.length) {
      return undefined;
    }

    const dServers: Record<string, MCPServerConfig> = {};

    for (const [vIndex, vEntry] of aRawServers.entries()) {
      if (!vEntry.enabled) continue;

      const cNormalized = this.mBuildServerConfig(vEntry);

      if (cNormalized) dServers[this.mGenerateServerId(vEntry.id, vIndex)] = cNormalized;
    }

    if (Object.keys(dServers).length === 0) {
      return undefined;
    }

    return {
      capabilities: cDefaultMcpConfig.capabilities,
      clientInfo: cDefaultMcpConfig.clientInfo,
      protocolVersion: cDefaultMcpConfig.protocolVersion,
      requestTimeoutMs: this.mGetRequestTimeoutMs(),
      servers: dServers
    };
  }

  /**
   * Builds server configuration from a settings entry.
   */
  private mBuildServerConfig(
    pEntry: MCPServerSettingsEntry,
    pConnectionTimeoutMs = cDefaultMcpConfig.connectionTimeoutMs
  ): MCPServerConfig | undefined {
    if (!pEntry?.url) {
      return undefined;
    }

    let dHeaders: Record<string, string> | undefined;

    if (pEntry.headers) {
      try {
        const cParsed = JSON.parse(pEntry.headers);

        if (typeof cParsed === 'object' && cParsed !== null && !Array.isArray(cParsed))
          dHeaders = cParsed as Record<string, string>;
      } catch {
        console.warn('[MCP] Failed to parse custom headers JSON:', pEntry.headers);
      }
    }

    return {
      handshakeTimeoutMs: pConnectionTimeoutMs,
      headers: dHeaders,
      requestTimeoutMs: this.mGetRequestTimeoutMs(),
      transport: fDetectMcpTransportFromUrl(pEntry.url),
      url: pEntry.url,
      useProxy: pEntry.useProxy
    };
  }

  private mCreateListChangedHandlers(pServerName: string): ListChangedHandlers {
    return {
      prompts: {
        onChanged: (pError: Error | null) => {
          if (pError) {
            console.warn(`[MCPStore][${pServerName}] Prompts list changed error:`, pError);

            return;
          }
        }
      },
      tools: {
        onChanged: (pError: Error | null, pTools: Tool[] | null) => {
          if (pError) {
            console.warn(`[MCPStore][${pServerName}] Tools list changed error:`, pError);

            return;
          }

          this.mHandleToolsListChanged(pServerName, pTools ?? []);
        }
      }
    };
  }

  private async mDoInitialize(
    pSignature: string,
    pMcpConfig: MCPClientConfig,
    pServerEntries: [string, MCPClientConfig['servers'][string]][]
  ): Promise<boolean> {
    const cClientInfo = pMcpConfig.clientInfo ?? cDefaultMcpConfig.clientInfo;
    const cCapabilities = pMcpConfig.capabilities ?? cDefaultMcpConfig.capabilities;
    const aResults = await Promise.allSettled(
      pServerEntries.map(async ([vName, vServerConfig]) => {
        this.dServerConfigs.set(vName, vServerConfig);

        const cListChangedHandlers = this.mCreateListChangedHandlers(vName);
        const cConnection = await MCPService.mConnect(
          vName,
          vServerConfig,
          cClientInfo,
          cCapabilities,
          (pPhase) => {
            if (pPhase === MCPConnectionPhase.cDisconnected) {
              console.log(`[MCPStore][${vName}] Connection lost, starting auto-reconnect`);
              this.mAutoReconnect(vName);
            }
          },
          cListChangedHandlers
        );

        return { connection: cConnection, name: vName };
      })
    );

    if (this.vConfigSignature !== pSignature) {
      for (const cResult of aResults) {
        if (cResult.status === 'fulfilled')
          await MCPService.mDisconnect(cResult.value.connection).catch(console.warn);
      }

      return false;
    }

    for (const cResult of aResults) {
      if (cResult.status === 'fulfilled') {
        const { connection: vConnection, name: vName } = cResult.value;

        this.dConnections.set(vName, vConnection);

        this.mIndexServerTools(vName, vConnection.tools);
      } else {
        console.error(`[MCPStore] Failed to connect:`, cResult.reason);
      }
    }

    const cSuccessCount = this.dConnections.size;

    if (cSuccessCount === 0 && pServerEntries.length > 0) {
      this.mUpdateState({
        connectedServers: [],
        error: 'All MCP server connections failed',
        isInitializing: false,
        toolCount: 0
      });
      this.vInitPromise = null;

      return false;
    }

    this.mUpdateState({
      connectedServers: Array.from(this.dConnections.keys()),
      error: null,
      isInitializing: false,
      toolCount: this.dToolsIndex.size
    });
    this.vInitPromise = null;

    return true;
  }

  /**
   * Generates a unique server ID from an optional ID string or index.
   */
  private mGenerateServerId(pId: unknown, pIndex: number): string {
    if (typeof pId === 'string' && pId.trim()) {
      return pId.trim();
    }

    return `${cMcpServerIdPrefix}-${pIndex + 1}`;
  }

  /** Server ids that are usable right now: globally enabled ones. */
  private mGlobalEnabledServerIds(): Set<string> {
    return new Set(
      this.mGetServers()
        .filter((pS) => pS.enabled)
        .map((pS) => pS.id)
    );
  }

  private mHandleToolsListChanged(pServerName: string, pTools: Tool[]): void {
    const cConnection = this.dConnections.get(pServerName);

    if (!cConnection) {
      return;
    }

    for (const [vToolName, vOwnerServer] of this.dToolsIndex.entries()) {
      if (vOwnerServer === pServerName) this.dToolsIndex.delete(vToolName);
    }

    cConnection.tools = pTools;

    for (const tool of pTools) {
      if (this.dToolsIndex.has(tool.name))
        console.warn(
          `[MCPStore] Tool name conflict after list change: "${tool.name}" exists in "${this.dToolsIndex.get(tool.name)}" and "${pServerName}". Using tool from "${pServerName}".`
        );

      this.dToolsIndex.set(tool.name, pServerName);
    }
    this.mUpdateState({ toolCount: this.dToolsIndex.size });
  }

  /**
   * Registers the tools exposed by a server into the global name->server index,
   * warning on conflicts. Shared by connect, reconnect and auto-reconnect.
   */
  private mIndexServerTools(pServerName: string, pTools: Tool[]): void {
    for (const tool of pTools) {
      if (this.dToolsIndex.has(tool.name))
        console.warn(
          `[MCPStore] Tool name conflict: "${tool.name}" exists in "${this.dToolsIndex.get(tool.name)}" and "${pServerName}". Using tool from "${pServerName}".`
        );

      this.dToolsIndex.set(tool.name, pServerName);
    }
  }

  private async mInitialize(pSignature: string, pMcpConfig: MCPClientConfig): Promise<boolean> {
    this.mUpdateState({ error: null, isInitializing: true });
    this.vConfigSignature = pSignature;

    const aServerEntries = Object.entries(pMcpConfig.servers);

    if (aServerEntries.length === 0) {
      this.mUpdateState({ connectedServers: [], isInitializing: false, toolCount: 0 });

      return false;
    }

    this.vInitPromise = this.mDoInitialize(pSignature, pMcpConfig, aServerEntries);

    return this.vInitPromise;
  }

  private mParseToolArguments(pArgs: string | Record<string, unknown>): Record<string, unknown> {
    if (typeof pArgs === 'string') {
      const cTrimmed = pArgs.trim();

      if (cTrimmed === '') {
        return {};
      }

      try {
        const cParsed = JSON.parse(cTrimmed);

        if (typeof cParsed !== 'object' || cParsed === null || Array.isArray(cParsed))
          throw new Error(
            `Tool arguments must be an object, got ${Array.isArray(cParsed) ? 'array' : typeof cParsed}`
          );

        return cParsed as Record<string, unknown>;
      } catch (error) {
        throw new Error(`Failed to parse tool arguments as JSON: ${(error as Error).message}`);
      }
    }

    if (typeof pArgs === 'object' && pArgs !== null && !Array.isArray(pArgs)) {
      return pArgs;
    }

    throw new Error(`Invalid tool arguments type: ${typeof pArgs}`);
  }

  /**
   * Immediately reconnect to a server by creating a fresh transport and session.
   * Used when a session-expired error (HTTP 404) is detected during tool execution.
   * Per MCP spec 2025-11-25: client MUST discard session ID and re-initialize.
   *
   * Unlike autoReconnect (which uses exponential backoff for connectivity issues),
   * this performs a single immediate reconnection attempt since the server is known
   * to be reachable (it responded with 404).
   */
  private async mReconnectServer(pServerName: string): Promise<void> {
    const cServerConfig = this.dServerConfigs.get(pServerName);

    if (!cServerConfig) {
      throw new Error(`[MCPStore] No config found for ${pServerName}, cannot reconnect`);
    }

    // Disconnect stale connection (clears old transport + session ID)
    const cOldConnection = this.dConnections.get(pServerName);

    if (cOldConnection) {
      await MCPService.mDisconnect(cOldConnection).catch(console.warn);
      this.dConnections.delete(pServerName);
    }

    console.log(`[MCPStore][${pServerName}] Session expired, reconnecting with fresh session...`);

    const cListChangedHandlers = this.mCreateListChangedHandlers(pServerName);
    const cConnection = await MCPService.mConnect(
      pServerName,
      cServerConfig,
      cDefaultMcpConfig.clientInfo,
      cDefaultMcpConfig.capabilities,
      (pPhase) => {
        if (pPhase === MCPConnectionPhase.cDisconnected) {
          console.log(`[MCPStore][${pServerName}] Connection lost, starting auto-reconnect`);
          this.mAutoReconnect(pServerName);
        }
      },
      cListChangedHandlers
    );

    this.dConnections.set(pServerName, cConnection);
    this.mIndexServerTools(pServerName, cConnection.tools);

    console.log(`[MCPStore][${pServerName}] Session recovered successfully`);
  }

  /**
   * Icons come from the MCP server's own origin, which the browser cannot load
   * under the UI's cross-origin isolation. With the proxy enabled the server
   * fetches them instead; data URLs need no help and stay as they are.
   */
  private mToRenderableIconUrl(pIconUrl: string | null): string | null {
    return fBuildProxiedIconUrl(pIconUrl, this.isProxyAvailable);
  }

  private mUpdateState(pState: {
    isInitializing?: boolean;
    error?: string | null;
    toolCount?: number;
    connectedServers?: string[];
  }): void {
    if (pState.isInitializing !== undefined) {
      this.vIsInitializing = pState.isInitializing;
    }

    if (pState.error !== undefined) {
      this._error = pState.error;
    }

    if (pState.toolCount !== undefined) {
      this.vToolCount = pState.toolCount;
    }

    if (pState.connectedServers !== undefined) {
      this.aConnectedServers = pState.connectedServers;
    }
  }
}

export const cMcpStore = new MCPStore();
