/**
 * MCPHealthCheckManager - Health checks for MCP servers
 *
 * Owns per-server connectivity probes: connection reuse, capability
 * snapshots, and promotion of a successful check to an active connection.
 * Created and owned by mcpStore; the host owns the connection registry the
 * probes draw from and promote into.
 */

import { cDefaultMcpConfig } from '$lib/constants';
import { HealthCheckStatus, MCPConnectionPhase, MCPLogLevel } from '$lib/enums';
import { MCPService } from '$lib/services/mcp.service';
import type {
  ClientCapabilities,
  HealthCheckParams,
  HealthCheckState,
  MCPCapabilitiesInfo,
  MCPConnection,
  MCPConnectionLog,
  MCPServerConfig,
  ServerCapabilities
} from '$lib/types';
import { fDetectMcpTransportFromUrl } from '$lib/utils';

// module-level so the timestamp is not flagged as reactive state by prefer-svelte-reactivity
function fCreateConnectionErrorLog(pMessage: string): MCPConnectionLog {
  return {
    level: MCPLogLevel.ERROR,
    message: `Connection failed: ${pMessage}`,
    phase: MCPConnectionPhase.ERROR,
    timestamp: new Date()
  };
}

/**
 * The slice of mcpStore the probes drive. Kept narrow on purpose so the
 * probes cannot reach around the host's full surface; mcpStore implements
 * this structurally.
 */
export interface McpHealthHost {
  mAutoReconnect(pServerName: string): Promise<void>;
  mGetExistingConnection(pServerId: string): MCPConnection | undefined;
  mGetRequestTimeoutMs(): number;
  mPromoteHealthCheckToConnection(pServerId: string, pConnection: MCPConnection): void;
  mRegisterServerConfig(pName: string, pConfig: MCPServerConfig): void;
  mRemoveConnection(pServerId: string): void;
}

export class MCPHealthCheckManager {
  private dChecks = $state<Record<string, HealthCheckState>>({});

  /** Raw per-server check states, for host-side capability scans. */
  get checks(): Record<string, HealthCheckState> {
    return this.dChecks;
  }

  mClear(pServerId: string): void {
    const { [pServerId]: vRemoved, ...vRest } = this.dChecks;

    this.dChecks = vRest;
  }

  constructor(private pHost: McpHealthHost) {}

  mGetState(pServerId: string): HealthCheckState {
    return this.dChecks[pServerId] ?? { status: HealthCheckStatus.cIdle };
  }

  mHasState(pServerId: string): boolean {
    return pServerId in this.dChecks && this.dChecks[pServerId].status !== HealthCheckStatus.cIdle;
  }

  /**
   * Run a health check for a server.
   * If the server already has an active connection, reuses it instead of creating a new one.
   * If promoteToActive is true and server is enabled, the connection will be kept
   * and promoted to an active connection instead of being disconnected.
   */
  async mRun(pServer: HealthCheckParams, pPromoteToActive = false): Promise<void> {
    const cExistingConnection = this.pHost.mGetExistingConnection(pServer.id);

    if (cExistingConnection) {
      // Reuse existing connection - just refresh tools list
      try {
        const aTools = await MCPService.mListTools(cExistingConnection);
        const cCapabilities = this.mBuildCapabilitiesInfo(
          cExistingConnection.serverCapabilities,
          cExistingConnection.clientCapabilities
        );

        this.mSetState(pServer.id, {
          capabilities: cCapabilities,
          connectionTimeMs: cExistingConnection.connectionTimeMs,
          instructions: cExistingConnection.instructions,
          logs: [],
          protocolVersion: cExistingConnection.protocolVersion,
          serverInfo: cExistingConnection.serverInfo,
          status: HealthCheckStatus.SUCCESS,
          tools: aTools.map((pTool) => ({
            description: pTool.description,
            name: pTool.name,
            title: pTool.title
          })),
          transportType: cExistingConnection.transportType
        });

        return;
      } catch (error) {
        console.warn(
          `[MCPStore] Failed to reuse connection for ${pServer.id}, creating new one:`,
          error
        );
        // Connection may be stale, remove it and create new one
        this.pHost.mRemoveConnection(pServer.id);
      }
    }

    const cTrimmedUrl = pServer.url.trim();
    const aLogs: MCPConnectionLog[] = [];

    let vCurrentPhase: MCPConnectionPhase = MCPConnectionPhase.cIdle;

    if (!cTrimmedUrl) {
      this.mSetState(pServer.id, {
        logs: [],
        message: 'Please enter a server URL first.',
        status: HealthCheckStatus.ERROR
      });

      return;
    }

    this.mSetState(pServer.id, {
      logs: [],
      phase: MCPConnectionPhase.cTransportCreating,
      status: HealthCheckStatus.cConnecting
    });

    const cTimeoutMs = this.pHost.mGetRequestTimeoutMs();
    const dHeaders = this.mParseHeaders(pServer.headers);

    try {
      const cServerConfig: MCPServerConfig = {
        handshakeTimeoutMs: cDefaultMcpConfig.connectionTimeoutMs,
        headers: dHeaders,
        requestTimeoutMs: cTimeoutMs,
        transport: fDetectMcpTransportFromUrl(cTrimmedUrl),
        url: cTrimmedUrl,
        useProxy: pServer.useProxy
      };

      this.pHost.mRegisterServerConfig(pServer.id, cServerConfig);

      const cConnection = await MCPService.mConnect(
        pServer.id,
        cServerConfig,
        cDefaultMcpConfig.clientInfo,
        cDefaultMcpConfig.capabilities,
        (pPhase, pLog) => {
          vCurrentPhase = pPhase;
          aLogs.push(pLog);
          this.mSetState(pServer.id, {
            logs: [...aLogs],
            phase: pPhase,
            status: HealthCheckStatus.cConnecting
          });

          if (pPhase === MCPConnectionPhase.cDisconnected && pPromoteToActive) {
            console.log(
              `[MCPStore][${pServer.id}] Connection lost during health check, starting auto-reconnect`
            );
            this.pHost.mAutoReconnect(pServer.id);
          }
        }
      );
      const aTools = cConnection.tools.map((pTool) => ({
        description: pTool.description,
        name: pTool.name,
        title: pTool.title
      }));
      const cCapabilities = this.mBuildCapabilitiesInfo(
        cConnection.serverCapabilities,
        cConnection.clientCapabilities
      );

      this.mSetState(pServer.id, {
        capabilities: cCapabilities,
        connectionTimeMs: cConnection.connectionTimeMs,
        instructions: cConnection.instructions,
        logs: aLogs,
        protocolVersion: cConnection.protocolVersion,
        serverInfo: cConnection.serverInfo,
        status: HealthCheckStatus.SUCCESS,
        tools: aTools,
        transportType: cConnection.transportType
      });

      if (pPromoteToActive && pServer.enabled) {
        this.pHost.mPromoteHealthCheckToConnection(pServer.id, cConnection);
      } else {
        await MCPService.mDisconnect(cConnection);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error occurred';

      if (aLogs.at(-1)?.phase !== MCPConnectionPhase.ERROR) {
        aLogs.push(fCreateConnectionErrorLog(message));
      }

      this.mSetState(pServer.id, {
        logs: aLogs,
        message,
        phase: vCurrentPhase,
        status: HealthCheckStatus.ERROR
      });
    }
  }

  async mRunForServers(
    pServers: {
      id: string;
      enabled: boolean;
      url: string;
      headers?: string;
    }[],
    pSkipIfChecked = true,
    pPromoteToActive = false
  ): Promise<void> {
    const aServersToCheck = pSkipIfChecked
      ? pServers.filter((pS) => !this.mHasState(pS.id) && pS.url.trim())
      : pServers.filter((pS) => pS.url.trim());

    if (aServersToCheck.length === 0) {
      return;
    }

    const cBatchSize = 5;

    for (let vI = 0; vI < aServersToCheck.length; vI += cBatchSize) {
      const aBatch = aServersToCheck.slice(vI, vI + cBatchSize);

      await Promise.allSettled(aBatch.map((pServer) => this.mRun(pServer, pPromoteToActive)));
    }
  }

  /**
   * Builds capabilities info from server and client capabilities.
   */
  private mBuildCapabilitiesInfo(
    pServerCaps?: ServerCapabilities,
    pClientCaps?: ClientCapabilities
  ): MCPCapabilitiesInfo {
    return {
      client: {
        elicitation: pClientCaps?.elicitation
          ? { form: !!pClientCaps.elicitation.form, url: !!pClientCaps.elicitation.url }
          : undefined,
        roots: pClientCaps?.roots ? { listChanged: pClientCaps.roots.listChanged } : undefined,
        sampling: !!pClientCaps?.sampling,
        tasks: !!pClientCaps?.tasks
      },
      server: {
        completions: !!pServerCaps?.completions,
        logging: !!pServerCaps?.logging,
        prompts: pServerCaps?.prompts
          ? { listChanged: pServerCaps.prompts.listChanged }
          : undefined,
        resources: pServerCaps?.resources
          ? {
              listChanged: pServerCaps.resources.listChanged,
              subscribe: pServerCaps.resources.subscribe
            }
          : undefined,
        tasks: !!pServerCaps?.tasks,
        tools: pServerCaps?.tools ? { listChanged: pServerCaps.tools.listChanged } : undefined
      }
    };
  }

  private mParseHeaders(pHeadersJson?: string): Record<string, string> | undefined {
    if (!pHeadersJson?.trim()) {
      return undefined;
    }

    try {
      const cParsed = JSON.parse(pHeadersJson);

      if (typeof cParsed === 'object' && cParsed !== null && !Array.isArray(cParsed))
        return cParsed as Record<string, string>;
    } catch {
      console.warn('[MCPStore] Failed to parse custom headers JSON:', pHeadersJson);
    }

    return undefined;
  }

  private mSetState(pServerId: string, pState: HealthCheckState): void {
    this.dChecks = { ...this.dChecks, [pServerId]: pState };
  }
}
