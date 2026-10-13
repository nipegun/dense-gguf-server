import { SharedPreferencesService } from '$lib/services/shared-preferences.service';
/**
 * toolsStore - Tool registry and enablement
 *
 * Owns the server tool listing (with working-directory resolution), built-in
 * browser tools, MCP tools and per-tool enablement, exposed as a unified
 * tool set for the LLM and the tools UI. Consumed by the agentic loop and
 * the chat flows.
 */

import { browser } from '$app/environment';
import {
  fBuildBrowserInfoToolDefinition,
  fBuildGetDatetimeToolDefinition,
  fBuildReadMediaToolDefinition,
  cDisabledToolCategoriesLocalstorageKey,
  cDisabledToolKeysLocalstorageKey,
  cHomeTilde,
  cToolGroupLabels,
  cToolServerLabels
} from '$lib/constants';
import {
  BuiltInTool,
  GlobSearchType,
  HealthCheckStatus,
  JsonSchemaType,
  ToolCallType,
  ToolSource
} from '$lib/enums';
import { ToolsService } from '$lib/services/tools.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cMcpStore } from '$lib/stores/mcp/index.svelte';
import { cModelsStore } from '$lib/stores/models/index.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import type { OpenAIToolDefinition, ToolEntry, ToolGroup } from '$lib/types';
import { ApiError, fBuildSandboxToolDefinition } from '$lib/utils';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

/** Stable selection identity for a tool, shared by the disabled set and the permission store */

class ToolsStore {
  // default disabled tool categories, seeded into newly created conversations;
  // the per-conversation policy lives on the conversation row
  private vDisabledToolCategories = $state(new SvelteSet<ToolSource>());
  private vDisabledTools = $state(new SvelteSet<string>());
  private _error = $state<string | null>(null);
  private _loading = $state(false);
  private vServerHome = $state<string | null | undefined>(undefined);
  private aServerTools = $state<OpenAIToolDefinition[]>([]);
  private vToolsEndpointUnreachable = $state(false);
  // server tools that resolve their paths against the working directory,
  // as declared by the server in its `/tools` listing
  private vCwdAwareTools = $state(new SvelteSet<string>());

  get allToolDefinitions(): OpenAIToolDefinition[] {
    return this.allTools.map((pT) => pT.definition);
  }

  /** Canonical flat list of tool entries with source metadata and stable keys, deduped by key */
  get allTools(): ToolEntry[] {
    const entries: ToolEntry[] = [];
    const cSeen = new SvelteSet<string>();
    const fPush = (pEntry: ToolEntry) => {
      if (cSeen.has(pEntry.key)) return;

      cSeen.add(pEntry.key);
      entries.push(pEntry);
    };

    for (const cDef of this.aServerTools) {
      const name = cDef.function.name;

      fPush({
        definition: cDef,
        key: this.mToolKey(ToolSource.SERVER, name),
        source: ToolSource.SERVER
      });
    }

    for (const cDef of this.browserTools) {
      const name = cDef.function.name;

      fPush({
        definition: cDef,
        key: this.mToolKey(ToolSource.cBrowser, name),
        source: ToolSource.cBrowser
      });
    }

    for (const {
      definition: vDefinition,
      serverId: vServerId,
      serverName: vServerName
    } of this.mMcpEntries()) {
      const name = vDefinition.function.name;

      fPush({
        definition: vDefinition,
        key: this.mToolKey(ToolSource.cMcp, name, vServerId),
        serverId: vServerId,
        serverName: vServerName,
        source: ToolSource.cMcp
      });
    }

    for (const cDef of this.customTools) {
      const name = cDef.function.name;

      fPush({
        definition: cDef,
        key: this.mToolKey(ToolSource.cCustom, name),
        source: ToolSource.cCustom
      });
    }

    return entries;
  }

  get browserTools(): OpenAIToolDefinition[] {
    const aTools: OpenAIToolDefinition[] = [fBuildGetDatetimeToolDefinition()];

    if (cSettingsStore.vConfig.jsSandboxEnabled) {
      aTools.push(fBuildSandboxToolDefinition(!!cSettingsStore.vConfig.symbolicMathEnabled));
    }

    const cReadMedia = this.mReadMediaTool();

    if (cReadMedia) aTools.push(cReadMedia);

    // provide browser's get_info tool if server doesn't provide one
    if (!this.mHasServerTool(BuiltInTool.cServerGetInfo)) {
      aTools.push(fBuildBrowserInfoToolDefinition());
    }

    return aTools;
  }

  get customTools(): OpenAIToolDefinition[] {
    const cRaw = cSettingsStore.vConfig.customJson;

    if (!cRaw || typeof cRaw !== 'string') return [];

    try {
      const cParsed = JSON.parse(cRaw);

      if (!Array.isArray(cParsed)) return [];

      return cParsed.filter(
        (pT: unknown): pT is OpenAIToolDefinition =>
          typeof pT === 'object' &&
          pT !== null &&
          'type' in pT &&
          (pT as OpenAIToolDefinition).type === 'function' &&
          'function' in pT &&
          typeof (pT as OpenAIToolDefinition).function?.name === 'string'
      );
    } catch {
      return [];
    }
  }

  get disabledToolCategories(): ReadonlySet<ToolSource> {
    return this.vDisabledToolCategories;
  }

  get disabledTools(): SvelteSet<string> {
    return this.vDisabledTools;
  }

  get error(): string | null {
    return this._error;
  }

  get isToolsEndpointUnreachable(): boolean {
    return this.vToolsEndpointUnreachable;
  }

  get loading(): boolean {
    return this._loading;
  }

  get mcpTools(): OpenAIToolDefinition[] {
    return this.mMcpEntries().map((pE) => pE.definition);
  }

  get serverHome(): string | null {
    return this.vServerHome ?? null;
  }

  get serverTools(): OpenAIToolDefinition[] {
    return this.aServerTools;
  }

  /** Tools grouped by category for tree display, derived from the canonical entries */
  get toolGroups(): ToolGroup[] {
    const aGroups: ToolGroup[] = [];
    const cByKey = new SvelteMap<string, ToolGroup>();

    for (const cEntry of this.allTools) {
      const cGroupKey =
        cEntry.source === ToolSource.cMcp ? `mcp:${cEntry.serverId ?? ''}` : cEntry.source;

      let vGroup = cByKey.get(cGroupKey);

      if (!vGroup) {
        vGroup = {
          key: cGroupKey,
          label: this.mGroupLabel(cEntry),
          serverId: cEntry.serverId,
          source: cEntry.source,
          tools: []
        };
        cByKey.set(cGroupKey, vGroup);
        aGroups.push(vGroup);
      }

      vGroup.tools.push(cEntry);
    }

    return aGroups;
  }

  /** Enable all tools belonging to a specific MCP server */
  mEnableAllToolsForServer(pServerId: string): void {
    const cConnection = cMcpStore.mGetConnections().get(pServerId);

    if (!cConnection) return;

    // the server-scoped group key disables every tool regardless of per-tool keys
    this.vDisabledTools.delete(this.mGetMcpServerToolsKey(pServerId));

    for (const tool of cConnection.tools) {
      this.vDisabledTools.delete(this.mToolKey(ToolSource.cMcp, tool.name, pServerId));
    }

    this.mPersistDisabledTools();
  }

  async mFetchServerTools(): Promise<void> {
    if (this._loading) return;

    this._loading = true;
    this._error = null;
    this.vToolsEndpointUnreachable = false;

    try {
      const aToolInfos = await ToolsService.mList();

      this.aServerTools = aToolInfos.map((pInfo) => pInfo.definition);
      this.vCwdAwareTools = new SvelteSet(
        aToolInfos.filter((pInfo) => pInfo.uses_cwd).map((pInfo) => pInfo.tool)
      );
    } catch (vErr) {
      this._error = vErr instanceof Error ? vErr.message : String(vErr);

      // 403 from /tools means the server was started without --tools
      if (vErr instanceof ApiError && vErr.status === 403) {
        this.vToolsEndpointUnreachable = true;
        console.info('[ToolsStore] Server tools are disabled on the server');
      } else {
        console.error('[ToolsStore] Failed to fetch server tools:', vErr);
      }
    } finally {
      this._loading = false;
    }
  }

  /**
   * Enabled tool definitions for sending to the LLM. Callers pass an
   * explicit policy (the active conversation's, resolved with global
   * defaults when absent); without arguments the store defaults apply.
   * MCP tool schemas are normalized here so the wire payload is consistent
   * across all four sources (server, browser/sandbox, MCP, custom JSON).
   * The API identifies tools by name, so a name is sent at most once.
   */
  mGetEnabledToolsForLLM(
    pDisabledTools: ReadonlySet<string> = this.vDisabledTools,
    pDisabledCategories: ReadonlySet<ToolSource> = this.vDisabledToolCategories
  ): OpenAIToolDefinition[] {
    const cEnabledNames = new SvelteSet<string>();

    for (const cEntry of this.allTools) {
      if (this.mIsEntryEnabled(cEntry, pDisabledTools, pDisabledCategories)) {
        cEnabledNames.add(cEntry.definition.function.name);
      }
    }

    const aResult: OpenAIToolDefinition[] = [];
    const cSeen = new SvelteSet<string>();
    const fTake = (pDef: OpenAIToolDefinition) => {
      const name = pDef.function.name;

      if (!cEnabledNames.has(name) || cSeen.has(name)) return;

      cSeen.add(name);
      aResult.push(pDef);
    };

    for (const cDef of this.aServerTools) fTake(cDef);
    for (const cDef of this.browserTools) fTake(cDef);
    // mcpEntries() over mcpStore directly so wire shape stays normalized and aligned with the tools UI.
    for (const cEntry of this.mMcpEntries()) fTake(cEntry.definition);
    for (const cDef of this.customTools) fTake(cDef);

    return aResult;
  }

  /** Server-scoped tool key: disabling it disables all of that server's tools. */
  mGetMcpServerToolsKey(pServerId: string): string {
    return `mcp:${pServerId}`;
  }

  /** Permission key for a tool name, identical to the selection key */
  mGetPermissionKey(pToolName: string): string | null {
    return this.mFindEntryByName(pToolName)?.key ?? null;
  }

  /** Get the display label for the server that owns a given tool */
  mGetToolServerLabel(pToolName: string): string {
    const cEntry = this.mFindEntryByName(pToolName);

    if (!cEntry) return '';

    if (cEntry.serverName) return cMcpStore.mGetServerDisplayName(cEntry.serverName);

    if (cEntry.source === ToolSource.SERVER) return cToolServerLabels[ToolSource.SERVER];

    if (cEntry.source === ToolSource.cCustom) return cToolServerLabels[ToolSource.cCustom];

    if (cEntry.source === ToolSource.cBrowser) return cToolServerLabels[ToolSource.cBrowser];

    return '';
  }

  /** Determine the source of a tool by its name */
  mGetToolSource(pToolName: string): ToolSource | null {
    return this.mFindEntryByName(pToolName)?.source ?? null;
  }

  /**
   * Check if a working directory is worth setting: at least one server tool
   * that reads it is both served and left enabled by the given policy
   * (defaults to the global defaults).
   */
  mHasEnabledCwdTools(
    pDisabledTools: ReadonlySet<string> = this.vDisabledTools,
    pDisabledCategories: ReadonlySet<ToolSource> = this.vDisabledToolCategories
  ): boolean {
    if (pDisabledCategories.has(ToolSource.SERVER)) return false;

    return this.aServerTools.some((pDef) => {
      const name = pDef.function.name;

      return (
        this.vCwdAwareTools.has(name) && !pDisabledTools.has(this.mToolKey(ToolSource.SERVER, name))
      );
    });
  }

  /**
   * Load persisted disabled tools and fetch the builtin tool list.
   * Called by initStores() after migrations have run.
   */
  mInitialize(): void {
    // browser-only init: skip on SSR to avoid localStorage/fetch side effects
    if (!browser) return;

    try {
      const cStored = SharedPreferencesService.mGetItem(cDisabledToolKeysLocalstorageKey);

      if (cStored) {
        const cParsed = JSON.parse(cStored);

        if (Array.isArray(cParsed)) {
          for (const cKey of cParsed) {
            if (typeof cKey === 'string') this.vDisabledTools.add(cKey);
          }
        }
      }
    } catch (vErr) {
      console.error('[ToolsStore] Failed to load disabled tools from localStorage:', vErr);
    }

    try {
      const cStored = SharedPreferencesService.mGetItem(cDisabledToolCategoriesLocalstorageKey);

      if (cStored) {
        const cParsed = JSON.parse(cStored);

        if (Array.isArray(cParsed)) {
          for (const cKey of cParsed) {
            if (Object.values(ToolSource).includes(cKey)) {
              this.vDisabledToolCategories.add(cKey as ToolSource);
            }
          }
        }
      }
    } catch (vErr) {
      console.error(
        '[ToolsStore] Failed to load disabled tool categories from localStorage:',
        vErr
      );
    }

    this.mFetchServerTools();
  }

  mIsCategoryEnabled(pSource: ToolSource): boolean {
    return !this.vDisabledToolCategories.has(pSource);
  }

  mIsEntryEnabled(
    pEntry: ToolEntry,
    pDisabledTools: ReadonlySet<string>,
    pDisabledCategories: ReadonlySet<ToolSource>
  ): boolean {
    if (pDisabledCategories.has(pEntry.source)) return false;

    if (pDisabledTools.has(pEntry.key)) return false;

    if (pEntry.source === ToolSource.cMcp && pEntry.serverId) {
      return !pDisabledTools.has(this.mGetMcpServerToolsKey(pEntry.serverId));
    }

    return true;
  }

  mIsToolEnabled(pKey: string): boolean {
    return !this.vDisabledTools.has(pKey);
  }

  /**
   * Absolute home directory on the server, resolved once per session via
   * file_glob_search's `base` field (the server expands `~`). Anchors the
   * directory picker's search scope and the `~` abbreviation of cwd
   * displays. Returns null when tools are unavailable.
   */
  async mResolveServerHome(): Promise<string | null> {
    if (this.vServerHome !== undefined) return this.vServerHome;

    try {
      const dRes = await ToolsService.mExecuteToolRaw(BuiltInTool.cServerFileGlobSearch, {
        limit: 1,
        max_depth: 1,
        path: cHomeTilde,
        type: GlobSearchType.cDir
      });

      this.vServerHome = typeof dRes.base === 'string' ? dRes.base : null;
    } catch {
      // searches still work via a literal `~`, only `~` abbreviation degrades
      this.vServerHome = null;
    }

    return this.vServerHome;
  }

  mSetCategoryEnabled(pSource: ToolSource, pEnabled: boolean): void {
    if (pEnabled) {
      this.vDisabledToolCategories.delete(pSource);
    } else {
      this.vDisabledToolCategories.add(pSource);
    }

    this.mPersistDisabledToolCategories();
  }

  mSetToolEnabled(pKey: string, pEnabled: boolean): void {
    if (pEnabled) {
      this.vDisabledTools.delete(pKey);
    } else {
      this.vDisabledTools.add(pKey);
    }

    this.mPersistDisabledTools();
  }

  mToggleCategory(pSource: ToolSource): void {
    this.mSetCategoryEnabled(pSource, !this.mIsCategoryEnabled(pSource));
  }

  mToggleTool(pKey: string): void {
    this.mSetToolEnabled(pKey, !this.mIsToolEnabled(pKey));
  }

  /** First canonical entry matching a tool name, runtime tool calls resolve by name */
  private mFindEntryByName(pToolName: string): ToolEntry | null {
    for (const cEntry of this.allTools) {
      if (cEntry.definition.function.name === pToolName) return cEntry;
    }

    return null;
  }

  /** Get MCP tools from health check data, used when live connections aren't established yet */
  private mGetMcpToolsFromHealthChecks(): {
    serverId: string;
    serverName: string;
    tools: { name: string; description?: string }[];
  }[] {
    const aResult: ReturnType<ToolsStore['mGetMcpToolsFromHealthChecks']> = [];

    for (const server of cMcpStore.mGetServers()) {
      if (!server.enabled) continue;

      const cHealth = cMcpStore.mGetHealthCheckState(server.id);

      if (cHealth.status === HealthCheckStatus.SUCCESS && cHealth.tools.length > 0) {
        aResult.push({
          serverId: server.id,
          serverName: cMcpStore.mGetServerLabel(server),
          tools: cHealth.tools
        });
      }
    }

    return aResult;
  }

  private mGroupLabel(pEntry: ToolEntry): string {
    switch (pEntry.source) {
      case ToolSource.cMcp:
        return pEntry.serverName ?? '';
      case ToolSource.cCustom:
        return cToolGroupLabels[ToolSource.cCustom];
      case ToolSource.cBrowser:
        return cToolGroupLabels[ToolSource.cBrowser];
      default:
        return cToolGroupLabels[ToolSource.SERVER];
    }
  }

  private mHasServerTool(pName: BuiltInTool): boolean {
    return this.aServerTools.some((pDef) => pDef.function.name === pName);
  }

  private mInferTypeFromDefault(pValue: unknown): string | undefined {
    if (typeof pValue === 'string') return 'string';

    if (typeof pValue === 'boolean') return 'boolean';

    if (typeof pValue === 'number') return Number.isInteger(pValue) ? 'integer' : 'number';

    if (Array.isArray(pValue)) return 'array';

    if (pValue !== null && typeof pValue === 'object') return 'object';

    return undefined;
  }

  private mMcpDefinition(
    pName: string,
    pDescription: string | undefined,
    pSchema?: Record<string, unknown>
  ): OpenAIToolDefinition {
    return {
      function: {
        description: pDescription,
        name: pName,
        parameters: pSchema ?? { properties: {}, required: [], type: JsonSchemaType.cObject }
      },
      type: ToolCallType.cFunction
    };
  }

  /** Normalize MCP tools from live connections when available, fall back to health check data */
  private mMcpEntries(): {
    serverId: string;
    serverName: string;
    definition: OpenAIToolDefinition;
  }[] {
    const aOut: { serverId: string; serverName: string; definition: OpenAIToolDefinition }[] = [];
    const dConnections = cMcpStore.mGetConnections();

    if (dConnections.size > 0) {
      for (const [vServerId, vConnection] of dConnections) {
        const cServerName = cMcpStore.mGetServerDisplayName(vServerId);

        for (const tool of vConnection.tools) {
          const dRawSchema = (tool.inputSchema as Record<string, unknown>) ?? {
            properties: {},
            required: [],
            type: JsonSchemaType.cObject
          };

          aOut.push({
            definition: {
              function: {
                description: tool.description,
                name: tool.name,
                parameters: this.mNormalizeJsonSchema(dRawSchema)
              },
              type: ToolCallType.cFunction
            },
            serverId: vServerId,
            serverName: cServerName
          });
        }
      }
    } else {
      for (const {
        serverId: vServerId,
        serverName: vServerName,
        tools: aTools
      } of this.mGetMcpToolsFromHealthChecks()) {
        for (const tool of aTools) {
          aOut.push({
            definition: this.mMcpDefinition(tool.name, tool.description),
            serverId: vServerId,
            serverName: vServerName
          });
        }
      }
    }

    return aOut;
  }

  /**
   * Recursively normalize a JSON Schema object: infers `type` from `default`
   * for properties / items that omit it, and descends into nested `properties`
   * and `items`. Returns a new object -- does not mutate the input.
   */
  private mNormalizeJsonSchema(pSchema: Record<string, unknown>): Record<string, unknown> {
    if (!pSchema || typeof pSchema !== 'object') return pSchema;

    const dNormalized: Record<string, unknown> = { ...pSchema };

    if (dNormalized.properties && typeof dNormalized.properties === 'object') {
      const dProps = dNormalized.properties as Record<string, Record<string, unknown>>;
      const dNormalizedProps: Record<string, Record<string, unknown>> = {};

      for (const [vKey, dProp] of Object.entries(dProps)) {
        if (!dProp || typeof dProp !== 'object') {
          dNormalizedProps[vKey] = dProp;

          continue;
        }

        const dNormalizedProp: Record<string, unknown> = { ...dProp };

        if (!dNormalizedProp.type && dNormalizedProp.default !== undefined) {
          const cInferred = this.mInferTypeFromDefault(dNormalizedProp.default);

          if (cInferred) dNormalizedProp.type = cInferred;
        }

        if (dNormalizedProp.properties) {
          Object.assign(
            dNormalizedProp,
            this.mNormalizeJsonSchema(dNormalizedProp as Record<string, unknown>)
          );
        }

        if (dNormalizedProp.items && typeof dNormalizedProp.items === 'object') {
          dNormalizedProp.items = this.mNormalizeJsonSchema(
            dNormalizedProp.items as Record<string, unknown>
          );
        }

        dNormalizedProps[vKey] = dNormalizedProp;
      }
      dNormalized.properties = dNormalizedProps;
    }

    return dNormalized;
  }

  private mPersistDisabledToolCategories(): void {
    try {
      SharedPreferencesService.mSetItem(
        cDisabledToolCategoriesLocalstorageKey,
        JSON.stringify([...this.vDisabledToolCategories])
      );
    } catch {
      // ignore storage errors
    }
  }

  private mPersistDisabledTools(): void {
    try {
      SharedPreferencesService.mSetItem(
        cDisabledToolKeysLocalstorageKey,
        JSON.stringify([...this.vDisabledTools])
      );
    } catch {
      // ignore storage errors
    }
  }

  /**
   * `read_media` runs in the browser on top of the server's `read_file`, so it
   * exists only when that tool is served and the active model can perceive the
   * bytes. The server cannot make this call - it does not know which model the
   * conversation uses.
   */
  private mReadMediaTool(): OpenAIToolDefinition | null {
    if (!this.mHasServerTool(BuiltInTool.cServerReadFile)) return null;

    const model = cModelsStore.vSelectedModelName ?? cModelsStore.aModels[0]?.model ?? '';

    if (!model) return null;

    const vision = cModelsStore.props.mModelSupportsVision(model);
    const audio = cModelsStore.props.mModelSupportsAudio(model);

    if (!vision && !audio) return null;

    return fBuildReadMediaToolDefinition(vision, audio);
  }

  private mToolKey(pSource: ToolSource, pName: string, pServerId?: string): string {
    switch (pSource) {
      case ToolSource.cMcp:
        // with a serverId this is a per-tool key; without one it hits the
        // server group key shape, which no MCP entry ever does
        return pServerId ? `mcp-${pServerId}:${pName}` : this.mGetMcpServerToolsKey(pName);
      case ToolSource.cCustom:
        return `custom:${pName}`;
      case ToolSource.cBrowser:
        return `browser:${pName}`;
      default:
        return `server:${pName}`;
    }
  }
}

export const cToolsStore = new ToolsStore();
