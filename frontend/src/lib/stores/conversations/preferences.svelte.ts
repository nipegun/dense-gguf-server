import { SharedPreferencesService } from '$lib/services/shared-preferences.service';
/**
 * ConversationPreferences - Per-chat options with global fallback
 *
 * Owns the options that resolve per conversation: the tool policy (disabled
 * categories and tool keys), reasoning effort, and the working directory.
 * Tool picks made on the empty new-chat screen edit the global defaults
 * directly (they seed every newly created conversation); cwd and reasoning
 * effort are buffered as pending state and threaded into the next created
 * conversation by the host.
 * Created and owned by conversationsStore; the host owns the conversation
 * rows these options persist onto.
 */

import { cReasoningEffortDefaultLocalstorageKey } from '$lib/constants';
import { ReasoningEffort, ToolSource } from '$lib/enums';
import { DatabaseService } from '$lib/services/database.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cMcpStore } from '$lib/stores/mcp/index.svelte';
import { cToolsStore } from '$lib/stores/tools.svelte';
import type { DatabaseConversation, ToolEntry, ToolGroup } from '$lib/types';

/** Load reasoning effort default from localStorage, DEFAULT defers to the server */
function fLoadReasoningEffortDefault(): ReasoningEffort {
  if (typeof globalThis.localStorage === 'undefined') return ReasoningEffort.cDefault;

  try {
    const cRaw = SharedPreferencesService.mGetItem(cReasoningEffortDefaultLocalstorageKey);

    return (cRaw as ReasoningEffort) || ReasoningEffort.cDefault;
  } catch {
    return ReasoningEffort.cDefault;
  }
}

/** Persist reasoning effort default to localStorage */
function fSaveReasoningEffortDefault(pEffort: ReasoningEffort): void {
  if (typeof globalThis.localStorage === 'undefined') return;

  SharedPreferencesService.mSetItem(cReasoningEffortDefaultLocalstorageKey, pEffort);
}

/**
 * The slice of conversationsStore the preferences read and write. Kept narrow
 * on purpose so they cannot reach around the host's full surface;
 * conversationsStore implements this structurally.
 */
export interface ConversationsPreferencesHost {
  vActiveConversation: DatabaseConversation | null;
  aConversations: DatabaseConversation[];
  mApplyConversationUpdate(pId: string, pUpdates: Partial<DatabaseConversation>): void;
}

/**
 * Effective disabled tool keys: the active conversation row, or the global
 * defaults when there is no conversation. An existing row with an unset
 * field has an empty policy, not a fallback to defaults.
 */
function fBuildDisabledTools(pConv: DatabaseConversation | null): Set<string> {
  return new Set(pConv ? (pConv.disabledTools ?? []) : [...cToolsStore.disabledTools]);
}

/**
 * Effective disabled tool categories: the active conversation row, or the
 * global defaults when there is no conversation. An existing row with an
 * unset field has an empty policy, not a fallback to defaults.
 */
function fBuildDisabledToolCategories(pConv: DatabaseConversation | null): Set<ToolSource> {
  return new Set(
    pConv ? (pConv.disabledToolCategories ?? []) : [...cToolsStore.disabledToolCategories]
  );
}

export class ConversationPreferences {
  /**
   * Working directory picked on the empty new-chat screen, before any
   * conversation exists. Consumed by `chatStore.sendMessage()`, which
   * records it into chat history as a synthetic message on first send.
   * Cleared by `loadConversation` and `clearActiveConversation` so a
   * stale pick can't bleed onto an unrelated chat.
   */
  vPendingCwd = $state<string | null>(null);

  /** Global (non-conversation-specific) reasoning effort default */
  vPendingReasoningEffort = $state<ReasoningEffort>(fLoadReasoningEffortDefault());

  private get _disabledToolCategories(): Set<ToolSource> {
    return fBuildDisabledToolCategories(this.pHost.vActiveConversation);
  }

  // Tool Policy

  // getters, not $derived fields: lazy evaluation keeps them off the class
  // field initialization order (host is assigned by the constructor), and
  // reads of the underlying $state stay tracked in reactive contexts
  private get _disabledTools(): Set<string> {
    return fBuildDisabledTools(this.pHost.vActiveConversation);
  }

  mReloadSharedSettings(): void {
    this.vPendingReasoningEffort = fLoadReasoningEffortDefault();
  }

  constructor(private pHost: ConversationsPreferencesHost) {}

  /** Effective disabled tool categories for the current context, captured at flow start. */
  mGetDisabledToolCategories(): ToolSource[] {
    return [...this._disabledToolCategories];
  }

  /** Effective disabled tool keys for the current context, captured at flow start. */
  mGetDisabledTools(): string[] {
    return [...this._disabledTools];
  }

  /**
   * Gets the effective reasoning effort for the active conversation.
   * Returns the conversation override if set, otherwise the global default.
   * DEFAULT means no override is sent and the server decides.
   */
  mGetReasoningEffort(): ReasoningEffort {
    if (this.pHost.vActiveConversation) {
      if (this.pHost.vActiveConversation.reasoningEffort !== undefined) {
        return this.pHost.vActiveConversation.reasoningEffort;
      }

      // conversations created before the tri-state store an explicit
      // opt-out only as thinkingEnabled = false
      if (this.pHost.vActiveConversation.thinkingEnabled === false) {
        return ReasoningEffort.cOff;
      }
    }

    return this.vPendingReasoningEffort;
  }

  /** Defaults snapshot for seeding a newly created conversation. */
  mGetToolPolicySnapshot(): { disabledTools?: string[]; disabledToolCategories?: ToolSource[] } {
    const aDisabledTools = [...cToolsStore.disabledTools];
    const aDisabledToolCategories = [...cToolsStore.disabledToolCategories];

    return {
      disabledToolCategories: aDisabledToolCategories.length ? aDisabledToolCategories : undefined,
      disabledTools: aDisabledTools.length ? aDisabledTools : undefined
    };
  }

  mHasEnabledCwdTools(): boolean {
    return cToolsStore.mHasEnabledCwdTools(this._disabledTools, this._disabledToolCategories);
  }

  mIsCategoryEnabled(pSource: ToolSource): boolean {
    return !this._disabledToolCategories.has(pSource);
  }

  /** Group checkbox state: the category flag, or the server key for MCP groups. */
  mIsGroupChecked(pGroup: ToolGroup): boolean {
    return pGroup.source === ToolSource.cMcp && pGroup.serverId
      ? this.mIsServerToolsEnabled(pGroup.serverId)
      : this.mIsCategoryEnabled(pGroup.source);
  }

  /** Server-scoped MCP group state: one key disables all of that server's tools. */
  mIsServerToolsEnabled(pServerId: string): boolean {
    return this.mIsToolEnabled(cToolsStore.mGetMcpServerToolsKey(pServerId));
  }

  /** Effective state: own key, MCP server group key, and category all on. */
  mIsToolActive(pEntry: ToolEntry): boolean {
    return cToolsStore.mIsEntryEnabled(pEntry, this._disabledTools, this._disabledToolCategories);
  }

  /** Own-level state: the tool key itself, ignoring category and server group. */
  mIsToolEnabled(pKey: string): boolean {
    return !this._disabledTools.has(pKey);
  }

  /** True when a parent level (category or MCP server group) disables this entry. */
  mIsToolParentDisabled(pEntry: ToolEntry): boolean {
    if (!this.mIsCategoryEnabled(pEntry.source)) return true;

    return (
      pEntry.source === ToolSource.cMcp &&
      !!pEntry.serverId &&
      !this.mIsServerToolsEnabled(pEntry.serverId)
    );
  }

  /**
   * MCP servers usable under the effective policy: globally enabled, url set,
   * MCP category on and the server-scoped key not disabled.
   */
  mPolicyEnabledServerIds(): string[] {
    if (!this.mIsCategoryEnabled(ToolSource.cMcp)) return [];

    return cMcpStore
      .mGetServers()
      .filter(
        (pServer) => pServer.enabled && pServer.url.trim() && this.mIsServerToolsEnabled(pServer.id)
      )
      .map((pServer) => pServer.id);
  }

  /** Reload persisted defaults, e.g. when the active conversation is cleared. */
  mResetPending(): void {
    this.vPendingReasoningEffort = fLoadReasoningEffortDefault();
    this.vPendingCwd = null;
  }

  // Working Directory

  /**
   * Sets the working directory for the active conversation. Pass `null` or
   * an empty string to clear it, which restores the picker's empty state.
   *
   * On the empty new-chat screen (no active conversation yet), the value
   * is buffered into `pendingCwd` so the user can pick before
   * sending the first message; `createConversation()` consumes it.
   *
   * @param value - Absolute server-side path to the working directory, or null to clear
   */
  async mSetCwd(pValue: string | null): Promise<void> {
    const cTrimmed = pValue?.trim() || undefined;

    // No chat yet - buffer for the first chat the user creates.
    if (!this.pHost.vActiveConversation) {
      this.vPendingCwd = cTrimmed ?? null;

      return;
    }

    const cId = this.pHost.vActiveConversation.id;

    this.pHost.mApplyConversationUpdate(cId, {
      cwd: cTrimmed
    });

    await DatabaseService.mUpdateConversation(cId, {
      cwd: cTrimmed
    });

    this.vPendingCwd = null;
  }

  // Reasoning Effort

  /**
   * Sets the reasoning effort for the active conversation.
   * If no conversation exists, stores the global default.
   * @param effort - The effort level ('default' | 'off' | 'low' | 'medium' | 'high' | 'max')
   */
  async mSetReasoningEffort(pEffort: ReasoningEffort): Promise<void> {
    if (!this.pHost.vActiveConversation) {
      this.vPendingReasoningEffort = pEffort;
      fSaveReasoningEffortDefault(pEffort);

      return;
    }

    this.pHost.mApplyConversationUpdate(this.pHost.vActiveConversation.id, {
      reasoningEffort: pEffort
    });

    await DatabaseService.mUpdateConversation(this.pHost.vActiveConversation.id, {
      reasoningEffort: pEffort
    });
  }

  async mToggleCategory(pSource: ToolSource): Promise<void> {
    const cConv: DatabaseConversation | null = this.pHost.vActiveConversation;

    if (!cConv) {
      cToolsStore.mToggleCategory(pSource);

      return;
    }

    const sNext = fBuildDisabledToolCategories(cConv);

    if (sNext.has(pSource)) sNext.delete(pSource);
    else sNext.add(pSource);

    await this.mPersistDisabledToolCategories(sNext);
  }

  async mToggleGroup(pGroup: ToolGroup): Promise<void> {
    if (pGroup.source === ToolSource.cMcp && pGroup.serverId) {
      await this.mToggleServerTools(pGroup.serverId);
    } else {
      await this.mToggleCategory(pGroup.source);
    }
  }

  async mToggleServerTools(pServerId: string): Promise<void> {
    await this.mToggleTool(cToolsStore.mGetMcpServerToolsKey(pServerId));
  }

  async mToggleTool(pKey: string): Promise<void> {
    const cConv: DatabaseConversation | null = this.pHost.vActiveConversation;

    if (!cConv) {
      cToolsStore.mToggleTool(pKey);

      return;
    }

    const sNext = fBuildDisabledTools(cConv);

    if (sNext.has(pKey)) sNext.delete(pKey);
    else sNext.add(pKey);

    await this.mPersistDisabledTools(sNext);
  }

  private async mPersistDisabledToolCategories(pDisabled: Set<ToolSource>): Promise<void> {
    const cConv = this.pHost.vActiveConversation;

    if (!cConv) return;

    const cDisabledToolCategories = pDisabled.size ? [...pDisabled] : undefined;

    this.pHost.mApplyConversationUpdate(cConv.id, {
      disabledToolCategories: cDisabledToolCategories
    });

    await DatabaseService.mUpdateConversation(cConv.id, {
      disabledToolCategories: cDisabledToolCategories
    });
  }

  private async mPersistDisabledTools(pDisabled: Set<string>): Promise<void> {
    const cConv = this.pHost.vActiveConversation;

    if (!cConv) return;

    const cDisabledTools = pDisabled.size ? [...pDisabled] : undefined;

    this.pHost.mApplyConversationUpdate(cConv.id, { disabledTools: cDisabledTools });

    await DatabaseService.mUpdateConversation(cConv.id, { disabledTools: cDisabledTools });
  }
}
