import { cCliFlags } from '$lib/constants';
import { ToolSource } from '$lib/enums';
import { cConversationsStore, cMcpStore, cToolsStore } from '$lib/stores';
import type { ToolEntry, ToolGroup } from '$lib/types';
import { SvelteSet } from 'svelte/reactivity';

export interface UseToolsPanelReturn {
  readonly expandedGroups: SvelteSet<string>;
  readonly categoryGroups: ToolGroup[];
  readonly mcpGroups: ToolGroup[];
  readonly totalToolCount: number;
  readonly noToolsInfoMessage: string | null;
  mIsGroupChecked(pGroup: ToolGroup): boolean;
  mGetEnabledToolCount(pGroup: ToolGroup): number;
  mGetGroupCheckState(pGroup: ToolGroup): { checked: boolean; indeterminate: boolean };
  mGetFavicon(pGroup: ToolGroup): string | null;
  mIsGroupDisabled(pGroup: ToolGroup): boolean;
  mIsToolEnabled(pEntry: ToolEntry): boolean;
  mIsToolParentDisabled(pEntry: ToolEntry): boolean;
  mToggleTool(pEntry: ToolEntry): void;
  mToggleGroupExpanded(pKey: string): void;
  /** Toggle all tools in a group by its stable key (avoids stale group object references). */
  mToggleGroupByKey(pKey: string): void;
  mHandleOpen(): void;
}

/**
 * Shared reactive state and helpers for the tools panel UI.
 *
 * Used by both the desktop dropdown (`ChatFormActionAddToolsSubmenu`)
 * and the mobile drawer (`ChatFormActionAddDrawer`) to avoid
 * duplicating group filtering, checked-state derivation, and favicon logic.
 *
 * All toggle state routes through `conversationsStore.preferences`: with an
 * active conversation it edits that conversation's tool policy, on the
 * new-chat screen it edits the global defaults seeded into new conversations.
 */
export function fUseToolsPanel(): UseToolsPanelReturn {
  const cExpandedGroups = new SvelteSet<string>();
  const aGroups = $derived(cToolsStore.toolGroups);
  // non-MCP groups are 1:1 with tool categories; MCP tools group per server
  const aCategoryGroups = $derived(aGroups.filter((pG) => pG.source !== ToolSource.cMcp));
  const aMcpGroups = $derived(aGroups.filter((pG) => pG.source === ToolSource.cMcp));
  const cTotalToolCount = $derived(aGroups.reduce((pN, pG) => pN + pG.tools.length, 0));
  const cNoToolsInfoMessage = $derived.by(() => {
    if (cToolsStore.loading) return null;

    if (cToolsStore.toolGroups.length > 0) return null;

    // Tools endpoint unreachable (403) — server started without tools
    if (cToolsStore.isToolsEndpointUnreachable) {
      return `To enable Server Tools you need to run dgs-server with ${cCliFlags.TOOLS} all or ${cCliFlags.TOOLS} <name> flag. To see MCP Tools you need to add / enable MCP Server(s).`;
    }

    // Other errors — return null so UI shows "Failed to load tools"
    if (cToolsStore.error) return null;

    return `To enable Server Tools you need to run dgs-server with ${cCliFlags.TOOLS} all or ${cCliFlags.TOOLS} <name> flag. To see MCP Tools you need to add / enable MCP Server(s).`;
  });

  function fIsGroupChecked(pGroup: ToolGroup): boolean {
    return cConversationsStore.preferences.mIsGroupChecked(pGroup);
  }

  function fGetEnabledToolCount(pGroup: ToolGroup): number {
    return pGroup.tools.filter((pTool) => cConversationsStore.preferences.mIsToolActive(pTool))
      .length;
  }

  /**
   * Group checkbox state: checked is the parent flag (category on, or the
   * server key on for MCP groups); indeterminate marks the mixed case where
   * the parent is on but nothing or only part of the group is enabled.
   * isToolActive folds the parent gates into the count, so a disabled parent
   * always yields plain unchecked.
   */
  function fGetGroupCheckState(pGroup: ToolGroup): { checked: boolean; indeterminate: boolean } {
    const cChecked = fIsGroupChecked(pGroup);
    const cEnabledCount = fGetEnabledToolCount(pGroup);
    const cIndeterminate =
      pGroup.tools.length > 0 &&
      (cEnabledCount === 0 ? cChecked : cEnabledCount < pGroup.tools.length);

    return { checked: cChecked, indeterminate: cIndeterminate };
  }

  function fGetFavicon(pGroup: ToolGroup): string | null {
    if (pGroup.source !== ToolSource.cMcp || !pGroup.serverId) return null;

    return cMcpStore.mGetServerFavicon(pGroup.serverId);
  }

  function fIsGroupDisabled(pGroup: ToolGroup): boolean {
    // MCP server groups gray out while the whole MCP category is off
    return (
      pGroup.source === ToolSource.cMcp &&
      !cConversationsStore.preferences.mIsCategoryEnabled(ToolSource.cMcp)
    );
  }

  function fIsToolEnabled(pEntry: ToolEntry): boolean {
    return cConversationsStore.preferences.mIsToolEnabled(pEntry.key);
  }

  function fIsToolParentDisabled(pEntry: ToolEntry): boolean {
    return cConversationsStore.preferences.mIsToolParentDisabled(pEntry);
  }

  function fToggleTool(pEntry: ToolEntry): void {
    void cConversationsStore.preferences.mToggleTool(pEntry.key);
  }

  function fToggleGroupExpanded(pKey: string): void {
    if (cExpandedGroups.has(pKey)) {
      cExpandedGroups.delete(pKey);
    } else {
      cExpandedGroups.add(pKey);
    }
  }

  function fToggleGroupByKey(pKey: string): void {
    // Find current group by key to get up-to-date tool references
    const cGroup = aGroups.find((pG) => pG.key === pKey);

    if (!cGroup) return;

    void cConversationsStore.preferences.mToggleGroup(cGroup);
  }

  function fHandleOpen(): void {
    if (cToolsStore.serverTools.length === 0 && !cToolsStore.loading) {
      cToolsStore.mFetchServerTools();
    }

    cMcpStore.mRunHealthChecksForServers(cMcpStore.mGetServers().filter((pS) => pS.enabled));
  }

  return {
    get categoryGroups() {
      return aCategoryGroups;
    },
    expandedGroups: cExpandedGroups,
    mGetEnabledToolCount: fGetEnabledToolCount,
    mGetFavicon: fGetFavicon,
    mGetGroupCheckState: fGetGroupCheckState,
    mHandleOpen: fHandleOpen,
    mIsGroupChecked: fIsGroupChecked,
    mIsGroupDisabled: fIsGroupDisabled,
    mIsToolEnabled: fIsToolEnabled,
    mIsToolParentDisabled: fIsToolParentDisabled,
    get mcpGroups() {
      return aMcpGroups;
    },
    get noToolsInfoMessage() {
      return cNoToolsInfoMessage;
    },
    mToggleGroupByKey: fToggleGroupByKey,
    mToggleGroupExpanded: fToggleGroupExpanded,
    mToggleTool: fToggleTool,
    get totalToolCount() {
      return cTotalToolCount;
    }
  };
}
