/**
 * tabsStore - Reactive State Store for Browser-Style Conversation Tabs
 *
 * Tracks which conversations and the new-chat screen are open as tabs in
 * the chat layout, in order. Real conversation tabs are `#/chat/<id>`
 * routes; the new-chat tab is the bare `#/` route, represented here by the
 * `NEW_CHAT_TAB_ID` sentinel (see {@link NEW_CHAT_TAB_ID}).
 *
 * **Architecture & Relationships:**
 * - **conversationsStore**: owns conversation data; calls `removeTabs()` /
 *   `close()` when conversations are deleted. This store never imports it,
 *   so there is no circular dependency - tab names are resolved by the
 *   ChatTabs component from conversationsStore.
 * - Tab order persists to localStorage and is pruned against the loaded
 *   conversation list on init. The new-chat tab is kept across reloads.
 */

import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { cConversationTabsLocalstorageKey, cNewChatTabId, cRoutes } from '$lib/constants';
import { RouterService } from '$lib/services/router.service';
import { untrack } from 'svelte';

class TabsStore {
  /** Ordered tab ids: conversation ids and the `NEW_CHAT_TAB_ID` sentinel */
  aOpenTabs = $state<string[]>([]);

  /** False until init() has read the persisted tabs; save() is a no-op before that */
  private vInitialized = false;

  /** Navigate to a tab (the new-chat sentinel maps to the bare `#/` route) */
  async mActivate(pId: string): Promise<void> {
    await goto(pId === cNewChatTabId ? cRoutes.START : RouterService.mChat(pId));
  }

  /** Remove all tabs (e.g. after deleting all conversations) */
  mClear(): void {
    this.aOpenTabs = [];
    this.mSave();
  }

  /**
   * Close a tab. When it belongs to the active route, navigate to the left
   * neighbor (or the right one when the closed tab was leftmost), falling
   * back to the new-chat screen when no tabs remain.
   * @param id - Tab id to close
   * @param activeTabId - Tab id of the current route, if any
   */
  async mClose(pId: string, pActiveTabId: string | null): Promise<void> {
    const cIdx = this.aOpenTabs.indexOf(pId);

    if (cIdx === -1) {
      // tab not tracked (e.g. Conversation tabs are off); still fall back to
      // the new-chat screen when closing the active conversation
      if (pId === pActiveTabId) {
        await goto(cRoutes.START);
      }

      return;
    }

    this.aOpenTabs = this.aOpenTabs.filter((pTabId) => pTabId !== pId);
    this.mSave();

    if (pId !== pActiveTabId) return;

    const cTarget = (cIdx > 0 ? this.aOpenTabs[cIdx - 1] : this.aOpenTabs[0]) ?? null;

    if (cTarget) {
      await goto(cTarget === cNewChatTabId ? cRoutes.START : RouterService.mChat(cTarget));
    } else {
      await goto(cRoutes.START);
    }
  }

  /**
   * Load persisted tabs, dropping conversation ids that no longer exist.
   * Called once from initStores() after conversations are loaded.
   * Merges with (rather than replaces) current openTabs: the chat layout
   * syncs the route's tab before this async init completes, and replacing
   * here would drop it.
   * @param validIds - Ids of conversations present in the database
   */
  mInit(pValidIds: string[]): void {
    if (!browser) return;

    // the new-chat sentinel is a pseudo-tab, not a conversation, but it is
    // still kept so a reload on `#/` does not drop the tab the user is on
    const fIsLive = (pId: string) => pValidIds.includes(pId) || pId === cNewChatTabId;
    const aPersisted = this.load().filter(fIsLive);
    // tabs already in openTabs come from the live route, so they stay as they
    // are: `validIds` is a snapshot and a conversation created while the list
    // was loading is not in it
    const aExtras = this.aOpenTabs.filter((pId) => !aPersisted.includes(pId));

    this.aOpenTabs = [...aPersisted, ...aExtras];
    this.vInitialized = true;
    this.mSave();
  }

  /**
   * Remove tabs without navigating. Used when conversations are deleted
   * while some other conversation stays open.
   * @param ids - Tab ids to drop
   */
  mRemoveTabs(pIds: string[]): void {
    const sRemoved = new Set(pIds);
    const aNext = this.aOpenTabs.filter((pId) => !sRemoved.has(pId));

    if (aNext.length !== this.aOpenTabs.length) {
      this.aOpenTabs = aNext;
      this.mSave();
    }
  }

  /**
   * Sync the tab strip with the route. Called from the chat layout on every
   * navigation, so any way of reaching a conversation or new-chat tab opens
   * a tab for it.
   * @param id - The conversation (or temporary new-chat) id of the route
   */
  mSyncWithRoute(pId: string): void {
    // untrack: callers invoke this from an effect keyed on the route, and
    // reading openTabs here would subscribe that effect to openTabs too -
    // closing the active tab would then re-run the effect and re-add the tab
    untrack(() => {
      if (!this.aOpenTabs.includes(pId)) {
        this.aOpenTabs = [...this.aOpenTabs, pId];
        this.mSave();
      }
    });
  }

  private load(): string[] {
    try {
      const cRaw = localStorage.getItem(cConversationTabsLocalstorageKey);
      const cParsed: unknown = cRaw ? JSON.parse(cRaw) : [];

      return Array.isArray(cParsed) ? cParsed.filter((pId) => typeof pId === 'string') : [];
    } catch {
      return [];
    }
  }

  private mSave(): void {
    // never write before init has read the persisted tabs, or an early
    // route sync (layout effect runs before async init) would clobber them
    if (!browser || !this.vInitialized) return;

    localStorage.setItem(cConversationTabsLocalstorageKey, JSON.stringify(this.aOpenTabs));
  }
}

export const cTabsStore = new TabsStore();
