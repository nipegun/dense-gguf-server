import { fTranslate } from '$lib/i18n';
/**
 * conversationsStore - Conversation lifecycle, persistence and navigation
 *
 * Owns conversation CRUD, message tree navigation, import/export and title
 * management, persisted through DatabaseService. Per-chat options (MCP
 * overrides, reasoning effort, cwd) live in ConversationPreferences,
 * composed as {@link ConversationsStore.preferences}.
 */

import { browser } from '$app/environment';
import { goto } from '$app/navigation';
import { cRoutes } from '$lib/constants';
import { MessageRole } from '$lib/enums';
import { ConversationTransferService } from '$lib/services/conversation-transfer.service';
import { DatabaseService } from '$lib/services/database.service';
import { RouterService } from '$lib/services/router.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import {
  ConversationPreferences,
  type ConversationsPreferencesHost
} from '$lib/stores/conversations/preferences.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import { cTabsStore } from '$lib/stores/tabs.svelte';
import { fFilterByLeafNodeId, fFindLeafNode, fGenerateConversationTitle } from '$lib/utils';
import { SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

class ConversationsStore implements ConversationsPreferencesHost {
  /** Currently active conversation */
  vActiveConversation = $state<DatabaseConversation | null>(null);

  /** Messages in the active conversation (filtered by currNode path) */
  aActiveMessages = $state<DatabaseMessage[]>([]);

  /** List of all conversations */
  aConversations = $state<DatabaseConversation[]>([]);

  /** Whether the store has been initialized */
  vIsInitialized = $state(false);

  /** Per-chat options (MCP overrides, reasoning effort, cwd), composed here. */
  private vPreferences = new ConversationPreferences(this);

  /**
   * Listeners notified with the ids of conversations that were deleted.
   * Lets dependent stores (e.g. agenticStore) drop per-conversation state
   * without introducing a circular import back into this store.
   */
  private sConversationDeletionListeners = new Set<(pConvIds: string[]) => void>();

  /** In-flight init run; shared by concurrent callers, reset on failure to allow retry */
  private vInitPromise: Promise<void> | null = null;

  /**
   * Messages loadConversation just read, handed off once so the chat
   * screen can reuse them for sibling info instead of re-fetching the
   * whole conversation a second time.
   */
  private vLastLoadedMessages: { convId: string; messages: DatabaseMessage[] } | null = null;

  /**
   * Memo of the last findMessageIndex() lookup. Streaming calls it once per
   * chunk for the same message, so a validated cache hit keeps that O(1)
   * instead of a linear scan of activeMessages on every token.
   */
  private vLastMessageIndex: { id: string; index: number } | null = null;

  get preferences() {
    return this.vPreferences;
  }

  /**
   * Adds a message to the active messages array
   */
  mAddMessageToActive(pMessage: DatabaseMessage): void {
    this.aActiveMessages.push(pMessage);
  }

  /**
   * Applies a field update to a conversation row, mirroring it into both the
   * conversations list and the active conversation when it is the target.
   * Shared by the rename/pin/preferences flows so no caller can forget to
   * mirror one side.
   */
  mApplyConversationUpdate(pId: string, pUpdates: Partial<DatabaseConversation>): void {
    const cConvIndex = this.aConversations.findIndex((pC) => pC.id === pId);

    if (cConvIndex !== -1) {
      const dTarget = this.aConversations[cConvIndex] as unknown as Record<string, unknown>;

      for (const [vKey, vValue] of Object.entries(pUpdates)) {
        if (dTarget[vKey] !== vValue) dTarget[vKey] = vValue;
      }
    }

    if (this.vActiveConversation?.id === pId) {
      // field-wise, not object replacement: effects that track the active
      // conversation identity would otherwise refire on every rename or pin
      const dTarget = this.vActiveConversation as unknown as Record<string, unknown>;

      for (const [vKey, vValue] of Object.entries(pUpdates)) {
        if (dTarget[vKey] !== vValue) dTarget[vKey] = vValue;
      }
    }
  }

  /**
   * Derives a conversation title from its first message content and applies
   * it, honoring the title-generation setting. Shared by every flow that
   * edits or creates the first user message.
   */
  async mApplyTitleFromContent(pConvId: string, pContent: string): Promise<void> {
    await this.mUpdateConversationName(
      pConvId,
      fGenerateConversationTitle(
        pContent,
        Boolean(cSettingsStore.vConfig.titleGenerationUseFirstLine)
      )
    );
  }

  /**
   * Deletes multiple conversations in sequence.
   * Mirrors deleteConversation() per-id; navigates to the new-chat screen only
   * if the currently-open chat was among the deleted ones.
   * @param convIds - Conversation IDs to delete
   */
  async mBulkDeleteConversations(pConvIds: string[]): Promise<void> {
    if (pConvIds.length === 0) return;

    try {
      const cIdsToRemove = new SvelteSet(pConvIds);
      // Collect all descendants recursively so the local cache stays consistent
      // even when deleteWithForks is omitted.
      const aQueue = [...pConvIds];

      while (aQueue.length > 0) {
        const cParentId = aQueue.pop()!;

        for (const c of this.aConversations) {
          if (c.forkedFromConversationId === cParentId && !cIdsToRemove.has(c.id)) {
            cIdsToRemove.add(c.id);
            aQueue.push(c.id);
          }
        }
      }

      const cActiveWasDeleted =
        this.vActiveConversation !== null && cIdsToRemove.has(this.vActiveConversation.id);

      await DatabaseService.mBulkDeleteConversations([...cIdsToRemove]);

      this.aConversations = this.aConversations.filter((pC) => !cIdsToRemove.has(pC.id));
      this.mNotifyConversationsDeleted([...cIdsToRemove]);

      if (cActiveWasDeleted) {
        const cActiveId = this.vActiveConversation!.id;

        cTabsStore.mRemoveTabs([...cIdsToRemove].filter((pId) => pId !== cActiveId));
        this.mClearActiveConversation();
        await cTabsStore.mClose(cActiveId, cActiveId);
      } else {
        cTabsStore.mRemoveTabs([...cIdsToRemove]);
      }

      toast.success(
        cIdsToRemove.size === 1
          ? 'Conversation deleted'
          : `${cIdsToRemove.size} conversations deleted`
      );
    } catch (error) {
      console.error('Failed to bulk delete conversations:', error);
      toast.error(fTranslate('text_4c0d52d097ba'));
    }
  }

  /**
   * Bundles the given conversations into a single zip archive and triggers a
   * browser download (one JSONL file per conversation).
   * @param convIds - Conversation IDs to export
   */
  async mBulkExportConversations(pConvIds: string[]): Promise<void> {
    if (pConvIds.length === 0) return;

    try {
      const aExported = await this.mGetConversationsForExport(pConvIds);

      if (aExported.length === 0) {
        toast.error(fTranslate('text_bf0779e06dd8'));

        return;
      }

      ConversationTransferService.mDownloadConversationsArchive(aExported);

      toast.success(
        aExported.length === 1
          ? 'Conversation exported'
          : `${aExported.length} conversations exported`
      );
    } catch (error) {
      console.error('Failed to bulk export conversations:', error);
      toast.error(fTranslate('text_96f775dbb3fc'));
    }
  }

  /**
   * Toggles the pinned state of each conversation individually.
   * Mixed-pin selections are intentionally not normalised here; the bulk
   * action UI surfaces them as a disabled mixed-state instead.
   * @param convIds - Conversation IDs to toggle
   */
  async mBulkToggleConversationPin(pConvIds: string[]): Promise<void> {
    if (pConvIds.length === 0) return;

    try {
      const dUpdates = await DatabaseService.mBulkToggleConversationPins(pConvIds);
      const cActiveId = this.vActiveConversation?.id;

      if (this.vActiveConversation && cActiveId && dUpdates.has(cActiveId)) {
        this.vActiveConversation.pinned = dUpdates.get(cActiveId)!;
      }

      for (let vI = 0; vI < this.aConversations.length; vI++) {
        const cNewPinned = dUpdates.get(this.aConversations[vI].id);

        if (cNewPinned !== undefined) this.aConversations[vI].pinned = cNewPinned;
      }

      toast.success(
        pConvIds.length === 1
          ? 'Conversation pin toggled'
          : `Updated pin state for ${pConvIds.length} conversations`
      );
    } catch (error) {
      console.error('Failed to bulk toggle pin:', error);
      toast.error(fTranslate('text_cbc9a6188b1f'));
    }
  }

  /**
   * Clears the active conversation and messages.
   */
  mClearActiveConversation(): void {
    this.vActiveConversation = null;
    this.aActiveMessages = [];
    // reload defaults so new chats inherit persisted state
    this.preferences.mResetPending();
  }

  /** One-shot handoff of the messages the last loadConversation read. */
  mConsumeLastLoadedMessages(pConvId: string): DatabaseMessage[] | null {
    if (this.vLastLoadedMessages?.convId !== pConvId) return null;

    const aMessages = this.vLastLoadedMessages.messages;

    this.vLastLoadedMessages = null;

    return aMessages;
  }

  /**
   * Creates a new conversation and navigates to it
   * @param name - Optional name for the conversation
   * @returns The ID of the created conversation
   */
  async mCreateConversation(pName?: string): Promise<string> {
    const cConversationName = pName || `Chat ${new Date().toLocaleString()}`;
    // The tool policy is seeded from the current defaults: edits made inside
    // the conversation afterwards live on its row and do not flow back into
    // the defaults. Working directory picked on the new-chat screen gets
    // threaded in here too, then cleared so it doesn't bleed onto subsequent
    // new chats.
    const cConversation = await DatabaseService.mCreateConversation(cConversationName, {
      cwd: this.preferences.vPendingCwd ?? undefined,
      reasoningEffort: this.preferences.vPendingReasoningEffort,
      ...this.preferences.mGetToolPolicySnapshot()
    });

    this.preferences.vPendingCwd = null;

    this.aConversations = [cConversation, ...this.aConversations];
    this.vActiveConversation = cConversation;
    this.aActiveMessages = [];

    await goto(RouterService.mChat(cConversation.id));

    return cConversation.id;
  }

  /**
   * Deletes all conversations and their messages
   */
  async mDeleteAll(): Promise<void> {
    try {
      const aAllConversations = await DatabaseService.mGetAllConversations();
      const aAllIds = aAllConversations.map((pC) => pC.id);

      await DatabaseService.mBulkDeleteConversations(aAllIds);

      this.mClearActiveConversation();
      this.aConversations = [];
      cTabsStore.mClear();
      this.mNotifyConversationsDeleted(aAllIds);

      toast.success(fTranslate('text_2f3c5f7a351a'));

      await goto(cRoutes.START);
    } catch (error) {
      console.error('Failed to delete all conversations:', error);
      toast.error(fTranslate('text_4c0d52d097ba'));
    }
  }

  /**
   * Deletes a conversation and all its messages
   * @param convId - The conversation ID to delete
   */
  async mDeleteConversation(
    pConvId: string,
    pOptions?: { deleteWithForks?: boolean }
  ): Promise<void> {
    try {
      await DatabaseService.mDeleteConversation(pConvId, pOptions);

      if (pOptions?.deleteWithForks) {
        // Collect all descendants recursively
        const cIdsToRemove = new SvelteSet([pConvId]);
        const aQueue = [pConvId];

        while (aQueue.length > 0) {
          const cParentId = aQueue.pop()!;

          for (const c of this.aConversations) {
            if (c.forkedFromConversationId === cParentId && !cIdsToRemove.has(c.id)) {
              cIdsToRemove.add(c.id);
              aQueue.push(c.id);
            }
          }
        }
        this.aConversations = this.aConversations.filter((pC) => !cIdsToRemove.has(pC.id));

        if (this.vActiveConversation && cIdsToRemove.has(this.vActiveConversation.id)) {
          const cActiveId = this.vActiveConversation.id;

          cTabsStore.mRemoveTabs([...cIdsToRemove].filter((pId) => pId !== cActiveId));
          this.mClearActiveConversation();
          await cTabsStore.mClose(cActiveId, cActiveId);
        } else {
          cTabsStore.mRemoveTabs([...cIdsToRemove]);
        }

        this.mNotifyConversationsDeleted([...cIdsToRemove]);
      } else {
        // Reparent direct children to deleted conv's parent (or promote to top-level)
        const cDeletedConv = this.aConversations.find((pC) => pC.id === pConvId);
        const cNewParent = cDeletedConv?.forkedFromConversationId;

        this.aConversations = this.aConversations
          .filter((pC) => pC.id !== pConvId)
          .map((pC) =>
            pC.forkedFromConversationId === pConvId
              ? { ...pC, forkedFromConversationId: cNewParent }
              : pC
          );

        if (this.vActiveConversation?.id === pConvId) {
          this.mClearActiveConversation();
          await cTabsStore.mClose(pConvId, pConvId);
        } else {
          cTabsStore.mRemoveTabs([pConvId]);
        }

        this.mNotifyConversationsDeleted([pConvId]);
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error);
    }
  }

  /**
   * Downloads a single conversation as a JSONL file, serializing the full message tree.
   * @param convId - The conversation ID to download
   */
  async mDownloadConversation(pConvId: string): Promise<void> {
    const [vExportedConversation] = await this.mGetConversationsForExport([pConvId]);

    if (!vExportedConversation) return;

    ConversationTransferService.mDownloadConversationFile(vExportedConversation);
  }

  /**
   * Finds the index of a message in active messages.
   *
   * The last lookup is memoized and reused when it still validates against
   * the current array (same id at the same position), which covers the
   * streaming hot path where the same message is looked up on every chunk
   * while the array itself only mutates by field. Any structural change
   * (splice, reassignment, reordering) fails validation and falls back to a
   * full scan.
   */
  mFindMessageIndex(pMessageId: string): number {
    const cLast = this.vLastMessageIndex;
    const aMessages = this.aActiveMessages;

    if (
      cLast &&
      cLast.id === pMessageId &&
      cLast.index >= 0 &&
      cLast.index < aMessages.length &&
      aMessages[cLast.index]?.id === pMessageId
    ) {
      return cLast.index;
    }

    const cIndex = aMessages.findIndex((pM) => pM.id === pMessageId);

    this.vLastMessageIndex = { id: pMessageId, index: cIndex };

    return cIndex;
  }

  /**
   * Forks a conversation at a specific message, creating a new conversation
   * containing messages from root up to the target message, then navigates to it.
   *
   * @param messageId - The message ID to fork at
   * @param options - Fork options (name and whether to include attachments)
   * @returns The new conversation ID, or null if fork failed
   */
  async mForkConversation(
    pMessageId: string,
    pOptions: { name: string; includeAttachments: boolean }
  ): Promise<string | null> {
    if (!this.vActiveConversation) return null;

    try {
      const cNewConv = await DatabaseService.mForkConversation(
        this.vActiveConversation.id,
        pMessageId,
        pOptions
      );

      this.aConversations = [cNewConv, ...this.aConversations];

      await goto(RouterService.mChat(cNewConv.id));

      toast.success(fTranslate('text_95de71e25038'));

      return cNewConv.id;
    } catch (error) {
      console.error('Failed to fork conversation:', error);
      toast.error(fTranslate('text_32ad3abe4479'));

      return null;
    }
  }

  /**
   * Gets all messages for a specific conversation
   * @param convId - The conversation ID
   * @returns Array of messages
   */
  async mGetConversationMessages(pConvId: string): Promise<DatabaseMessage[]> {
    return await DatabaseService.mGetConversationMessages(pConvId);
  }

  /**
   * Gets conversations and their messages from the database for export.
   * @param convIds - Conversation IDs
   * @returns List of conversations with messages, ordered by the input IDs
   */
  async mGetConversationsForExport(pConvIds: string[]): Promise<ExportedConversation[]> {
    const dFetched = await DatabaseService.mGetConversationsWithMessages(pConvIds);

    return pConvIds
      .map((pId) => dFetched.get(pId))
      .filter((pEntry): pEntry is ExportedConversation => pEntry !== undefined);
  }

  /**
   * Imports conversations from provided data (without file picker)
   * @param data - Array of conversation data with messages
   * @returns The conversations written to the database and the ones skipped
   */
  async mImportConversationsData(
    pData: ExportedConversations
  ): Promise<{ imported: DatabaseConversation[]; skipped: DatabaseConversation[] }> {
    const cResult = await DatabaseService.mImportConversations(pData);

    await this.mLoadConversations();

    return cResult;
  }

  /**
   * Initialize the store by loading conversations from database.
   * Safe to call multiple times: concurrent callers share a single run,
   * and a failed run can be retried by calling again.
   */
  mInitialize(): Promise<void> {
    if (!browser) return Promise.resolve();

    if (this.vInitPromise) return this.vInitPromise;

    this.vInitPromise = (async () => {
      try {
        await this.mLoadConversations();
        this.vIsInitialized = true;
      } catch (error) {
        console.error('Failed to initialize conversations:', error);
        this.vInitPromise = null;
      }
    })();

    return this.vInitPromise;
  }

  /**
   * Loads a specific conversation and its messages
   * @param convId - The conversation ID to load
   * @returns True if conversation was loaded successfully
   */
  async mLoadConversation(pConvId: string): Promise<boolean> {
    try {
      const cConversation = await DatabaseService.mGetConversation(pConvId);

      if (!cConversation) {
        return false;
      }

      // Drop any cwd the user drafted on the empty new-chat screen -
      // it doesn't belong to this conversation.
      this.preferences.vPendingCwd = null;

      const aAllMessages = await DatabaseService.mGetConversationMessages(pConvId);

      // set conversation and messages in one sync block so effects never see
      // the new conversation with the previous conversation's messages
      this.vLastLoadedMessages = { convId: pConvId, messages: aAllMessages };
      this.vActiveConversation = cConversation;
      this.aActiveMessages = cConversation.currNode
        ? (fFilterByLeafNodeId(aAllMessages, cConversation.currNode, false) as DatabaseMessage[])
        : aAllMessages;

      return true;
    } catch (error) {
      console.error('Failed to load conversation:', error);

      return false;
    }
  }

  /**
   * Loads all conversations from the database
   */
  async mLoadConversations(): Promise<void> {
    const aConversations = await DatabaseService.mGetAllConversations();

    this.aConversations = aConversations;
  }

  /**
   * Navigates to a specific sibling branch by updating currNode and refreshing messages.
   * @param siblingId - The sibling message ID to navigate to
   */
  async mNavigateToSibling(pSiblingId: string): Promise<void> {
    if (!this.vActiveConversation) return;

    const aAllMessages = await DatabaseService.mGetConversationMessages(
      this.vActiveConversation.id
    );
    const cRootMessage = aAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
    const cCurrentFirstUserMessage = this.aActiveMessages.find(
      (pM) => pM.role === MessageRole.cUser && pM.parent === cRootMessage?.id
    );
    const cCurrentLeafNodeId = fFindLeafNode(aAllMessages, pSiblingId);

    await DatabaseService.mUpdateCurrentNode(this.vActiveConversation.id, cCurrentLeafNodeId);
    this.vActiveConversation.currNode = cCurrentLeafNodeId;
    await this.mRefreshActiveMessages();

    if (cRootMessage && this.aActiveMessages.length > 0) {
      const cNewFirstUserMessage = this.aActiveMessages.find(
        (pM) => pM.role === MessageRole.cUser && pM.parent === cRootMessage.id
      );

      if (
        cNewFirstUserMessage &&
        cNewFirstUserMessage.content.trim() &&
        (!cCurrentFirstUserMessage ||
          cNewFirstUserMessage.id !== cCurrentFirstUserMessage.id ||
          cNewFirstUserMessage.content.trim() !== cCurrentFirstUserMessage.content.trim())
      ) {
        await this.mApplyTitleFromContent(
          this.vActiveConversation.id,
          cNewFirstUserMessage.content
        );
      }
    }
  }

  /**
   * Registers a listener invoked with the ids of deleted conversations.
   * Returns an unsubscribe function.
   */
  mOnConversationsDeleted(pListener: (pConvIds: string[]) => void): () => void {
    this.sConversationDeletionListeners.add(pListener);

    return () => this.sConversationDeletionListeners.delete(pListener);
  }

  /**
   * Start a fresh chat by navigating to the bare `#/` new-chat screen. The
   * chat layout opens a new-chat tab for it when Conversation tabs are on.
   */
  async mOpenNewChat(): Promise<void> {
    this.mClearActiveConversation();
    await goto(cRoutes.START);
  }

  /**
   * Refreshes active messages based on currNode after branch navigation.
   */
  async mRefreshActiveMessages(): Promise<void> {
    if (!this.vActiveConversation) return;

    const aAllMessages = await DatabaseService.mGetConversationMessages(
      this.vActiveConversation.id
    );

    if (aAllMessages.length === 0) {
      this.aActiveMessages = [];

      return;
    }

    const cLeafNodeId =
      this.vActiveConversation.currNode ||
      aAllMessages.reduce((pLatest, pMsg) => (pMsg.timestamp > pLatest.timestamp ? pMsg : pLatest))
        .id;
    const aCurrentPath = fFilterByLeafNodeId(aAllMessages, cLeafNodeId, false) as DatabaseMessage[];

    this.aActiveMessages = aCurrentPath;
  }

  /**
   * Removes a message from active messages by index
   */
  mRemoveMessageAtIndex(pIndex: number): DatabaseMessage | undefined {
    if (pIndex !== -1) {
      return this.aActiveMessages.splice(pIndex, 1)[0];
    }

    return undefined;
  }

  /**
   * Removes messages from active messages starting at an index
   */
  mSliceActiveMessages(pStartIndex: number): void {
    this.aActiveMessages = this.aActiveMessages.slice(0, pStartIndex);
  }

  /**
   * Toggles the pinned status of a conversation.
   * @param convId - The conversation ID to toggle
   * @returns The new pinned status
   */
  async mToggleConversationPin(pConvId: string): Promise<boolean> {
    try {
      const cNewPinnedState = await DatabaseService.mToggleConversationPin(pConvId);

      this.mApplyConversationUpdate(pConvId, { pinned: cNewPinnedState });

      return cNewPinnedState;
    } catch (error) {
      console.error('Failed to toggle conversation pin:', error);

      return false;
    }
  }

  /**
   * Updates the name of a conversation.
   * @param convId - The conversation ID to update
   * @param name - The new name for the conversation
   */
  async mUpdateConversationName(pConvId: string, pName: string): Promise<void> {
    try {
      await DatabaseService.mUpdateConversation(pConvId, { name: pName });

      this.mApplyConversationUpdate(pConvId, { name: pName });
    } catch (error) {
      console.error('Failed to update conversation name:', error);
    }
  }

  /**
   * Marks a conversation as recently active: stamps lastModified (persisted)
   * and moves it to the top of the list. Only message-activity flows call
   * this; metadata updates (rename, pin, settings) do not.
   *
   * @param convId - Conversation that produced the activity, defaults to the active one
   */
  mUpdateConversationTimestamp(pConvId?: string): void {
    const cTargetId = pConvId ?? this.vActiveConversation?.id;

    if (!cTargetId) return;

    const cNow = Date.now();
    const cChatIndex = this.aConversations.findIndex((pC) => pC.id === cTargetId);

    if (cChatIndex !== -1) {
      this.aConversations[cChatIndex].lastModified = cNow;
      const cUpdatedConv = this.aConversations.splice(cChatIndex, 1)[0];

      this.aConversations = [cUpdatedConv, ...this.aConversations];
    }

    if (this.vActiveConversation?.id === cTargetId) {
      this.vActiveConversation.lastModified = cNow;
    }

    DatabaseService.mUpdateConversation(cTargetId, { lastModified: cNow }).catch((pError) =>
      console.error('Failed to update conversation timestamp:', pError)
    );
  }

  /**
   * Updates the current node of the active conversation
   * @param nodeId - The new current node ID
   */
  async mUpdateCurrentNode(pNodeId: string): Promise<void> {
    if (!this.vActiveConversation) return;

    await DatabaseService.mUpdateCurrentNode(this.vActiveConversation.id, pNodeId);
    this.vActiveConversation.currNode = pNodeId;
  }

  /**
   * Updates a message at a specific index in active messages
   */
  mUpdateMessageAtIndex(pIndex: number, pUpdates: Partial<DatabaseMessage>): void {
    const message = pIndex === -1 ? undefined : this.aActiveMessages[pIndex];

    if (!message) return;

    // Assign field by field rather than replacing the object. Replacing it
    // changes the array slot, which invalidates every consumer that merely
    // walks the list - notably ChatMessages.displayMessages, which rebuilds
    // entries for every message in the conversation. Deep $state proxies make
    // per-field writes fine-grained, so only readers of the changed field wake.
    const dTarget = message as unknown as Record<string, unknown>;

    for (const [vKey, vValue] of Object.entries(pUpdates)) {
      if (dTarget[vKey] !== vValue) {
        dTarget[vKey] = vValue;
      }
    }
  }

  /**
   *
   *
   * Import & Export
   *
   *
   */

  private mNotifyConversationsDeleted(pConvIds: string[]): void {
    if (pConvIds.length === 0) return;

    for (const fListener of this.sConversationDeletionListeners) {
      fListener(pConvIds);
    }
  }
}

export const cConversationsStore = new ConversationsStore();
