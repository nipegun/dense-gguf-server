import { fTranslate } from '$lib/i18n';
/**
 * DatabaseService - IndexedDB persistence for conversations and messages
 *
 * Thin Dexie layer over the conversations/messages tables: CRUD, tree
 * navigation (descendants, reparenting) and cascading deletes. No reactive
 * state; consumed by conversationsStore and the chat flows.
 */

import { cIdxdbStores, cIdxdbTables, cStorageAppName } from '$lib/constants';
import { MessageRole } from '$lib/enums';
import type { ExportedConversation } from '$lib/types/database';
import { fFilterByLeafNodeId, fFindDescendantMessages, fUuid } from '$lib/utils';
import Dexie, { type EntityTable } from 'dexie';

class DgsUiDatabase extends Dexie {
  [cIdxdbTables.conversations]!: EntityTable<DatabaseConversation, string>;
  [cIdxdbTables.messages]!: EntityTable<DatabaseMessage, string>;

  constructor() {
    super(cStorageAppName);

    this.version(1).stores(cIdxdbStores);
  }
}

const cDb = new DgsUiDatabase();

export class DatabaseService {
  /**
   * Deletes multiple conversations in a single transaction. Each deleted
   * conversation has its direct children reparented to the nearest surviving
   * ancestor (or promoted to top-level). Children also in `ids` are dropped
   * entirely rather than reparented.
   *
   * @param ids - Conversation IDs to delete
   */
  static async mBulkDeleteConversations(pIds: string[]): Promise<void> {
    const aCleanIds = pIds.filter(
      (pId): pId is string => typeof pId === 'string' && pId.length > 0
    );

    if (aCleanIds.length === 0) return;

    const sIdSet = new Set(aCleanIds);

    await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        // Pre-load each to-delete conversation so the per-id reparent
        // walk-up doesn't ping-pong the same ancestry chain.
        const dPrefetched = new Map<string, DatabaseConversation>();

        let aFrontier = [...aCleanIds];

        const sRequested = new Set<string>(aFrontier);

        while (aFrontier.length > 0) {
          const aFetched = await cDb[cIdxdbTables.conversations].bulkGet(aFrontier);

          aFrontier = [];
          for (let vI = 0; vI < aFetched.length; vI++) {
            const cConv = aFetched[vI];

            if (!cConv || !cConv.id) continue;

            dPrefetched.set(cConv.id, cConv);
            const cAncestor = cConv.forkedFromConversationId;

            if (cAncestor && !dPrefetched.has(cAncestor) && !sRequested.has(cAncestor)) {
              aFrontier.push(cAncestor);
              sRequested.add(cAncestor);
            }
          }
        }

        for (const cId of aCleanIds) {
          await this.mReparentDirectChildren(cId, sIdSet, dPrefetched);
        }

        await cDb[cIdxdbTables.conversations].bulkDelete(aCleanIds);
        await cDb[cIdxdbTables.messages].where('convId').anyOf(aCleanIds).delete();
      }
    );
  }

  /**
   * Toggles the pinned status of each conversation in `ids` inside a single
   * transaction. Treats `pinned === undefined` as `false`, matching the
   * semantics of {@link toggleConversationPin} where `!undefined` evaluates
   * to `true`. Returns the resulting pinned state for every id that was
   * updated; missing ids are omitted from the map.
   *
   * @param ids - Conversation IDs to toggle
   * @returns Map of id -> new pinned state
   */
  static async mBulkToggleConversationPins(pIds: string[]): Promise<Map<string, boolean>> {
    const aCleanIds = pIds.filter(
      (pId): pId is string => typeof pId === 'string' && pId.length > 0
    );
    const dResult = new Map<string, boolean>();

    if (aCleanIds.length === 0) return dResult;

    await cDb.transaction('rw', cDb[cIdxdbTables.conversations], async () => {
      const aConvs = await cDb[cIdxdbTables.conversations].bulkGet(aCleanIds);
      const aUpdates: DatabaseConversation[] = [];

      for (let vI = 0; vI < aCleanIds.length; vI++) {
        const cConv = aConvs[vI];

        if (!cConv) continue;

        const cNewPinned = !cConv.pinned;

        aUpdates.push({ ...cConv, pinned: cNewPinned });
        dResult.set(aCleanIds[vI], cNewPinned);
      }

      if (aUpdates.length === 0) return;

      await cDb[cIdxdbTables.conversations].bulkPut(aUpdates);
    });

    return dResult;
  }

  /**
   * Creates a new conversation.
   *
   * @param name - Name of the conversation
   * @param fields - Optional extra fields (e.g. reasoningEffort)
   * @returns The created conversation
   */
  static async mCreateConversation(
    pName: string,
    pFields?: Partial<Omit<DatabaseConversation, 'id' | 'name' | 'lastModified'>>
  ): Promise<DatabaseConversation> {
    const cConversation: DatabaseConversation = {
      currNode: '',
      id: fUuid(),
      lastModified: Date.now(),
      name: pName,
      ...pFields
    };

    await cDb[cIdxdbTables.conversations].add(cConversation);

    return cConversation;
  }

  /**
   * Creates a new message branch by adding a message and updating parent/child relationships.
   * Also updates the conversation's currNode to point to the new message.
   *
   * @param message - Message to add (without id)
   * @param parentId - Parent message ID to attach to
   * @returns The created message
   */
  static async mCreateMessageBranch(
    pMessage: Omit<DatabaseMessage, 'id'>,
    pParentId: string | null
  ): Promise<DatabaseMessage> {
    return await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        // Handle null parent (root message case)
        if (pParentId !== null) {
          const cParentMessage = await cDb[cIdxdbTables.messages].get(pParentId);

          if (!cParentMessage) {
            throw new Error(`Parent message ${pParentId} not found`);
          }
        }

        const cNewMessage: DatabaseMessage = {
          ...pMessage,
          children: [],
          id: fUuid(),
          parent: pParentId,
          toolCalls: pMessage.toolCalls ?? ''
        };

        await cDb[cIdxdbTables.messages].add(cNewMessage);

        // Update parent's children array if parent exists
        if (pParentId !== null) {
          await this.mAddChildToParent(pParentId, cNewMessage.id);
        }

        await this.mUpdateConversation(pMessage.convId, {
          currNode: cNewMessage.id
        });

        return cNewMessage;
      }
    );
  }

  /**
   * Creates a root message for a new conversation.
   * Root messages are not displayed but serve as the tree root for branching.
   *
   * @param convId - Conversation ID
   * @returns The created root message
   */
  static async mCreateRootMessage(pConvId: string): Promise<string> {
    const cRootMessage: DatabaseMessage = {
      children: [],
      content: '',
      convId: pConvId,
      id: fUuid(),
      parent: null,
      role: MessageRole.cSystem,
      timestamp: Date.now(),
      toolCalls: '',
      type: 'root'
    };

    await cDb[cIdxdbTables.messages].add(cRootMessage);

    return cRootMessage.id;
  }

  /**
   * Creates a system prompt message for a conversation.
   *
   * @param convId - Conversation ID
   * @param systemPrompt - The system prompt content (must be non-empty)
   * @param parentId - Parent message ID (typically the root message)
   * @returns The created system message
   * @throws Error if systemPrompt is empty or the parent message does not exist
   */
  static async mCreateSystemMessage(
    pConvId: string,
    pSystemPrompt: string,
    pParentId: string
  ): Promise<DatabaseMessage> {
    const cTrimmedPrompt = pSystemPrompt.trim();

    if (!cTrimmedPrompt) {
      throw new Error(fTranslate('text_3e4406904217'));
    }

    return await cDb.transaction('rw', cDb[cIdxdbTables.messages], async () => {
      const cParentMessage = await cDb[cIdxdbTables.messages].get(pParentId);

      if (!cParentMessage) {
        throw new Error(`Parent message ${pParentId} not found`);
      }

      const systemMessage: DatabaseMessage = {
        children: [],
        content: cTrimmedPrompt,
        convId: pConvId,
        id: fUuid(),
        parent: pParentId,
        role: MessageRole.cSystem,
        timestamp: Date.now(),
        type: MessageRole.cSystem
      };

      await cDb[cIdxdbTables.messages].add(systemMessage);
      await this.mAddChildToParent(pParentId, systemMessage.id);

      return systemMessage;
    });
  }

  /**
   * Deletes a conversation and all its messages.
   *
   * @param id - Conversation ID
   */
  static async mDeleteConversation(
    pId: string,
    pOptions?: { deleteWithForks?: boolean }
  ): Promise<void> {
    await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        if (pOptions?.deleteWithForks) {
          // Recursively collect all descendant IDs
          const aIdsToDelete: string[] = [];
          const aQueue = [pId];

          while (aQueue.length > 0) {
            const cParentId = aQueue.pop()!;
            const aChildren = await cDb[cIdxdbTables.conversations]
              .filter((pC) => pC.forkedFromConversationId === cParentId)
              .toArray();

            for (const cChild of aChildren) {
              aIdsToDelete.push(cChild.id);
              aQueue.push(cChild.id);
            }
          }

          for (const cForkId of aIdsToDelete) {
            await cDb[cIdxdbTables.conversations].delete(cForkId);
            await cDb[cIdxdbTables.messages].where('convId').equals(cForkId).delete();
          }
        } else {
          await this.mReparentDirectChildren(pId);
        }

        await cDb[cIdxdbTables.conversations].delete(pId);
        await cDb[cIdxdbTables.messages].where('convId').equals(pId).delete();
      }
    );
  }

  /**
   * Deletes a message and removes it from its parent's children array.
   *
   * @param messageId - ID of the message to delete
   */
  static async mDeleteMessage(pMessageId: string): Promise<void> {
    await cDb.transaction('rw', cDb[cIdxdbTables.messages], async () => {
      const message = await cDb[cIdxdbTables.messages].get(pMessageId);

      if (!message) return;

      await this.mRemoveChildFromParent(pMessageId);

      await cDb[cIdxdbTables.messages].delete(pMessageId);
    });
  }

  /**
   * Deletes a message and all its descendant messages (cascading deletion).
   * This removes the entire branch starting from the specified message.
   *
   * @param conversationId - ID of the conversation containing the message
   * @param messageId - ID of the root message to delete (along with all descendants)
   * @returns Array of all deleted message IDs
   */
  static async mDeleteMessageCascading(
    pConversationId: string,
    pMessageId: string
  ): Promise<string[]> {
    return await cDb.transaction('rw', cDb[cIdxdbTables.messages], async () => {
      // Get all messages in the conversation to find descendants
      const aAllMessages = await cDb[cIdxdbTables.messages]
        .where('convId')
        .equals(pConversationId)
        .toArray();
      const aDescendants = fFindDescendantMessages(aAllMessages, pMessageId);
      const aAllToDelete = [pMessageId, ...aDescendants];

      await this.mRemoveChildFromParent(pMessageId);

      // Delete all messages in the branch
      await cDb[cIdxdbTables.messages].bulkDelete(aAllToDelete);

      return aAllToDelete;
    });
  }

  /**
   * Forks a conversation at a specific message, creating a new conversation
   * containing all messages from the root up to (and including) the target message.
   *
   * @param sourceConvId - The source conversation ID
   * @param atMessageId - The message ID to fork at (the new conversation ends here)
   * @param options - Fork options (name and whether to include attachments)
   * @returns The newly created conversation
   */
  static async mForkConversation(
    pSourceConvId: string,
    pAtMessageId: string,
    pOptions: { name: string; includeAttachments: boolean }
  ): Promise<DatabaseConversation> {
    return await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        const cSourceConv = await cDb[cIdxdbTables.conversations].get(pSourceConvId);

        if (!cSourceConv) {
          throw new Error(`Source conversation ${pSourceConvId} not found`);
        }

        const aAllMessages = await cDb[cIdxdbTables.messages]
          .where('convId')
          .equals(pSourceConvId)
          .toArray();
        const aPathMessages = fFilterByLeafNodeId(
          aAllMessages,
          pAtMessageId,
          true
        ) as DatabaseMessage[];

        if (aPathMessages.length === 0) {
          throw new Error(`Could not resolve message path to ${pAtMessageId}`);
        }

        const dIdMap = new Map<string, string>();

        for (const cMsg of aPathMessages) {
          dIdMap.set(cMsg.id, fUuid());
        }

        const cNewConvId = fUuid();
        const aClonedMessages: DatabaseMessage[] = aPathMessages.map((pMsg) => {
          const cNewId = dIdMap.get(pMsg.id)!;
          const cNewParent = pMsg.parent ? (dIdMap.get(pMsg.parent) ?? null) : null;
          const cNewChildren = pMsg.children
            .filter((pChildId: string) => dIdMap.has(pChildId))
            .map((pChildId: string) => dIdMap.get(pChildId)!);

          return {
            ...pMsg,
            children: cNewChildren,
            convId: cNewConvId,
            extra: pOptions.includeAttachments ? pMsg.extra : undefined,
            id: cNewId,
            parent: cNewParent
          };
        });
        const cLastClonedMessage = aClonedMessages[aClonedMessages.length - 1];
        const cNewConv: DatabaseConversation = {
          currNode: cLastClonedMessage.id,
          cwd: cSourceConv.cwd,
          forkedFromConversationId: pSourceConvId,
          id: cNewConvId,
          lastModified: Date.now(),
          name: pOptions.name
        };

        await cDb[cIdxdbTables.conversations].add(cNewConv);
        await cDb[cIdxdbTables.messages].bulkAdd(aClonedMessages);

        return cNewConv;
      }
    );
  }

  /**
   * Gets all conversations, sorted by last modified time (newest first).
   *
   * @returns Array of conversations
   */
  static async mGetAllConversations(): Promise<DatabaseConversation[]> {
    return await cDb[cIdxdbTables.conversations].orderBy('lastModified').reverse().toArray();
  }

  /**
   * Gets a conversation by ID.
   *
   * @param id - Conversation ID
   * @returns The conversation if found, otherwise undefined
   */
  static async mGetConversation(pId: string): Promise<DatabaseConversation | undefined> {
    return await cDb[cIdxdbTables.conversations].get(pId);
  }

  /**
   * Gets all messages in a conversation, sorted by timestamp (oldest first).
   *
   * @param convId - Conversation ID
   * @returns Array of messages in the conversation
   */
  static async mGetConversationMessages(pConvId: string): Promise<DatabaseMessage[]> {
    return await cDb[cIdxdbTables.messages].where('convId').equals(pConvId).sortBy('timestamp');
  }

  /**
   * Loads multiple conversations with all of their messages in two bulk
   * reads. Missing conversations are silently omitted from the result.
   *
   * @param convIds - Conversation IDs to load
   * @returns Map of id -> { conv, messages }. Messages are sorted ascending by timestamp.
   */
  static async mGetConversationsWithMessages(
    pConvIds: string[]
  ): Promise<Map<string, ExportedConversation>> {
    const dResult = new Map<string, ExportedConversation>();
    const aCleanIds = pConvIds.filter(
      (pId): pId is string => typeof pId === 'string' && pId.length > 0
    );

    if (aCleanIds.length === 0) return dResult;

    const [aConvs, aAllMessages] = await Promise.all([
      cDb[cIdxdbTables.conversations].bulkGet(aCleanIds),
      cDb[cIdxdbTables.messages].where('convId').anyOf(aCleanIds).toArray()
    ]);
    const dMessagesByConv = new Map<string, DatabaseMessage[]>();

    for (const cMsg of aAllMessages) {
      const cBucket = dMessagesByConv.get(cMsg.convId);

      if (cBucket) cBucket.push(cMsg);
      else dMessagesByConv.set(cMsg.convId, [cMsg]);
    }

    for (let vI = 0; vI < aCleanIds.length; vI++) {
      const cConv = aConvs[vI];

      if (!cConv) continue;

      const aMessages = (dMessagesByConv.get(cConv.id) ?? []).sort(
        (pA, pB) => pA.timestamp - pB.timestamp
      );

      dResult.set(cConv.id, { conv: cConv, messages: aMessages });
    }

    return dResult;
  }

  /**
   * Imports multiple conversations and their messages.
   * Skips conversations that already exist.
   *
   * @param data - Array of { conv, messages } objects
   * @returns The conversations written to the database and the ones skipped
   */
  static async mImportConversations(
    pData: { conv: DatabaseConversation; messages: DatabaseMessage[] }[]
  ): Promise<{ imported: DatabaseConversation[]; skipped: DatabaseConversation[] }> {
    const aImported: DatabaseConversation[] = [];
    const aSkipped: DatabaseConversation[] = [];

    return await cDb.transaction(
      'rw',
      [cDb[cIdxdbTables.conversations], cDb[cIdxdbTables.messages]],
      async () => {
        for (const cItem of pData) {
          const { conv: vConv, messages: aMessages } = cItem;
          const cExisting = await cDb[cIdxdbTables.conversations].get(vConv.id);

          if (cExisting) {
            aSkipped.push(vConv);

            continue;
          }

          await cDb[cIdxdbTables.conversations].add(vConv);
          for (const cMsg of aMessages) {
            await cDb[cIdxdbTables.messages].put(cMsg);
          }

          aImported.push(vConv);
        }

        return { imported: aImported, skipped: aSkipped };
      }
    );
  }

  /**
   * Toggles the pinned status of a conversation.
   *
   * @param id - Conversation ID
   * @returns The new pinned status
   */
  static async mToggleConversationPin(pId: string): Promise<boolean> {
    const cConversation = await cDb[cIdxdbTables.conversations].get(pId);

    if (!cConversation) {
      throw new Error(`Conversation ${pId} not found`);
    }

    const cNewPinnedState = !cConversation.pinned;

    await this.mUpdateConversation(pId, { pinned: cNewPinnedState });

    return cNewPinnedState;
  }

  /**
   * Updates a conversation. `lastModified` is never stamped implicitly;
   * pass it in `updates` to bump the conversation in recency ordering.
   *
   * @param id - Conversation ID
   * @param updates - Partial updates to apply
   * @returns Promise that resolves when the conversation is updated
   */
  static async mUpdateConversation(
    pId: string,
    pUpdates: Partial<Omit<DatabaseConversation, 'id'>>
  ): Promise<void> {
    await cDb[cIdxdbTables.conversations].update(pId, pUpdates);
  }

  /**
   * Updates the conversation's current node (active branch).
   * This determines which conversation path is currently being viewed.
   *
   * @param convId - Conversation ID
   * @param nodeId - Message ID to set as current node
   */
  static async mUpdateCurrentNode(pConvId: string, pNodeId: string): Promise<void> {
    await this.mUpdateConversation(pConvId, {
      currNode: pNodeId
    });
  }

  /**
   * Updates a message.
   *
   * @param id - Message ID
   * @param updates - Partial updates to apply
   * @returns Promise that resolves when the message is updated
   */
  static async mUpdateMessage(
    pId: string,
    pUpdates: Partial<Omit<DatabaseMessage, 'id'>>
  ): Promise<void> {
    await cDb[cIdxdbTables.messages].update(pId, pUpdates);
  }

  /**
   * Appends a child id to a parent message's children array.
   */
  private static async mAddChildToParent(pParentId: string, pChildId: string): Promise<void> {
    const cParent = await cDb[cIdxdbTables.messages].get(pParentId);

    if (!cParent) return;

    await cDb[cIdxdbTables.messages].update(pParentId, {
      children: [...cParent.children, pChildId]
    });
  }

  /**
   * Removes a child id from its parent message's children array.
   */
  private static async mRemoveChildFromParent(pMessageId: string): Promise<void> {
    const message = await cDb[cIdxdbTables.messages].get(pMessageId);

    if (!message?.parent) return;

    const cParent = await cDb[cIdxdbTables.messages].get(message.parent);

    if (!cParent) return;

    cParent.children = cParent.children.filter((pChildId: string) => pChildId !== pMessageId);
    await cDb[cIdxdbTables.messages].put(cParent);
  }

  /**
   * Reparents direct children of `parentId` to the nearest surviving
   * ancestor (or promotes them to top-level when the immediate parent was
   * top-level). Walking skips any ancestor listed in `excludeIds`, since
   * those will be deleted in the same batch — leaving a grandchild pointing
   * at an `excludeIds` entry would orphan it. Children whose own id is in
   * `excludeIds` are dropped from the updates (the bulk-delete pass will
   * remove them). `prefetched` may carry a pre-fetched ancestor map to
   * avoid repeat reads inside a bulk transaction.
   */
  private static async mReparentDirectChildren(
    pParentId: string,
    pExcludeIds: ReadonlySet<string> = new Set(),
    pPrefetched?: ReadonlyMap<string, DatabaseConversation>
  ): Promise<void> {
    const cConv =
      pPrefetched?.get(pParentId) ?? (await cDb[cIdxdbTables.conversations].get(pParentId));

    if (!cConv) return;

    let vNewParent = cConv.forkedFromConversationId;

    const sVisited = new Set<string>([pParentId]);

    while (vNewParent && pExcludeIds.has(vNewParent)) {
      if (sVisited.has(vNewParent)) {
        vNewParent = undefined;

        break;
      }

      sVisited.add(vNewParent);
      const cNext =
        pPrefetched?.get(vNewParent) ?? (await cDb[cIdxdbTables.conversations].get(vNewParent));

      if (!cNext) {
        vNewParent = undefined;

        break;
      }

      vNewParent = cNext.forkedFromConversationId;
    }

    const aDirectChildren = await cDb[cIdxdbTables.conversations]
      .filter((pC) => pC.forkedFromConversationId === pParentId)
      .toArray();
    const aUpdates: DatabaseConversation[] = [];

    for (const cChild of aDirectChildren) {
      if (pExcludeIds.has(cChild.id)) continue;

      aUpdates.push({ ...cChild, forkedFromConversationId: vNewParent });
    }

    if (aUpdates.length === 0) return;

    await cDb[cIdxdbTables.conversations].bulkPut(aUpdates);
  }
}
