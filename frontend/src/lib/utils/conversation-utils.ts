/**
 * Utility functions for conversation data manipulation
 */
import { MessageRole } from '$lib/enums';
import type { DatabaseMessage } from '$lib/types';

/**
 * Model that generated the conversation's latest assistant message, or null
 * when no assistant message carries one.
 */
export function fGetConversationModel(pMessages: readonly DatabaseMessage[]): string | null {
  for (let vI = pMessages.length - 1; vI >= 0; vI--) {
    const message = pMessages[vI];

    if (message.role === MessageRole.cAssistant && message.model) return message.model;
  }

  return null;
}

/**
 * Creates a map of conversation IDs to their message counts from exported conversation data
 * @param exportedData - Array of exported conversations with their messages
 * @returns Map of conversation ID to message count
 */
export function fCreateMessageCountMap(
  pExportedData: Array<{ conv: DatabaseConversation; messages: DatabaseMessage[] }>
): Map<string, number> {
  const dCountMap = new Map<string, number>();

  for (const cItem of pExportedData) {
    dCountMap.set(cItem.conv.id, cItem.messages.length);
  }

  return dCountMap;
}

/**
 * Gets the message count for a specific conversation from the count map
 * @param conversationId - The ID of the conversation
 * @param countMap - Map of conversation IDs to message counts
 * @returns The message count, or 0 if not found
 */
export function fGetMessageCount(pConversationId: string, pCountMap: Map<string, number>): number {
  return pCountMap.get(pConversationId) ?? 0;
}

export interface ConversationTreeItem {
  conversation: DatabaseConversation;
  depth: number;
}

// Pinned conversations first, then by lastModified descending
const fComparePinnedThenRecent = (pA: DatabaseConversation, pB: DatabaseConversation) => {
  if (pA.pinned && !pB.pinned) return -1;

  if (!pA.pinned && pB.pinned) return 1;

  return pB.lastModified - pA.lastModified;
};

/**
 * Builds a flat tree of conversations with depth levels for nested forks.
 * Accepts a pre-filtered list so search filtering stays in the component.
 *
 * Output order matches the sidebar render exactly: pinned first, then
 * unpinned by lastModified desc, with forks interleaved under their parents.
 * Range-select / marquee in the sidebar rely on this alignment.
 */
export function fBuildConversationTree(pConvs: DatabaseConversation[]): ConversationTreeItem[] {
  const dChildrenByParent = new Map<string, DatabaseConversation[]>();
  const sForkIds = new Set<string>();

  for (const cConv of pConvs) {
    if (cConv.forkedFromConversationId) {
      sForkIds.add(cConv.id);

      const aSiblings = dChildrenByParent.get(cConv.forkedFromConversationId) || [];

      aSiblings.push(cConv);
      dChildrenByParent.set(cConv.forkedFromConversationId, aSiblings);
    }
  }

  const aResult: ConversationTreeItem[] = [];
  const sVisited = new Set<string>();

  function fWalk(pConv: DatabaseConversation, pDepth: number) {
    sVisited.add(pConv.id);
    aResult.push({ conversation: pConv, depth: pDepth });

    const cChildren = dChildrenByParent.get(pConv.id);

    if (cChildren) {
      cChildren.sort(fComparePinnedThenRecent);

      for (const cChild of cChildren) {
        fWalk(cChild, pDepth + 1);
      }
    }
  }

  const aRoots = pConvs.filter((pC) => !sForkIds.has(pC.id)).sort(fComparePinnedThenRecent);

  for (const root of aRoots) {
    fWalk(root, 0);
  }

  for (const cConv of pConvs) {
    if (!sVisited.has(cConv.id)) {
      fWalk(cConv, 1);
    }
  }

  return aResult;
}
