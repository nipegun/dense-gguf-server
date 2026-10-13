/**
 * DraftMessagesStore - Per-conversation input drafts
 *
 * Keeps in-memory drafts (message text + files) keyed by conversation id,
 * plus a dedicated key for the new-chat screen, so the input box restores
 * its content when switching conversations.
 */

import { cNewChatDraftKey } from '$lib/constants';

interface DraftMessage {
  message: string;
  files: ChatUploadedFile[];
}

class DraftMessagesStore {
  private dDrafts = new Map<string, DraftMessage>();

  mClearDraftMessage(pChatId: string | undefined): void {
    const cKey = pChatId ?? cNewChatDraftKey;

    this.dDrafts.delete(cKey);
  }

  mGetDraftMessage(pChatId: string | undefined): DraftMessage {
    const cKey = pChatId ?? cNewChatDraftKey;

    return this.dDrafts.get(cKey) ?? { files: [], message: '' };
  }

  mSaveDraftMessage(
    pChatId: string | undefined,
    pMessage: string,
    pFiles: ChatUploadedFile[]
  ): void {
    const cKey = pChatId ?? cNewChatDraftKey;

    if (pMessage || pFiles.length > 0) {
      this.dDrafts.set(cKey, { files: [...pFiles], message: pMessage });
    } else {
      this.dDrafts.delete(cKey);
    }
  }
}

export const cDraftMessagesStore = new DraftMessagesStore();
