import { SvelteMap } from 'svelte/reactivity';

export interface QueueStatus {
  ahead: number;
  ahead_api: number;
  ahead_webui: number;
}

class QueueStore {
  private dStatus = new SvelteMap<string, QueueStatus>();

  mGet(pConversationId?: string): QueueStatus | undefined {
    return pConversationId ? this.dStatus.get(pConversationId) : undefined;
  }

  mSet(pConversationId: string | undefined, pStatus?: QueueStatus): void {
    if (!pConversationId) return;
    if (pStatus && pStatus.ahead > 0) this.dStatus.set(pConversationId, pStatus);
    else this.dStatus.delete(pConversationId);
  }
}

export const cQueueStore = new QueueStore();
