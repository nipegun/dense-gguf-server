/**
 * ChatActivityStore - Conversation activity ledger
 *
 * Single owner of the "is this conversation doing something" state:
 * - `local`  - this browser is piping a stream (send, server-stream attach,
 *              or resume-wait while the owning model loads)
 * - `remote` - the backend reports a running session, no local pipe yet
 *              (global snapshot on mount / visibilitychange)
 *
 * The union of both drives the sidebar spinners (`loadingConvs`); `local`
 * drives the per-conversation loading flags. When a local pipe ends it is
 * the authoritative observer of session end, so it also drops the stale
 * remote hint in the same call - no cross-owner cleanup, no ghosted
 * spinners waiting for the next visibilitychange snapshot.
 *
 * Composed under chatStore.activity; not exported from the stores barrel.
 */

import { SvelteSet } from 'svelte/reactivity';

export class ChatActivityStore {
  /** Convs this browser is piping a stream for (send, attach, resume-wait). */
  private vLocal = new SvelteSet<string>();
  /** Convs the backend reports as having a running session (snapshot sync). */
  private vRemote = new SvelteSet<string>();

  /** Convs with any activity, the union the sidebar spinners render. */
  aLoadingConvs = $derived.by(() => {
    const cOut = new SvelteSet<string>(this.vLocal);

    for (const cId of this.vRemote) cOut.add(cId);

    return Array.from(cOut);
  });

  /**
   * Apply a backend snapshot of running sessions (mount / visibilitychange).
   * Diffed so unchanged entries do not re-trigger reactivity.
   */
  mApplyRemoteSnapshot(pRunning: Iterable<string>): void {
    const cNext = new SvelteSet<string>(pRunning);

    for (const cId of Array.from(this.vRemote)) {
      if (!cNext.has(cId)) this.vRemote.delete(cId);
    }

    for (const cId of cNext) this.vRemote.add(cId);
  }

  mIsLocal(pConvId: string): boolean {
    return this.vLocal.has(pConvId);
  }

  mIsRemote(pConvId: string): boolean {
    return this.vRemote.has(pConvId);
  }

  /**
   * A local pipe ended for the conv. Also drops the remote hint: the local
   * pipe is the authoritative observer of session end, so the sidebar hint
   * goes away right away instead of ghosting until the next snapshot.
   */
  mLocalEnded(pConvId: string): void {
    this.vLocal.delete(pConvId);
    this.vRemote.delete(pConvId);
  }

  /** A local pipe (send, attach or resume-wait) started for the conv. */
  mMarkLocal(pConvId: string): void {
    this.vLocal.add(pConvId);
  }
}

export const cChatActivityStore = new ChatActivityStore();
