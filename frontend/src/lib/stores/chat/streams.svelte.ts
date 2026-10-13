/**
 * ChatStreamManager - Server-side stream sessions for conversations
 *
 * Owns the attach lifecycle for streams that live on the server: discovery,
 * replay from byte 0, and resume retry while the owning model loads. The
 * remote-running snapshot it produces feeds the chat activity ledger
 * (chatStore.activity), which owns the actual running-conv state. Created
 * and owned by chatStore; the host exposes the per-conversation state setters.
 */

import { cConversationIdSeparator, cStreamResumeRetryMs } from '$lib/constants';
import { MessageRole, MessageType, StreamConnectionState } from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import type { ChatActivityStore } from '$lib/stores/chat/activity.svelte';
import type { ChatProcessingStore } from '$lib/stores/chat/processing.svelte';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cConversationsStore } from '$lib/stores/conversations/index.svelte';
import { cModelsStore } from '$lib/stores/models/index.svelte';
import type { ApiStreamSession, ChatMessageTimings, DatabaseMessage } from '$lib/types';
import { fStreamIdentity } from '$lib/utils';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

/**
 * The slice of chatStore the manager drives. Kept narrow on purpose so the
 * manager cannot reach around the host's full surface; chatStore implements
 * this structurally.
 */
export interface ChatStreamHost {
  activity: ChatActivityStore;
  processing: ChatProcessingStore;
  vChatStreamingStates: SvelteMap<
    string,
    { response: string; messageId: string; model?: string | null }
  >;
  vStreamConnectionState: StreamConnectionState;
  mGetOrCreateAbortController(pConvId: string): AbortController;
  mSetChatLoading(pConvId: string, pLoading: boolean): void;
  mSetChatStreaming(
    pConvId: string,
    pResponse: string,
    pMessageId: string,
    pModel?: string | null
  ): void;
  mClearChatStreaming(pConvId: string, pMessageId?: string): void;
}

export class ChatStreamManager {
  // in-flight discoverActiveStream guard, keyed by conv id
  private vDiscoveringConvs = new SvelteSet<string>();
  // convs whose resume waits on a model load: their loading state belongs to the retry loop,
  // so discoverActiveStream must not treat it as a live send and bail
  private vResumePendingConvs = new SvelteSet<string>();
  // pending resume retry timers while an owning model loads, one per conv
  private vResumeRetryTimers = new SvelteMap<string, ReturnType<typeof setTimeout>>();

  /** Kill a pending resume retry, e.g. on explicit stop. */
  mCancelResumeRetry(pConvId: string): void {
    const cTimer = this.vResumeRetryTimers.get(pConvId);

    if (cTimer !== undefined) {
      clearTimeout(cTimer);
      this.vResumeRetryTimers.delete(pConvId);
    }

    this.vResumePendingConvs.delete(pConvId);
  }

  constructor(private pHost: ChatStreamHost) {}

  async mDiscoverActiveStream(pConvId: string): Promise<void> {
    if (!pConvId) return;

    if (this.pHost.vChatStreamingStates.has(pConvId)) return;

    if (this.pHost.activity.mIsLocal(pConvId) && !this.vResumePendingConvs.has(pConvId)) return;

    // concurrency guard: another discover may already be running for this conv (typical race
    // between mount and visibilitychange on tab switch). a second concurrent fetch on the same
    // /v1/stream would duplicate every byte into the DB message, this guard bounces it
    if (this.vDiscoveringConvs.has(pConvId)) return;

    this.vDiscoveringConvs.add(pConvId);

    try {
      // the model is frozen at POST time, rebuild the exact conv::model identity from the
      // persisted state so the lookup key matches what the server stored. null means a single
      // model conv with no ::suffix, only guess from the dropdown with no persisted state
      const cLocalState = ChatService.mGetStreamState(pConvId);
      const cStreamId = ChatService.mResumeStreamIdentity(
        pConvId,
        cLocalState,
        cModelsStore.vSelectedModelName
      );
      // primary path: ask the server which sessions exist for this identity
      const cServerTarget = await this.mProbeServerStream(cStreamId);

      if (cServerTarget) {
        // pass the full server side identity (may carry a ::model suffix) so the GET routes
        // straight to the owning session, no probe or fan out
        await this.mAttachServerStream(pConvId, cServerTarget.conversation_id);

        return;
      }

      // fallback: local state remembers an interrupted byte offset for this conv, the server may
      // still have a live session matching that identity (we just lost the bytes mid stream). retry
      // with the frozen identity, the server probe inside attachServerStream tells us if it exists
      if (!cLocalState) {
        return;
      }

      // quiet status probe first: a full attach flips the loading UI on every try, probing
      // keeps the retry loop invisible while the owning model is still loading (503)
      const cStatus = await ChatService.mProbeResumeStatus(cStreamId);

      if (cStatus === 503) {
        // make the wait visible: the empty assistant row persisted at send time renders
        // the processing info, whose model load percentage flows from the models feed
        this.vResumePendingConvs.add(pConvId);
        this.pHost.mSetChatLoading(pConvId, true);

        if (!this.vResumeRetryTimers.has(pConvId)) {
          this.vResumeRetryTimers.set(
            pConvId,
            setTimeout(() => {
              this.vResumeRetryTimers.delete(pConvId);
              void this.mDiscoverActiveStream(pConvId);
            }, cStreamResumeRetryMs)
          );
        }

        return;
      }

      if (this.vResumePendingConvs.delete(pConvId) && cStatus !== 200) {
        // the wait is over without a session to attach, drop the visible loading state
        this.pHost.mSetChatLoading(pConvId, false);
      }

      if (cStatus === 0) {
        // transient network failure, the next mount or visibility change retries
        return;
      }

      if (cStatus !== 200) {
        // the session is gone (stopped, TTL expired), nothing to resume anymore
        ChatService.mClearStreamState(pConvId);

        return;
      }

      await this.mAttachServerStream(pConvId, cStreamId);

      // if attachServerStream failed (session gone, TTL expired), clear the local state to avoid retrying forever
      if (!this.pHost.vChatStreamingStates.has(pConvId) && !this.pHost.activity.mIsLocal(pConvId)) {
        ChatService.mClearStreamState(pConvId);
      }
    } finally {
      this.vDiscoveringConvs.delete(pConvId);
    }
  }

  /**
   * Model frozen at send time for a stream awaiting resume, from the persisted stream state.
   * The load progress indicator targets it after a reload, when the message row has no model
   * yet and the dropdown selection may not be restored.
   */
  mGetResumeModel(pConvId: string): string | null {
    return ChatService.mGetStreamState(pConvId)?.model ?? null;
  }

  /**
   * Resync the activity ledger's remote set from the backend. Called by the layout at mount and
   * on visibilitychange, no polling. A snapshot semantic: stale entries for sessions that
   * finalized while the browser was elsewhere are dropped naturally.
   */
  async mSyncRemoteRunningStreams(): Promise<void> {
    // the conversations store loads from IndexedDB asynchronously, the +layout onMount caller
    // fires before that finishes. read ids straight from the DB so the result does not depend
    // on the store init race, and the sidebar spinners light up at first paint for every conv
    // the user owns even if it has not been hydrated into the store yet
    let aIds: string[];

    try {
      const aAll = await DatabaseService.mGetAllConversations();

      aIds = aAll.map((pC) => pC.id).filter((pId) => !!pId);
    } catch (vE) {
      console.warn('syncRemoteRunningStreams DB read failed:', vE);

      return;
    }

    // only ask about conv ids the user already owns
    if (aIds.length === 0) {
      this.pHost.activity.mApplyRemoteSnapshot([]);

      return;
    }

    // rebuild the frozen conv::model identity per conv so a session started with a model still
    // matches. the server response is mapped back to the bare id below for the sidebar set
    const aLookupIds = aIds.map((pId) =>
      ChatService.mResumeStreamIdentity(pId, ChatService.mGetStreamState(pId), null)
    );

    let aSessions: ApiStreamSession[];

    try {
      aSessions = await ChatService.mLookupStreamSessions(aLookupIds);
    } catch (vE) {
      console.warn('syncRemoteRunningStreams lookup failed:', vE);

      return;
    }
    const cRunning = new SvelteSet<string>();

    for (const cS of aSessions) {
      if (cS && !cS.is_done && typeof cS.conversation_id === 'string' && cS.conversation_id) {
        // strip the optional ::model suffix, the sidebar set is keyed by the bare conv id
        const cSepIdx = cS.conversation_id.indexOf(cConversationIdSeparator);
        const cBareId = cSepIdx === -1 ? cS.conversation_id : cS.conversation_id.slice(0, cSepIdx);

        cRunning.add(cBareId);
      }
    }
    this.pHost.activity.mApplyRemoteSnapshot(cRunning);
  }

  private async mAttachServerStream(pConvId: string, pStreamId?: string): Promise<void> {
    if (!pConvId) return;

    if (this.pHost.vChatStreamingStates.has(pConvId)) return;

    // flip the spinner immediately, the user sees activity as soon as the conv becomes active
    this.pHost.mSetChatLoading(pConvId, true);

    // only set the active processing conv if we are looking at it, otherwise a background
    // attach would steal the indicator from the conv the user is currently viewing
    if (pConvId === cConversationsStore.vActiveConversation?.id) {
      this.pHost.processing.mSetActiveConversation(pConvId);
    }

    const fUnlock = () => {
      this.pHost.mSetChatLoading(pConvId, false);
      this.pHost.mClearChatStreaming(pConvId);
    };
    // fetch the replay stream from byte 0, rebuild the assistant message from scratch.
    // resolve the server side identity, fall back to streamIdentity when the caller does not
    // pass a streamId. probeServerStream returns the full id (with ::model suffix when present)
    const cId = pStreamId || fStreamIdentity(pConvId, cModelsStore.vSelectedModelName);

    let vResponse: Response;

    try {
      vResponse = await ChatService.mFetchStreamReplay(cId);
    } catch (vE) {
      console.error(`attachServerStream replay failed for conv ${pConvId}:`, vE);
      fUnlock();

      return;
    }

    // load the target conversation messages by id, not via the active store. when multiple
    // attaches run in parallel the active store may reflect another conv and writing through
    // its index mixes content across convs (CoT flicker, message bleed). by going through the
    // DB we stay isolated, and only mirror into the active store when the attached conv is
    // the one currently displayed
    let aMessages: DatabaseMessage[];

    try {
      aMessages = await DatabaseService.mGetConversationMessages(pConvId);
    } catch (vE) {
      console.error('attachServerStream load messages failed:', vE);
      fUnlock();

      return;
    }

    // locate the slot to splice into, create a placeholder assistant message if there is none.
    // we use the conv-scoped findLastAssistantIdx helpers, they only depend on the array
    let vTargetIdx = this.mFindLastAssistantIdx(aMessages);

    if (vTargetIdx === -1) {
      const cLastUserIdx = this.mFindLastUserIdx(aMessages);

      if (cLastUserIdx === -1) {
        console.warn(
          `attachServerStream: conv ${pConvId} has no user or assistant message, cannot splice`
        );
        fUnlock();

        return;
      }

      try {
        const cPlaceholder = await DatabaseService.mCreateMessageBranch(
          {
            children: [],
            content: '',
            convId: pConvId,
            parent: aMessages[cLastUserIdx].id,
            role: MessageRole.cAssistant,
            timestamp: Date.now(),
            toolCalls: '',
            type: MessageType.TEXT
          } as Omit<DatabaseMessage, 'id'>,
          aMessages[cLastUserIdx].id
        );

        aMessages = [...aMessages, cPlaceholder];
        vTargetIdx = aMessages.length - 1;

        // only push into the active store when this conv is the one displayed right now
        if (pConvId === cConversationsStore.vActiveConversation?.id) {
          cConversationsStore.mAddMessageToActive(cPlaceholder);
        }
      } catch (vE) {
        console.error('attachServerStream placeholder creation failed:', vE);
        fUnlock();

        return;
      }
    }

    if (vTargetIdx === -1) {
      fUnlock();

      return;
    }

    const cTargetMessage = aMessages[vTargetIdx];
    const cTargetMessageId = cTargetMessage.id;
    // when the assistant slot already has content, the running session is a continue or
    // another append flow and its buffer holds only the appended deltas. preserve the prefix
    // and let the replay add to it. when the slot is empty the session buffer holds the whole
    // message so we wipe and rebuild from byte 0
    const cExistingContent = cTargetMessage.content ?? '';
    const cExistingReasoning = cTargetMessage.reasoningContent ?? '';
    const cIsAppendMode = cExistingContent.length > 0;
    // helper: write to the active store only when the attached conv is currently displayed.
    // the lookup by message id is robust to reordering of activeMessages, two parallel attaches
    // can no longer step on each other's indices
    const fWriteActive = (pUpdates: Partial<DatabaseMessage>) => {
      if (pConvId !== cConversationsStore.vActiveConversation?.id) {
        return;
      }

      const cLiveIdx = cConversationsStore.mFindMessageIndex(cTargetMessageId);

      if (cLiveIdx === -1) return;

      cConversationsStore.mUpdateMessageAtIndex(cLiveIdx, pUpdates);
    };

    if (!cIsAppendMode) {
      fWriteActive({ content: '', reasoningContent: undefined });
    }

    // extract the model suffix, the resume calls in handleStreamResponse must reuse the model
    // the session was tagged with, not the live dropdown
    const cSepIdx = cId.indexOf(cConversationIdSeparator);
    const cAttachedModel: string | null = cSepIdx === -1 ? null : cId.slice(cSepIdx + 2);

    this.pHost.mSetChatStreaming(pConvId, cExistingContent, cTargetMessageId, cAttachedModel);
    const cAbortController = this.pHost.mGetOrCreateAbortController(pConvId);

    let vStreamedContent = '';
    let vStreamedReasoningContent = '';

    const fCleanup = () => {
      fUnlock();
      this.pHost.processing.mSetState(pConvId, null);
    };

    try {
      await ChatService.mHandleStreamResponse(
        vResponse,
        (pChunk: string) => {
          vStreamedContent += pChunk;
          const cDisplayed = cIsAppendMode ? cExistingContent + vStreamedContent : vStreamedContent;

          fWriteActive({ content: cDisplayed });
          this.pHost.mSetChatStreaming(pConvId, cDisplayed, cTargetMessageId);
        },
        async (
          pFinalContent?: string,
          pReasoningContent?: string,
          pTimings?: ChatMessageTimings,
          pToolCalls?: string
        ) => {
          const cStreamed = vStreamedContent || pFinalContent || '';
          const cStreamedR = vStreamedReasoningContent || pReasoningContent || '';
          const cContent = cIsAppendMode ? cExistingContent + cStreamed : cStreamed;
          const reasoning = cIsAppendMode ? cExistingReasoning + cStreamedR : cStreamedR;

          // the DB write is the source of truth, mirror to the active store only when
          // the conv is currently displayed
          await DatabaseService.mUpdateMessage(cTargetMessageId, {
            content: cContent,
            reasoningContent: reasoning || undefined,
            timings: pTimings,
            toolCalls: pToolCalls || ''
          });
          fWriteActive({
            content: cContent,
            reasoningContent: reasoning || undefined,
            timings: pTimings
          });
          fCleanup();
        },
        (pErr: Error) => {
          console.error('attachServerStream pipe error:', pErr);
          fCleanup();
        },
        (pChunk: string) => {
          vStreamedReasoningContent += pChunk;
          const cDisplayed = cIsAppendMode
            ? cExistingReasoning + vStreamedReasoningContent
            : vStreamedReasoningContent;

          fWriteActive({ reasoningContent: cDisplayed });
        },
        undefined,
        undefined,
        undefined,
        undefined,
        pConvId,
        cAbortController.signal,
        (pConnState: StreamConnectionState) => {
          if (pConvId === cConversationsStore.vActiveConversation?.id) {
            this.pHost.vStreamConnectionState = pConnState;
          }
        },
        cAttachedModel
      );
    } catch (vE) {
      console.error('attachServerStream pipe crashed:', vE);
      fCleanup();
    }
  }

  private mFindLastAssistantIdx(pMessages: DatabaseMessage[]): number {
    for (let vI = pMessages.length - 1; vI >= 0; vI--) {
      if (pMessages[vI].role === MessageRole.cAssistant) return vI;
    }

    return -1;
  }

  private mFindLastUserIdx(pMessages: DatabaseMessage[]): number {
    for (let vI = pMessages.length - 1; vI >= 0; vI--) {
      if (pMessages[vI].role === MessageRole.cUser) return vI;
    }

    return -1;
  }

  /**
   * Server side stream discovery, split in three pieces:
   *
   * probeServerStream(convId) -> hits POST /v1/streams/lookup with the conv id, returns the session to attach
   *   to or null. Pure read, no side effect, no UI lock. Safe to fire in parallel with anything.
   *
   * attachServerStream(convId) -> flips the spinner immediately, fetches the replay stream
   *   from byte 0, finds the assistant slot to splice into (creates a placeholder if the conv has
   *   no assistant message yet, for cross device or fresh local DB cases), and pipes the SSE bytes
   *   into the message via handleStreamResponse.
   *
   * discoverActiveStream(convId) -> probe + attach in one call. Used by callers that do not need
   *   to overlap the probe with other async work.
   *
   * The chat page in +page.svelte calls discoverActiveStream once the conversation is active
   * (immediately if it already is, after loadConversation settles otherwise), and re-runs it on
   * visibilitychange. Attaching only after the conversation is loaded gives the earliest
   * possible time to spinner and avoids racing against an empty activeMessages array.
   */
  private async mProbeServerStream(pConvId: string): Promise<ApiStreamSession | null> {
    if (!pConvId) return null;

    let aSessions: ApiStreamSession[];

    try {
      aSessions = await ChatService.mLookupStreamSessions([pConvId]);
    } catch (vE) {
      console.warn(`probeServerStream failed for conv ${pConvId}:`, vE);

      return null;
    }

    return ChatService.mSelectActiveStream(aSessions);
  }
}
