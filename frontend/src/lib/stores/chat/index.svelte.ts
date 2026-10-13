import { fTranslate } from '$lib/i18n';
/**
 * chatStore - Chat lifecycle, streaming and message operations
 *
 * Owns the active conversation's chat state: sending messages, streaming
 * responses, editing/regeneration flows and per-conversation processing
 * activity. Composes the stream manager, message flows, activity ledger and
 * processing snapshot; persists through conversationsStore.
 *
 * Uses ChatService for the API layer and conversationsStore for persistence.
 */

import { cCwdClearedText, cSystemMessagePlaceholder, cTitleGeneration } from '$lib/constants';
import {
  ErrorDialogType,
  MessageRole,
  MessageType,
  ReasoningEffort,
  StreamConnectionState
} from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cAgenticStore } from '$lib/stores/agentic/index.svelte';
import { cChatActivityStore } from '$lib/stores/chat/activity.svelte';
import { type ChatFlowsHost, ChatMessageFlows } from '$lib/stores/chat/flows.svelte';
import { cChatProcessingStore } from '$lib/stores/chat/processing.svelte';
import { type ChatStreamHost, ChatStreamManager } from '$lib/stores/chat/streams.svelte';
import { cConversationsStore } from '$lib/stores/conversations/index.svelte';
import { cMcpStore } from '$lib/stores/mcp/index.svelte';
import { cModelsStore } from '$lib/stores/models/index.svelte';
import { cServerStore } from '$lib/stores/server.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import { cToolsStore } from '$lib/stores/tools.svelte';
import type {
  ApiChatMessageData,
  ChatMessagePromptProgress,
  ChatMessageTimings,
  ChatStreamCallbacks,
  DatabaseMessage,
  DatabaseMessageExtra,
  ErrorDialogState
} from '$lib/types';
import {
  fFindMessageById,
  fFormatCwdMessage,
  fGetConversationModel,
  fIsAbortError,
  fNormalizeModelName
} from '$lib/utils';
import { SvelteMap } from 'svelte/reactivity';

class ChatStore implements ChatStreamHost, ChatFlowsHost {
  vChatReasoningStates = new SvelteMap<string, boolean>();
  vChatStreamingStates = new SvelteMap<
    string,
    { response: string; messageId: string; model?: string | null }
  >();
  vErrorDialogState = $state<ErrorDialogState | null>(null);
  // true while the active conversation has a local pipe (send, attach or resume-wait)
  isLoading = $derived(this.activity.mIsLocal(cConversationsStore.vActiveConversation?.id ?? ''));
  // true while the active conversation streams reasoning content but no visible content yet
  vIsReasoning = $derived(
    this.vChatReasoningStates.get(cConversationsStore.vActiveConversation?.id ?? '') ?? false
  );
  vPendingEditMessageId = $state<string | null>(null);
  // resumable stream connection state for the active conversation
  // streaming -> bytes flowing normally, resuming -> waiting on /v1/stream reconnect, lost -> unrecoverable
  vStreamConnectionState = $state<StreamConnectionState>(StreamConnectionState.cStreaming);
  private vAbortControllers = new SvelteMap<string, AbortController>();
  private vAddFilesHandler: ((pFiles: File[]) => void) | null = $state(null);
  // message flows: edit, regenerate, continue, delete
  private vFlows = new ChatMessageFlows(this);
  private vIsEditModeActive = $state(false);
  private aPendingDraftFiles = $state<ChatUploadedFile[]>([]);
  private vPendingDraftMessage = $state<string>('');
  /** Reactive: queued pending messages for non-agentic streaming */
  private vPendingMessages = new SvelteMap<
    string,
    { content: string; extras?: DatabaseMessageExtra[] }
  >();
  private vPreEncodeAbortController: AbortController | null = null;

  // server-side stream sessions: discovery, attach/replay, resume retry, remote sync
  private vStreams = new ChatStreamManager(this);

  /** Conv activity (local pipe / remote session), composed here. */
  get activity() {
    return cChatActivityStore;
  }

  /** Processing state, composed here so consumers have a single chat scope. */
  get processing() {
    return cChatProcessingStore;
  }

  /**
   * Abort the current agentic flow signal without clearing loading state.
   * Used by "Send immediately" to force the agentic loop to exit so that
   * the pending steering message can be re-sent.
   *
   * Any tool calls captured mid-stream are dropped before the abort so the
   * pending message (or a manual follow-up) does not re-send a half-received
   * tool call with invalid JSON arguments to the server. Mirrors what the
   * Stop button already does through stopGenerationForChat.
   */
  async mAbortCurrentFlow(pConvId: string): Promise<void> {
    await this.mSavePartialResponseIfNeeded(pConvId);
    const c = this.vAbortControllers.get(pConvId);

    if (c) {
      c.abort();
      this.vAbortControllers.delete(pConvId);
    }
  }

  async mAddMessage(
    pRole: MessageRole,
    pContent: string,
    pType: MessageType = MessageType.TEXT,
    pParent: string = '-1',
    pExtras?: DatabaseMessageExtra[],
    pIsSynthetic?: boolean
  ): Promise<DatabaseMessage> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) throw new Error(fTranslate('text_f9edd3337773'));

    let vParentId: string | null = null;

    if (pParent === '-1') {
      const aAm = cConversationsStore.aActiveMessages;

      if (aAm.length > 0) vParentId = aAm[aAm.length - 1].id;
      else {
        const aAll = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
        const r = aAll.find((pM) => pM.parent === null && pM.type === 'root');

        vParentId = r ? r.id : await DatabaseService.mCreateRootMessage(cActiveConv.id);
      }
    } else vParentId = pParent;

    const message = await DatabaseService.mCreateMessageBranch(
      {
        children: [],
        content: pContent,
        convId: cActiveConv.id,
        extra: pExtras,
        isSynthetic: pIsSynthetic,
        role: pRole,
        timestamp: Date.now(),
        toolCalls: '',
        type: pType
      },
      vParentId
    );

    cConversationsStore.mAddMessageToActive(message);
    await cConversationsStore.mUpdateCurrentNode(message.id);
    cConversationsStore.mUpdateConversationTimestamp();

    return message;
  }
  async mAddSystemPrompt(): Promise<void> {
    let vActiveConv = cConversationsStore.vActiveConversation;

    if (!vActiveConv) {
      await cConversationsStore.mCreateConversation();
      vActiveConv = cConversationsStore.vActiveConversation;
    }

    if (!vActiveConv) return;

    try {
      const aAllMessages = await cConversationsStore.mGetConversationMessages(vActiveConv.id);
      const cRootMessage = aAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
      const cRootId = cRootMessage
        ? cRootMessage.id
        : await DatabaseService.mCreateRootMessage(vActiveConv.id);
      const cExistingSystemMessage = aAllMessages.find(
        (pM) => pM.role === MessageRole.cSystem && pM.parent === cRootId
      );

      if (cExistingSystemMessage) {
        this.vPendingEditMessageId = cExistingSystemMessage.id;

        if (!cConversationsStore.aActiveMessages.some((pM) => pM.id === cExistingSystemMessage.id))
          cConversationsStore.aActiveMessages.unshift(cExistingSystemMessage);

        return;
      }

      const aAm = cConversationsStore.aActiveMessages;
      const cFirstActiveMessage = aAm.find((pM) => pM.parent === cRootId);
      const systemMessage = await DatabaseService.mCreateSystemMessage(
        vActiveConv.id,
        cSystemMessagePlaceholder,
        cRootId
      );

      if (cFirstActiveMessage) {
        await DatabaseService.mUpdateMessage(cFirstActiveMessage.id, {
          parent: systemMessage.id
        });
        await DatabaseService.mUpdateMessage(systemMessage.id, {
          children: [cFirstActiveMessage.id]
        });
        const cUpdatedRootChildren = cRootMessage
          ? cRootMessage.children.filter((pId: string) => pId !== cFirstActiveMessage.id)
          : [];

        await DatabaseService.mUpdateMessage(cRootId, {
          children: [
            ...cUpdatedRootChildren.filter((pId: string) => pId !== systemMessage.id),
            systemMessage.id
          ]
        });
        const cFirstMsgIndex = cConversationsStore.mFindMessageIndex(cFirstActiveMessage.id);

        if (cFirstMsgIndex !== -1)
          cConversationsStore.mUpdateMessageAtIndex(cFirstMsgIndex, {
            parent: systemMessage.id
          });
      }

      cConversationsStore.aActiveMessages.unshift(systemMessage);
      this.vPendingEditMessageId = systemMessage.id;
      cConversationsStore.mUpdateConversationTimestamp();
    } catch (error) {
      console.error('Failed to add system prompt:', error);
    }
  }
  mCancelPreEncode(): void {
    if (this.vPreEncodeAbortController) {
      this.vPreEncodeAbortController.abort();
      this.vPreEncodeAbortController = null;
    }
  }

  /**
   * Resets the loading, streaming and processing state for a conversation
   * after a generation ends or errors. Shared by the flows' exit paths.
   */
  mCleanupStreaming(pConvId: string): void {
    this.mSetChatLoading(pConvId, false);
    this.mClearChatStreaming(pConvId);
    this.processing.mSetState(pConvId, null);
  }
  mClearChatStreaming(pConvId: string, pMessageId?: string): void {
    // session aware: a stale generation must not wipe a newer one's streaming state on the
    // same conversation, that would drop the frozen stop identity and stop the wrong session
    if (pMessageId !== undefined) {
      const cCur = this.vChatStreamingStates.get(pConvId);

      if (cCur && cCur.messageId !== pMessageId) return;
    }

    this.vChatStreamingStates.delete(pConvId);
  }
  mClearEditMode(): void {
    this.vIsEditModeActive = false;
    this.vAddFilesHandler = null;
  }

  mClearPendingEditMessageId(): void {
    this.vPendingEditMessageId = null;
  }

  mClearPendingMessage(pConvId: string): void {
    this.vPendingMessages.delete(pConvId);
  }

  mConsumePendingDraft(): { message: string; files: ChatUploadedFile[] } | null {
    if (!this.vPendingDraftMessage && this.aPendingDraftFiles.length === 0) return null;

    const cD = { files: [...this.aPendingDraftFiles], message: this.vPendingDraftMessage };

    this.vPendingDraftMessage = '';
    this.aPendingDraftFiles = [];

    return cD;
  }

  mConsumePendingMessage(
    pConvId: string
  ): { content: string; extras?: DatabaseMessageExtra[] } | null {
    const cMsg = this.vPendingMessages.get(pConvId);

    if (!cMsg) return null;

    this.vPendingMessages.delete(pConvId);

    return cMsg;
  }

  async mContinueAssistantMessage(pMessageId: string): Promise<void> {
    return this.vFlows.mContinueAssistantMessage(pMessageId);
  }

  async mCreateAssistantMessage(pParentId?: string): Promise<DatabaseMessage> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) throw new Error(fTranslate('text_f9edd3337773'));

    return await DatabaseService.mCreateMessageBranch(
      {
        children: [],
        content: '',
        convId: cActiveConv.id,
        model: null,
        role: MessageRole.cAssistant,
        timestamp: Date.now(),
        toolCalls: '',
        type: MessageType.TEXT
      },
      pParentId || null
    );
  }

  async mDeleteMessage(pMessageId: string): Promise<void> {
    return this.vFlows.mDeleteMessage(pMessageId);
  }

  /**
   * Server-side stream sessions (discovery, attach/replay, resume retry,
   * remote-running snapshot) live in ChatStreamManager.
   */
  async mDiscoverActiveStream(pConvId: string): Promise<void> {
    return this.vStreams.mDiscoverActiveStream(pConvId);
  }

  mDismissErrorDialog(): void {
    this.vErrorDialogState = null;
  }

  async mEditAssistantMessage(
    pMessageId: string,
    pNewContent: string,
    pShouldBranch: boolean
  ): Promise<void> {
    return this.vFlows.mEditAssistantMessage(pMessageId, pNewContent, pShouldBranch);
  }

  async mEditMessageWithBranching(
    pMessageId: string,
    pNewContent: string,
    pNewExtras?: DatabaseMessageExtra[]
  ): Promise<void> {
    return this.vFlows.mEditMessageWithBranching(pMessageId, pNewContent, pNewExtras);
  }

  async mEditUserMessagePreserveResponses(
    pMessageId: string,
    pNewContent: string,
    pNewExtras?: DatabaseMessageExtra[]
  ): Promise<void> {
    return this.vFlows.mEditUserMessagePreserveResponses(pMessageId, pNewContent, pNewExtras);
  }

  mGetAddFilesHandler(): ((pFiles: File[]) => void) | null {
    return this.vAddFilesHandler;
  }

  /** Convs with any activity (local pipe or remote session), sidebar spinners. */
  mGetAllLoadingChats(): string[] {
    return this.activity.aLoadingConvs;
  }

  mGetApiOptions(): Record<string, unknown> {
    const cCurrentConfig = cSettingsStore.vConfig;
    const fHasValue = (pValue: unknown): boolean =>
      pValue !== undefined && pValue !== null && pValue !== '';
    const dApiOptions: Record<string, unknown> = { stream: true, timings_per_token: true };

    if (cServerStore.isRouterMode) {
      const cModelName = cModelsStore.vSelectedModelName;

      if (cModelName) dApiOptions.model = cModelName;
    }

    if (cCurrentConfig.systemMessage) dApiOptions.systemMessage = cCurrentConfig.systemMessage;

    if (cCurrentConfig.disableReasoningParsing) dApiOptions.disableReasoningParsing = true;

    if (cCurrentConfig.excludeReasoningFromContext) dApiOptions.excludeReasoningFromContext = true;

    // an explicit reasoning choice overrides the server default, DEFAULT sends nothing
    const cEffort = cConversationsStore.preferences.mGetReasoningEffort();

    if (cEffort !== ReasoningEffort.cDefault) {
      dApiOptions.enableThinking = cEffort !== ReasoningEffort.cOff;

      if (cEffort !== ReasoningEffort.cOff) dApiOptions.reasoningEffort = cEffort;
    }

    if (fHasValue(cCurrentConfig.temperature))
      dApiOptions.temperature = Number(cCurrentConfig.temperature);

    if (fHasValue(cCurrentConfig.max_tokens))
      dApiOptions.max_tokens = Number(cCurrentConfig.max_tokens);

    if (fHasValue(cCurrentConfig.dynatemp_range))
      dApiOptions.dynatemp_range = Number(cCurrentConfig.dynatemp_range);

    if (fHasValue(cCurrentConfig.dynatemp_exponent))
      dApiOptions.dynatemp_exponent = Number(cCurrentConfig.dynatemp_exponent);

    if (fHasValue(cCurrentConfig.top_k)) dApiOptions.top_k = Number(cCurrentConfig.top_k);

    if (fHasValue(cCurrentConfig.top_p)) dApiOptions.top_p = Number(cCurrentConfig.top_p);

    if (fHasValue(cCurrentConfig.min_p)) dApiOptions.min_p = Number(cCurrentConfig.min_p);

    if (fHasValue(cCurrentConfig.xtc_probability))
      dApiOptions.xtc_probability = Number(cCurrentConfig.xtc_probability);

    if (fHasValue(cCurrentConfig.xtc_threshold))
      dApiOptions.xtc_threshold = Number(cCurrentConfig.xtc_threshold);

    if (fHasValue(cCurrentConfig.typ_p)) dApiOptions.typ_p = Number(cCurrentConfig.typ_p);

    if (fHasValue(cCurrentConfig.repeat_last_n))
      dApiOptions.repeat_last_n = Number(cCurrentConfig.repeat_last_n);

    if (fHasValue(cCurrentConfig.repeat_penalty))
      dApiOptions.repeat_penalty = Number(cCurrentConfig.repeat_penalty);

    if (fHasValue(cCurrentConfig.presence_penalty))
      dApiOptions.presence_penalty = Number(cCurrentConfig.presence_penalty);

    if (fHasValue(cCurrentConfig.frequency_penalty))
      dApiOptions.frequency_penalty = Number(cCurrentConfig.frequency_penalty);

    if (fHasValue(cCurrentConfig.dry_multiplier))
      dApiOptions.dry_multiplier = Number(cCurrentConfig.dry_multiplier);

    if (fHasValue(cCurrentConfig.dry_base)) dApiOptions.dry_base = Number(cCurrentConfig.dry_base);

    if (fHasValue(cCurrentConfig.dry_allowed_length))
      dApiOptions.dry_allowed_length = Number(cCurrentConfig.dry_allowed_length);

    if (fHasValue(cCurrentConfig.dry_penalty_last_n))
      dApiOptions.dry_penalty_last_n = Number(cCurrentConfig.dry_penalty_last_n);

    if (cCurrentConfig.samplers) dApiOptions.samplers = cCurrentConfig.samplers;

    if (fHasValue(cCurrentConfig.backend_sampling))
      dApiOptions.backend_sampling = cCurrentConfig.backend_sampling;

    if (cCurrentConfig.customJson) dApiOptions.custom = cCurrentConfig.customJson;

    return dApiOptions;
  }

  mGetChatStreaming(pConvId: string): { response: string; messageId: string } | undefined {
    return this.mGetChatStreamingState(pConvId);
  }

  async mGetDeletionInfo(pMessageId: string): Promise<{
    totalCount: number;
    userMessages: number;
    assistantMessages: number;
    messageTypes: string[];
  }> {
    return this.vFlows.mGetDeletionInfo(pMessageId);
  }

  mGetOrCreateAbortController(pConvId: string): AbortController {
    let vC = this.vAbortControllers.get(pConvId);

    if (!vC || vC.signal.aborted) {
      vC = new AbortController();
      this.vAbortControllers.set(pConvId, vC);
    }

    return vC;
  }

  mGetPendingMessageContent(pConvId: string): string | null {
    return this.vPendingMessages.get(pConvId)?.content ?? null;
  }

  mGetPendingMessageExtras(pConvId: string): DatabaseMessageExtra[] | undefined {
    return this.vPendingMessages.get(pConvId)?.extras;
  }

  mGetResumeModel(pConvId: string): string | null {
    return this.vStreams.mGetResumeModel(pConvId);
  }

  mHasPendingDraft(): boolean {
    return Boolean(this.vPendingDraftMessage) || this.aPendingDraftFiles.length > 0;
  }

  mHasPendingMessage(pConvId: string): boolean {
    return this.vPendingMessages.has(pConvId);
  }

  mInjectPendingMessage(pConvId: string, pContent: string, pExtras?: DatabaseMessageExtra[]): void {
    this.vPendingMessages.set(pConvId, { content: pContent, extras: pExtras });
  }

  mIsChatLoading(pConvId: string): boolean {
    return this.activity.mIsLocal(pConvId);
  }

  mIsChatLoadingInternal(pConvId: string): boolean {
    return this.activity.mIsLocal(pConvId) || this.vChatStreamingStates.has(pConvId);
  }

  mIsEditing(): boolean {
    return this.vIsEditModeActive;
  }

  /** True while the active conversation has a live streaming pipe. */
  mIsStreaming(): boolean {
    return this.vChatStreamingStates.has(cConversationsStore.vActiveConversation?.id ?? '');
  }

  /**
   * Record a working-directory change into chat history as a synthetic
   * user message, so the model sees it on its next turn (the client
   * sends the cwd itself via the x-tool-cwd header on tool calls).
   * A plain user message is used because some chat templates reject
   * tool messages without a preceding tool call.
   */
  async mRecordCwdChange(pCwd: string | null): Promise<void> {
    const cContent = pCwd
      ? fFormatCwdMessage(pCwd, await cToolsStore.mResolveServerHome())
      : cCwdClearedText;
    // Reuse the trailing cwd row when it is already the last message, so
    // repeated picks update it in place instead of stacking another row.
    const cLast =
      cConversationsStore.aActiveMessages[cConversationsStore.aActiveMessages.length - 1];

    if (cLast && cLast.role === MessageRole.cUser && cLast.isSynthetic === true) {
      await DatabaseService.mUpdateMessage(cLast.id, { content: cContent, isSynthetic: true });
      cConversationsStore.mUpdateMessageAtIndex(cConversationsStore.aActiveMessages.length - 1, {
        content: cContent,
        isSynthetic: true
      });

      return;
    }

    await this.mAddMessage(MessageRole.cUser, cContent, MessageType.TEXT, '-1', undefined, true);
  }

  async mRegenerateMessage(pMessageId: string): Promise<void> {
    return this.vFlows.mRegenerateMessage(pMessageId);
  }

  async mRegenerateMessageWithBranching(
    pMessageId: string,
    pModelOverride?: string
  ): Promise<void> {
    return this.vFlows.mRegenerateMessageWithBranching(pMessageId, pModelOverride);
  }

  async mRemoveSystemPromptPlaceholder(pMessageId: string): Promise<boolean> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) return false;

    try {
      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const systemMessage = fFindMessageById(aAllMessages, pMessageId);

      if (!systemMessage || systemMessage.role !== MessageRole.cSystem) return false;

      const cRootMessage = aAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);

      if (!cRootMessage) return false;

      if (aAllMessages.length === 2 && systemMessage.children.length === 0) {
        await cConversationsStore.mDeleteConversation(cActiveConv.id);

        return true;
      }

      for (const cChildId of systemMessage.children) {
        await DatabaseService.mUpdateMessage(cChildId, { parent: cRootMessage.id });
        const cChildIndex = cConversationsStore.mFindMessageIndex(cChildId);

        if (cChildIndex !== -1)
          cConversationsStore.mUpdateMessageAtIndex(cChildIndex, { parent: cRootMessage.id });
      }
      await DatabaseService.mUpdateMessage(cRootMessage.id, {
        children: [
          ...cRootMessage.children.filter((pId: string) => pId !== pMessageId),
          ...systemMessage.children
        ]
      });
      await DatabaseService.mDeleteMessage(pMessageId);
      const cSystemIndex = cConversationsStore.mFindMessageIndex(pMessageId);

      if (cSystemIndex !== -1) cConversationsStore.aActiveMessages.splice(cSystemIndex, 1);

      cConversationsStore.mUpdateConversationTimestamp();

      return false;
    } catch (error) {
      console.error('Failed to remove system prompt placeholder:', error);

      return false;
    }
  }

  mSavePendingDraft(pMessage: string, pFiles: ChatUploadedFile[]): void {
    this.vPendingDraftMessage = pMessage;
    this.aPendingDraftFiles = [...pFiles];
  }
  async mSendMessage(pContent: string, pExtras?: DatabaseMessageExtra[]): Promise<void> {
    if (!pContent.trim() && (!pExtras || pExtras.length === 0)) return;

    const cActiveConv = cConversationsStore.vActiveConversation;

    // If agentic loop is running, inject as a steering message instead of starting a new flow
    if (cActiveConv && cAgenticStore.mIsRunning(cActiveConv.id)) {
      cAgenticStore.mInjectSteeringMessage(cActiveConv.id, pContent, pExtras);

      return;
    }

    // If non-agentic streaming is active, queue as a pending message to send after completion
    if (cActiveConv && this.mIsChatLoadingInternal(cActiveConv.id)) {
      this.mInjectPendingMessage(cActiveConv.id, pContent, pExtras);

      return;
    }

    // Cancel any in-flight pre-encode request
    this.mCancelPreEncode();

    // Consume MCP resource attachments - converts them to extras and clears the live store
    const aResourceExtras = cMcpStore.mConsumeResourceAttachmentsAsExtras();
    const cAllExtras =
      aResourceExtras.length > 0 ? [...(pExtras || []), ...aResourceExtras] : pExtras;

    let vIsNewConversation = false;

    if (!cActiveConv) {
      await cConversationsStore.mCreateConversation();
      vIsNewConversation = true;
    }

    const cCurrentConv = cConversationsStore.vActiveConversation;

    if (!cCurrentConv) return;

    this.mShowErrorDialog(null);
    this.mSetChatLoading(cCurrentConv.id, true);
    this.mClearChatStreaming(cCurrentConv.id);
    try {
      let vParentIdForUserMessage: string | undefined;

      if (vIsNewConversation) {
        const cRootId = await DatabaseService.mCreateRootMessage(cCurrentConv.id);
        const cCurrentConfig = cSettingsStore.vConfig;
        const cSystemPrompt = cCurrentConfig.systemMessage?.toString().trim();

        let vSysOrRootId = cRootId;

        if (cSystemPrompt) {
          const systemMessage = await DatabaseService.mCreateSystemMessage(
            cCurrentConv.id,
            cSystemPrompt,
            cRootId
          );

          cConversationsStore.mAddMessageToActive(systemMessage);
          vSysOrRootId = systemMessage.id;
        }

        // Reflect a working directory picked on the new-chat screen into
        // chat history before the first user message, so the model sees
        // it on its first turn. createConversation() has already threaded
        // the pending pick onto the conversation.
        if (cCurrentConv.cwd) {
          const cCwdMessage = await this.mAddMessage(
            MessageRole.cUser,
            fFormatCwdMessage(cCurrentConv.cwd, await cToolsStore.mResolveServerHome()),
            MessageType.TEXT,
            vSysOrRootId,
            undefined,
            true
          );

          vParentIdForUserMessage = cCwdMessage.id;
        } else {
          vParentIdForUserMessage = vSysOrRootId;
        }
      }

      const cUserMessage = await this.mAddMessage(
        MessageRole.cUser,
        pContent,
        MessageType.TEXT,
        vParentIdForUserMessage ?? '-1',
        cAllExtras
      );

      if (vIsNewConversation && pContent)
        await cConversationsStore.mApplyTitleFromContent(cCurrentConv.id, pContent);

      const cAssistantMessage = await this.mCreateAssistantMessage(cUserMessage.id);

      cConversationsStore.mAddMessageToActive(cAssistantMessage);
      await this.mStreamChatCompletion(
        cConversationsStore.aActiveMessages.slice(0, -1),
        cAssistantMessage,
        undefined,
        undefined,
        undefined,
        cSettingsStore.vConfig.titleGenerationUseLLM && vIsNewConversation ? pContent : undefined
      );
    } catch (error) {
      if (fIsAbortError(error)) {
        this.mSetChatLoading(cCurrentConv.id, false);

        return;
      }

      console.error('Failed to send message:', error);
      this.mSetChatLoading(cCurrentConv.id, false);
      const cDialogType =
        error instanceof Error && error.name === 'TimeoutError'
          ? ErrorDialogType.cTimeout
          : ErrorDialogType.SERVER;
      const cContextInfo = (
        error as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
      ).contextInfo;

      this.mShowErrorDialog({
        contextInfo: cContextInfo,
        message: error instanceof Error ? error.message : 'Unknown error',
        type: cDialogType
      });
    }
  }

  mSetChatLoading(pConvId: string, pLoading: boolean): void {
    if (pLoading) {
      this.activity.mMarkLocal(pConvId);
    } else {
      this.activity.mLocalEnded(pConvId);
      this.mSetChatReasoning(pConvId, false);
    }
  }

  mSetChatReasoning(pConvId: string, pReasoning: boolean): void {
    if (pReasoning) this.vChatReasoningStates.set(pConvId, true);
    else this.vChatReasoningStates.delete(pConvId);
  }

  mSetChatStreaming(
    pConvId: string,
    pResponse: string,
    pMessageId: string,
    pModel?: string | null
  ): void {
    this.vChatStreamingStates.set(pConvId, {
      messageId: pMessageId,
      model: pModel ?? this.vChatStreamingStates.get(pConvId)?.model,
      response: pResponse
    });
  }

  mSetEditModeActive(pHandler: (pFiles: File[]) => void): void {
    this.vIsEditModeActive = true;
    this.vAddFilesHandler = pHandler;
  }

  mShowErrorDialog(pState: ErrorDialogState | null): void {
    this.vErrorDialogState = pState;
  }

  async mStopGeneration(): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) return;

    await this.mStopGenerationForChat(cActiveConv.id);
  }

  async mStopGenerationForChat(pConvId: string): Promise<void> {
    await this.mSavePartialResponseIfNeeded(pConvId);
    // tell the server to stop the generation, not just drop the HTTP socket. without this the
    // detached drain keeps producing tokens until eos or max_tokens. use the frozen identity
    // captured when the session started, not the live dropdown
    const cStreamStateForStop = this.vChatStreamingStates.get(pConvId);
    const cModelForStop = cStreamStateForStop?.model ?? ChatService.mGetStreamState(pConvId)?.model;

    void ChatService.mCancelServerStream(pConvId, cModelForStop);
    // an explicit stop leaves nothing to resume and kills a pending resume retry
    ChatService.mClearStreamState(pConvId);
    this.vStreams.mCancelResumeRetry(pConvId);
    this.mAbortRequest(pConvId);
    this.mSetChatLoading(pConvId, false);
    this.mClearChatStreaming(pConvId);
    this.processing.mSetState(pConvId, null);
    this.mClearPendingMessage(pConvId);
  }

  async mStreamChatCompletion(
    pAllMessages: DatabaseMessage[],
    pAssistantMessage: DatabaseMessage,
    pOnComplete?: (pContent: string) => Promise<void>,
    pOnError?: (pError: Error) => void,
    pModelOverride?: string | null,
    pFirstUserMessageContent?: string
  ): Promise<void> {
    // the ::model suffix in the stream identity is only for router mode, where it routes to the
    // owning child. in single-model mode the identity stays the bare conv id so that attach, stop
    // and reattach all agree, regardless of fresh send vs regenerate passing a resolved model
    let vEffectiveModel: string | null | undefined = undefined;

    if (cServerStore.isRouterMode) {
      const cConversationModel = fGetConversationModel(pAllMessages);

      vEffectiveModel = pModelOverride || cModelsStore.vSelectedModelName || cConversationModel;
    }

    if (cServerStore.isRouterMode && vEffectiveModel) {
      if (!cModelsStore.props.mGetModelProps(vEffectiveModel))
        await cModelsStore.props.mFetchModelProps(vEffectiveModel);
    }

    // Mutable state for the current message being streamed
    let vCurrentMessageId = pAssistantMessage.id;
    let vStreamedContent = '';
    let vStreamedReasoningContent = '';
    let vResolvedModel: string | null = null;
    let vModelPersisted = false;

    const cConvId = pAssistantMessage.convId;

    // Tracks the last message created in this flow. Used as the parent for the next
    // turn's assistant message so createAssistantMessage does not have to read
    // conversationsStore.activeMessages, which may belong to a different conversation
    // after the user navigates while the loop is still running.
    let vLastCreatedInFlow = vCurrentMessageId;

    // freeze the POST identity from t0 so a stop cancels with the exact session key,
    // never a stale or empty model resolved later
    this.mSetChatStreaming(cConvId, vStreamedContent, vCurrentMessageId, vEffectiveModel);

    const fRecordModel = (
      pModelName: string | null | undefined,
      pPersistImmediately = true
    ): void => {
      if (!pModelName) return;

      const cN = fNormalizeModelName(pModelName);

      if (!cN || cN === vResolvedModel) return;

      vResolvedModel = cN;
      const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);

      cConversationsStore.mUpdateMessageAtIndex(cIdx, { model: cN });

      if (pPersistImmediately && !vModelPersisted) {
        vModelPersisted = true;
        DatabaseService.mUpdateMessage(vCurrentMessageId, { model: cN }).catch(() => {
          vModelPersisted = false;
          vResolvedModel = null;
        });
      }
    };

    let vCompletionIdRecorded = false;

    const fRecordCompletionId = (pId: string): void => {
      if (!pId || vCompletionIdRecorded) return;

      vCompletionIdRecorded = true;
      const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);

      cConversationsStore.mUpdateMessageAtIndex(cIdx, { completionId: pId });
      DatabaseService.mUpdateMessage(vCurrentMessageId, { completionId: pId }).catch(() => {
        vCompletionIdRecorded = false;
      });
    };
    const fUpdateStreamingUI = () => {
      this.mSetChatStreaming(cConvId, vStreamedContent, vCurrentMessageId, vEffectiveModel);
      const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);

      cConversationsStore.mUpdateMessageAtIndex(cIdx, { content: vStreamedContent });
    };
    const fCleanupStreamingState = () => {
      this.mSetChatLoading(cConvId, false);
      this.mClearChatStreaming(cConvId, vCurrentMessageId);
      this.processing.mSetState(cConvId, null);
    };

    this.processing.mSetActiveConversation(cConvId);
    const cAbortController = this.mGetOrCreateAbortController(cConvId);
    const cStreamCallbacks: ChatStreamCallbacks = {
      createAssistantMessage: async () => {
        // Reset streaming state for new message
        vStreamedContent = '';
        vStreamedReasoningContent = '';

        const cMsg = await DatabaseService.mCreateMessageBranch(
          {
            children: [],
            content: '',
            convId: cConvId,
            model: vResolvedModel,
            role: MessageRole.cAssistant,
            timestamp: Date.now(),
            toolCalls: '',
            type: MessageType.TEXT
          },
          vLastCreatedInFlow
        );

        if (cConversationsStore.vActiveConversation?.id === cConvId) {
          cConversationsStore.mAddMessageToActive(cMsg);
        }

        vCurrentMessageId = cMsg.id;
        vLastCreatedInFlow = cMsg.id;

        return cMsg;
      },
      createToolResultMessage: async (
        pToolCallId: string,
        pContent: string,
        pExtras?: DatabaseMessageExtra[],
        pToolCwd?: string
      ) => {
        const cMsg = await DatabaseService.mCreateMessageBranch(
          {
            children: [],
            content: pContent,
            convId: cConvId,
            extra: pExtras,
            role: MessageRole.TOOL,
            timestamp: Date.now(),
            toolCallId: pToolCallId,
            toolCalls: '',
            toolCwd: pToolCwd,
            type: MessageType.TEXT
          },
          vCurrentMessageId
        );

        // mirror into the active store and move the node pointer only when this
        // conversation is displayed; otherwise persist the node move straight to
        // the db for the owning conv so a foreign conv's currNode stays untouched
        if (cConversationsStore.vActiveConversation?.id === cConvId) {
          cConversationsStore.mAddMessageToActive(cMsg);
          await cConversationsStore.mUpdateCurrentNode(cMsg.id);
        } else {
          await DatabaseService.mUpdateCurrentNode(cConvId, cMsg.id);
        }

        vLastCreatedInFlow = cMsg.id;

        return cMsg;
      },
      onAssistantTurnComplete: async (
        pContent: string,
        pReasoningContent: string | undefined,
        pTimings: ChatMessageTimings | undefined,
        pToolCalls: import('$lib/types/api').ApiChatCompletionToolCall[] | undefined
      ) => {
        const dUpdateData: Record<string, unknown> = {
          content: pContent,
          reasoningContent: pReasoningContent || undefined,
          timings: pTimings,
          toolCalls: pToolCalls ? JSON.stringify(pToolCalls) : ''
        };

        if (vResolvedModel && !vModelPersisted) dUpdateData.model = vResolvedModel;

        await DatabaseService.mUpdateMessage(vCurrentMessageId, dUpdateData);
        const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
        const cUiUpdate: Partial<DatabaseMessage> = {
          content: pContent,
          reasoningContent: pReasoningContent || undefined,
          toolCalls: pToolCalls ? JSON.stringify(pToolCalls) : ''
        };

        if (pTimings) cUiUpdate.timings = pTimings;

        if (vResolvedModel) cUiUpdate.model = vResolvedModel;

        // touch the active ui array and node pointer only when this conversation
        // is displayed; otherwise persist the node move straight to the db so a
        // foreign conv's currNode stays untouched
        if (cConversationsStore.vActiveConversation?.id === cConvId) {
          cConversationsStore.mUpdateMessageAtIndex(cIdx, cUiUpdate);
          await cConversationsStore.mUpdateCurrentNode(vCurrentMessageId);
        } else {
          await DatabaseService.mUpdateCurrentNode(cConvId, vCurrentMessageId);
        }
      },
      onAttachments: (pMessageId: string, pExtras: DatabaseMessageExtra[]) => {
        if (!pExtras.length) return;

        const cIdx = cConversationsStore.mFindMessageIndex(pMessageId);

        if (cIdx === -1) return;

        const cMsg = cConversationsStore.aActiveMessages[cIdx];
        const aUpdatedExtras = [...(cMsg.extra || []), ...pExtras];

        cConversationsStore.mUpdateMessageAtIndex(cIdx, { extra: aUpdatedExtras });
        DatabaseService.mUpdateMessage(pMessageId, { extra: aUpdatedExtras }).catch(console.error);
      },
      onChunk: (pChunk: string) => {
        vStreamedContent += pChunk;
        fUpdateStreamingUI();
        this.mSetChatReasoning(cConvId, false);
      },
      onCompletionId: (pId: string) => fRecordCompletionId(pId),
      onError: async (pError: Error) => {
        if (fIsAbortError(pError)) {
          fCleanupStreamingState();
          // If aborted with a pending message (e.g. "Send immediately"), re-send it
          const cPending = this.mConsumePendingMessage(cConvId);

          if (cPending) {
            this.mSendMessage(cPending.content, cPending.extras);
          }

          return;
        }

        console.error('Streaming error:', pError);
        // keep whatever was streamed so far, the message stays in memory and in DB
        await this.mSavePartialResponseIfNeeded(cConvId);
        fCleanupStreamingState();
        this.mClearPendingMessage(cConvId);

        const cContextInfo = (
          pError as Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }
        ).contextInfo;

        this.mShowErrorDialog({
          contextInfo: cContextInfo,
          message: pError.message,
          type: pError.name === 'TimeoutError' ? ErrorDialogType.cTimeout : ErrorDialogType.SERVER
        });

        if (pOnError) pOnError(pError);
      },
      onFlowComplete: (pFinalTimings?: ChatMessageTimings) => {
        if (pFinalTimings) {
          const cIdx = cConversationsStore.mFindMessageIndex(pAssistantMessage.id);

          cConversationsStore.mUpdateMessageAtIndex(cIdx, { timings: pFinalTimings });
          DatabaseService.mUpdateMessage(pAssistantMessage.id, {
            timings: pFinalTimings
          }).catch(console.error);
        }

        fCleanupStreamingState();

        if (pOnComplete) pOnComplete(vStreamedContent);

        if (cServerStore.isRouterMode) cModelsStore.mFetchRouterModels().catch(console.error);

        // Pre-encode conversation in KV cache for faster next turn
        if (cSettingsStore.vConfig.preEncodeConversation) {
          this.mTriggerPreEncode(
            pAllMessages,
            pAssistantMessage,
            vStreamedContent,
            vEffectiveModel,
            !!cSettingsStore.vConfig.excludeReasoningFromContext
          );
        }
      },
      onModel: (pModelName: string) => fRecordModel(pModelName),
      onReasoningChunk: (pChunk: string) => {
        vStreamedReasoningContent += pChunk;
        // mark streaming state so a stop mid-thinking can persist the partial reasoning
        this.mSetChatStreaming(cConvId, vStreamedContent, vCurrentMessageId, vEffectiveModel);
        const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);

        cConversationsStore.mUpdateMessageAtIndex(cIdx, {
          reasoningContent: vStreamedReasoningContent
        });
        this.mSetChatReasoning(cConvId, true);
      },
      onTimings: (pTimings?: ChatMessageTimings, pPromptProgress?: ChatMessagePromptProgress) => {
        this.processing.mApplyStreamTimings(pTimings, pPromptProgress, cConvId);
      },
      onToolCallsStreaming: (pToolCalls) => {
        const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);

        cConversationsStore.mUpdateMessageAtIndex(cIdx, {
          toolCalls: JSON.stringify(pToolCalls)
        });
      },
      onTurnComplete: (pIntermediateTimings: ChatMessageTimings) => {
        // Update the first assistant message with cumulative agentic timings
        const cIdx = cConversationsStore.mFindMessageIndex(pAssistantMessage.id);

        cConversationsStore.mUpdateMessageAtIndex(cIdx, { timings: pIntermediateTimings });
      },
      updateToolResultMessage: async (
        pMessageId: string,
        pContent: string,
        pExtras?: DatabaseMessageExtra[]
      ) => {
        // Persist latest content + merged extras; mirror into the active
        // store so the chat view sees live updates for streaming tools
        // (e.g. exec_shell_command). The existing tool message node
        // pointer stays put - the renderer is already scoped to it.
        const cUpdates: Partial<DatabaseMessage> = { content: pContent };

        if (pExtras) {
          const cIdx = cConversationsStore.mFindMessageIndex(pMessageId);
          const cExisting =
            cIdx >= 0 ? (cConversationsStore.aActiveMessages[cIdx]?.extra ?? []) : [];
          const aMerged = [...cExisting, ...pExtras];

          cUpdates.extra = aMerged;
        }

        if (cConversationsStore.vActiveConversation?.id === cConvId) {
          const cIdx = cConversationsStore.mFindMessageIndex(pMessageId);

          if (cIdx >= 0) cConversationsStore.mUpdateMessageAtIndex(cIdx, cUpdates);
        }

        await DatabaseService.mUpdateMessage(pMessageId, cUpdates);
      }
    };
    const cToolPolicy = {
      disabledToolCategories: cConversationsStore.preferences.mGetDisabledToolCategories(),
      disabledTools: cConversationsStore.preferences.mGetDisabledTools()
    };

    {
      const cAgenticResult = await cAgenticStore.mRunAgenticFlow({
        callbacks: cStreamCallbacks,
        conversationId: cConvId,
        flowRootMessageId: pAssistantMessage.id,
        messages: pAllMessages,
        options: {
          ...this.mGetApiOptions(),
          ...(vEffectiveModel ? { model: vEffectiveModel } : {})
        },
        signal: cAbortController.signal,
        toolPolicy: cToolPolicy
      });

      if (cAgenticResult.handled) {
        // Generate LLM based title for new conversations after agentic flow completes
        if (pFirstUserMessageContent) {
          await this.mGenerateTitleWithLLM(pFirstUserMessageContent, vStreamedContent, cConvId);
        }

        // Check if there's a pending steering message to re-send
        const cPending = cAgenticStore.mConsumePendingSteeringMessage(cConvId);

        if (cPending) {
          await this.mSendMessage(cPending.content, cPending.extras);
        }

        return;
      }
    }

    await ChatService.mSendMessage(
      pAllMessages,
      {
        ...this.mGetApiOptions(),
        ...(vEffectiveModel ? { model: vEffectiveModel } : {}),
        onChunk: cStreamCallbacks.onChunk,
        onComplete: async (
          pFinalContent?: string,
          pReasoningContent?: string,
          pTimings?: ChatMessageTimings,
          pToolCalls?: string
        ) => {
          const cContent = vStreamedContent || pFinalContent || '';
          const reasoning = vStreamedReasoningContent || pReasoningContent;
          const dUpdateData: Record<string, unknown> = {
            content: cContent,
            reasoningContent: reasoning || undefined,
            timings: pTimings,
            toolCalls: pToolCalls || ''
          };

          if (vResolvedModel && !vModelPersisted) dUpdateData.model = vResolvedModel;

          await DatabaseService.mUpdateMessage(vCurrentMessageId, dUpdateData);
          const cIdx = cConversationsStore.mFindMessageIndex(vCurrentMessageId);
          const cUiUpdate: Partial<DatabaseMessage> = {
            content: cContent,
            reasoningContent: reasoning || undefined,
            toolCalls: pToolCalls || ''
          };

          if (pTimings) cUiUpdate.timings = pTimings;

          if (vResolvedModel) cUiUpdate.model = vResolvedModel;

          cConversationsStore.mUpdateMessageAtIndex(cIdx, cUiUpdate);
          await cConversationsStore.mUpdateCurrentNode(vCurrentMessageId);
          fCleanupStreamingState();

          if (pOnComplete) await pOnComplete(cContent);

          if (cServerStore.isRouterMode) cModelsStore.mFetchRouterModels().catch(console.error);

          // Generate LLM based title for new conversations (avoids stale reference
          // issue when user switches conversations while streaming)
          if (pFirstUserMessageContent) {
            await this.mGenerateTitleWithLLM(pFirstUserMessageContent, vStreamedContent, cConvId);
          }

          // Check if there's a pending message queued during streaming
          const cPending = this.mConsumePendingMessage(cConvId);

          if (cPending) {
            await this.mSendMessage(cPending.content, cPending.extras);
          }
        },
        onCompletionId: cStreamCallbacks.onCompletionId,
        onConnectionState: (pState: StreamConnectionState) => {
          if (cConvId === cConversationsStore.vActiveConversation?.id) {
            this.vStreamConnectionState = pState;
          }
        },
        onError: cStreamCallbacks.onError,
        onModel: cStreamCallbacks.onModel,
        onReasoningChunk: cStreamCallbacks.onReasoningChunk,
        onTimings: cStreamCallbacks.onTimings,
        stream: true
      },
      cConvId,
      cAbortController.signal
    );
  }

  mSyncLoadingStateForChat(pConvId: string): void {
    const cS = this.vChatStreamingStates.get(pConvId);

    this.processing.mSetActiveConversation(pConvId);

    // Sync streaming content to activeMessages so UI displays current content
    if (cS?.response && cS?.messageId) {
      const cIdx = cConversationsStore.mFindMessageIndex(cS.messageId);

      if (cIdx !== -1) {
        cConversationsStore.mUpdateMessageAtIndex(cIdx, { content: cS.response });
      }
    }
  }

  async mSyncRemoteRunningStreams(): Promise<void> {
    return this.vStreams.mSyncRemoteRunningStreams();
  }

  /**
   * Message flows (edit / regenerate / continue / delete) live in
   * ChatMessageFlows; these delegate so consumers keep a single entry point.
   */
  async mUpdateMessage(pMessageId: string, pNewContent: string): Promise<void> {
    return this.vFlows.mUpdateMessage(pMessageId, pNewContent);
  }
  private mAbortRequest(pConvId?: string): void {
    if (pConvId) {
      const c = this.vAbortControllers.get(pConvId);

      if (c) {
        c.abort();
        this.vAbortControllers.delete(pConvId);
      }
    } else {
      for (const c of this.vAbortControllers.values()) c.abort();
      this.vAbortControllers.clear();
    }
  }

  private async mGenerateTitleWithLLM(
    pUserContent: string,
    pAssistantContent: string,
    pConvId: string
  ): Promise<void> {
    const cEffectiveModel =
      cServerStore.isRouterMode && cModelsStore.vSelectedModelName
        ? cModelsStore.vSelectedModelName
        : undefined;
    const cConfigValue = cSettingsStore.vConfig;
    const cTitlePromptTemplate =
      typeof cConfigValue.titleGenerationPrompt === 'string' &&
      cConfigValue.titleGenerationPrompt.trim()
        ? cConfigValue.titleGenerationPrompt
        : cTitleGeneration.DEFAULT_PROMPT;
    const cTitlePrompt = cTitlePromptTemplate
      .replace('{{USER}}', String(pUserContent || ''))
      .replace('{{ASSISTANT}}', String(pAssistantContent || ''));
    const cTitleMessage: ApiChatMessageData = {
      content: cTitlePrompt,
      role: MessageRole.cUser
    };
    const cTitleResponse = await ChatService.mGenerateTitle(cTitleMessage, cEffectiveModel);

    if (!cTitleResponse) {
      return;
    }

    let vCleanTitle = cTitleResponse.trim();

    vCleanTitle = vCleanTitle
      .replace(cTitleGeneration.PREFIX_PATTERN, '')
      .replace(cTitleGeneration.QUOTE_PATTERN, '')
      .trim();

    if (!vCleanTitle || vCleanTitle.length < cTitleGeneration.MIN_LENGTH) {
      const cFirstLine = pUserContent.split('\n').find((pL) => pL.trim().length > 0);

      vCleanTitle = cFirstLine ? cFirstLine.trim() : cTitleGeneration.FALLBACK;
    }

    if (vCleanTitle && vCleanTitle.length >= cTitleGeneration.MIN_LENGTH) {
      await cConversationsStore.mUpdateConversationName(pConvId, vCleanTitle);
    }
  }

  private mGetChatStreamingState(
    pConvId: string
  ): { response: string; messageId: string } | undefined {
    return this.vChatStreamingStates.get(pConvId);
  }

  private async mSavePartialResponseIfNeeded(pConvId?: string): Promise<void> {
    const cConversationId = pConvId || cConversationsStore.vActiveConversation?.id;

    if (!cConversationId) return;

    const cStreamingState = this.mGetChatStreamingState(cConversationId);

    if (!cStreamingState) return;

    const aMessages =
      cConversationId === cConversationsStore.vActiveConversation?.id
        ? cConversationsStore.aActiveMessages
        : await cConversationsStore.mGetConversationMessages(cConversationId);

    if (!aMessages.length) return;

    const cLastMessage = aMessages[aMessages.length - 1];

    if (cLastMessage?.role !== MessageRole.cAssistant) return;

    const cPartialContent = cStreamingState.response;
    const cPartialReasoning = cLastMessage.reasoningContent || '';
    // snapshot the streamed tool calls before clearing so we still know whether
    // anything was captured when deciding to skip the DB write below
    const cHadPartialToolCalls = !!cLastMessage.toolCalls?.trim();

    // nothing to persist when content, reasoning, and streamed tool calls are all empty
    // (e.g. stop before any token). otherwise drop the partial tool call and write whatever
    // was streamed: incomplete arguments (truncated JSON, missing closing quote) would
    // otherwise be re-sent to the server on the next turn and rejected.
    if (!cPartialContent.trim() && !cPartialReasoning.trim() && !cHadPartialToolCalls) return;

    try {
      const cUpdateData: {
        content?: string;
        reasoningContent?: string;
        toolCalls?: string;
        timings?: ChatMessageTimings;
      } = {
        toolCalls: ''
      };

      if (cPartialContent.trim()) cUpdateData.content = cPartialContent;

      if (cPartialReasoning.trim()) cUpdateData.reasoningContent = cPartialReasoning;

      const cLastKnownState = this.processing.mGetState(cConversationId);

      if (cLastKnownState) {
        cUpdateData.timings = {
          cache_n: cLastKnownState.cacheTokens || 0,
          predicted_ms:
            cLastKnownState.tokensPerSecond && cLastKnownState.tokensDecoded
              ? (cLastKnownState.tokensDecoded / cLastKnownState.tokensPerSecond) * 1000
              : undefined,
          predicted_n: cLastKnownState.tokensDecoded || 0,
          prompt_ms: cLastKnownState.promptMs,
          prompt_n: cLastKnownState.promptTokens || 0
        };
      }

      await DatabaseService.mUpdateMessage(cLastMessage.id, cUpdateData);
      cLastMessage.content = cPartialContent;
      // mirror the drop into the in-memory message so the next request sent via
      // sendMessage (queued pending, Send immediately, or manual follow-up) reads
      // the cleared value, not whatever the streaming widget had been showing
      cLastMessage.toolCalls = '';

      if (cUpdateData.timings) cLastMessage.timings = cUpdateData.timings;
    } catch (error) {
      cLastMessage.content = cPartialContent;
      cLastMessage.toolCalls = '';
      console.error('Failed to save partial response:', error);
    }
  }

  private async mTriggerPreEncode(
    pAllMessages: DatabaseMessage[],
    pAssistantMessage: DatabaseMessage,
    pAssistantContent: string,
    pModel?: string | null,
    pExcludeReasoning?: boolean
  ): Promise<void> {
    this.mCancelPreEncode();
    this.vPreEncodeAbortController = new AbortController();

    const cSignal = this.vPreEncodeAbortController.signal;

    try {
      const cAllIdle = await ChatService.mAreAllSlotsIdle(pModel, cSignal);

      if (!cAllIdle || cSignal.aborted) return;

      const aMessagesWithAssistant: DatabaseMessage[] = [
        ...pAllMessages,
        { ...pAssistantMessage, content: pAssistantContent }
      ];

      await ChatService.mPreEncode(aMessagesWithAssistant, pModel, pExcludeReasoning, cSignal);
    } catch (vErr) {
      if (!fIsAbortError(vErr)) {
        console.warn('[ChatStore] Pre-encode failed:', vErr);
      }
    }
  }
}

export const cChatStore = new ChatStore();
