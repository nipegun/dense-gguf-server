import { fTranslate } from '$lib/i18n';
/**
 * AgenticStore - Multi-turn agentic loop orchestration
 *
 * Drives the agentic loop over MCP tools: streams each LLM turn, detects
 * tool calls, executes them via mcpStore, and enforces the turn limit. Each
 * turn produces one assistant message (with tool_calls) and one tool result
 * message per executed call, persisted as separate DB rows.
 *
 * Uses ChatService for streaming and mcpStore for tool execution; waits on
 * the permission/continue/steering gates owned by {@link AgenticGates}.
 */

import { cDefaultAgenticConfig, cNewline } from '$lib/constants';
import {
  dAudioMimeToExtension,
  cDataUriBase64Regex,
  cDefaultAudioExtension,
  cDefaultImageExtension,
  dImageMimeToExtension,
  cMcpAttachmentNamePrefix,
  cMimeTypePrefixes
} from '$lib/constants';
import { BuiltInTool, ToolPermissionDecision, ToolSource } from '$lib/enums';
import {
  AttachmentType,
  ContentPartType,
  MessageRole,
  MimeTypePrefix,
  ToolCallType
} from '$lib/enums';
import { ChatService } from '$lib/services';
import { ReadMediaService } from '$lib/services/read-media.service';
import { SandboxService } from '$lib/services/sandbox.service';
import { ToolsService } from '$lib/services/tools.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { AgenticGates } from '$lib/stores/agentic/gates.svelte';
import { cConversationsStore } from '$lib/stores/conversations/index.svelte';
import { cMcpStore } from '$lib/stores/mcp/index.svelte';
import { cModelsStore } from '$lib/stores/models/index.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import { cToolsStore } from '$lib/stores/tools.svelte';
import type {
  AgenticConfig,
  AgenticFlowParams,
  AgenticFlowResult,
  AgenticSession,
  MCPToolCall,
  SettingsConfigType,
  ToolExecutionResult
} from '$lib/types';
import type {
  AgenticFlowCallbacks,
  AgenticFlowOptions,
  AgenticMessage,
  AgenticToolCallList,
  SteeringMessage
} from '$lib/types/agentic';
import type {
  ApiChatCompletionToolCall,
  ApiChatMessageContentPart,
  ApiChatMessageData
} from '$lib/types/api';
import type {
  ChatMessageAgenticTimings,
  ChatMessageAgenticTurnStats,
  ChatMessagePromptProgress,
  ChatMessageTimings,
  ChatMessageToolCallTiming
} from '$lib/types/chat';
import type {
  DatabaseMessage,
  DatabaseMessageExtra,
  DatabaseMessageExtraAudioFile,
  DatabaseMessageExtraImageFile
} from '$lib/types/database';
import {
  fExecuteBrowserInfoTool,
  fExecuteGetDatetimeTool,
  fGetAudioInputFormat,
  fIsAbortError
} from '$lib/utils';
import { SvelteMap } from 'svelte/reactivity';

function fCreateDefaultSession(): AgenticSession {
  return {
    currentTurn: 0,
    executingToolCallId: null,
    flowRootMessageId: null,
    isRunning: false,
    lastError: null,
    liveLlm: null,
    pendingPermissionRequest: null,
    streamingToolCall: null,
    totalToolCalls: 0
  };
}

function fToAgenticMessages(pMessages: ApiChatMessageData[]): AgenticMessage[] {
  return pMessages.map((pMessage) => {
    if (
      pMessage.role === MessageRole.cAssistant &&
      pMessage.tool_calls &&
      pMessage.tool_calls.length > 0
    ) {
      return {
        content: pMessage.content,
        reasoning_content: pMessage.reasoning_content,
        role: MessageRole.cAssistant,
        tool_calls: pMessage.tool_calls.map((pCall, pIndex) => ({
          function: {
            arguments: pCall.function?.arguments ?? '',
            name: pCall.function?.name ?? ''
          },
          id: pCall.id ?? `call_${pIndex}`,
          type: (pCall.type as ToolCallType.cFunction) ?? ToolCallType.cFunction
        }))
      } satisfies AgenticMessage;
    }

    if (pMessage.role === MessageRole.cAssistant) {
      return {
        content: pMessage.content,
        reasoning_content: pMessage.reasoning_content,
        role: MessageRole.cAssistant
      } satisfies AgenticMessage;
    }

    if (pMessage.role === MessageRole.TOOL && pMessage.tool_call_id) {
      return {
        content: typeof pMessage.content === 'string' ? pMessage.content : '',
        role: MessageRole.TOOL,
        tool_call_id: pMessage.tool_call_id
      } satisfies AgenticMessage;
    }

    return {
      content: pMessage.content,
      role: pMessage.role as MessageRole.cSystem | MessageRole.cUser
    } satisfies AgenticMessage;
  });
}

class AgenticStore {
  // permission, continue and steering gates the loop waits on between turns
  private vGates = new AgenticGates();
  private vSessions = new SvelteMap<string, AgenticSession>();

  get isAnyRunning(): boolean {
    for (const session of this.vSessions.values()) {
      if (session.isRunning) return true;
    }

    return false;
  }

  get isReady(): boolean {
    return true;
  }

  mClearError(pConversationId: string): void {
    this.mUpdateSession(pConversationId, { lastError: null });
  }

  mClearSession(pConversationId: string): void {
    this.vSessions.delete(pConversationId);
  }

  /**
   * Clear the pending steering message without consuming it.
   */
  mClearSteeringMessage(pConversationId: string): void {
    this.vGates.mClearSteeringMessage(pConversationId);
  }

  constructor() {
    // drop per-conversation session state when the conversation is deleted,
    // otherwise every conversation that ever ran a flow leaks a session here
    cConversationsStore.mOnConversationsDeleted((pConvIds) => {
      for (const cConvId of pConvIds) {
        this.vSessions.delete(cConvId);
      }
    });
  }

  /**
   * Consume and return the pending steering message for re-sending.
   * Called by chatStore after the agentic flow exits.
   */
  mConsumePendingSteeringMessage(pConversationId: string): SteeringMessage | null {
    return this.vGates.mConsumePendingSteeringMessage(pConversationId);
  }

  mGetActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
    const aActive: Array<{ conversationId: string; session: AgenticSession }> = [];

    for (const [vConversationId, vSession] of this.vSessions.entries()) {
      if (vSession.isRunning) aActive.push({ conversationId: vConversationId, session: vSession });
    }

    return aActive;
  }

  mGetConfig(pSettings: SettingsConfigType): AgenticConfig {
    const cMaxTurns = Number(pSettings.agenticMaxTurns) || cDefaultAgenticConfig.maxTurns;
    const cHasTools =
      cMcpStore.mHasEnabledServers() ||
      cToolsStore.serverTools.length > 0 ||
      cToolsStore.browserTools.length > 0 ||
      cToolsStore.customTools.length > 0;

    return {
      enabled: cHasTools && cDefaultAgenticConfig.enabled,
      maxTurns: cMaxTurns
    };
  }

  mGetCurrentTurn(pConversationId: string): number {
    return this.vSessions.get(pConversationId)?.currentTurn ?? 0;
  }

  mGetExecutingToolCallId(pConversationId: string): string | null {
    return this.vSessions.get(pConversationId)?.executingToolCallId ?? null;
  }

  // read-only: safe to call from derivations, unlike getSession
  mGetFlowRootMessageId(pConversationId: string): string | null {
    return this.vSessions.get(pConversationId)?.flowRootMessageId ?? null;
  }

  mGetLastError(pConversationId: string): Error | null {
    return this.vSessions.get(pConversationId)?.lastError ?? null;
  }

  // read-only: safe to call from derivations, unlike getSession
  mGetLiveLlmTotals(pConversationId: string): AgenticSession['liveLlm'] {
    return this.vSessions.get(pConversationId)?.liveLlm ?? null;
  }

  mGetPendingContinueRequest(pConversationId: string): boolean {
    return this.vGates.mGetPendingContinueRequest(pConversationId);
  }

  mGetPendingPermissionRequest(
    pConversationId: string
  ): { toolName: string; serverLabel: string } | null {
    return this.vGates.mGetPendingPermissionRequest(pConversationId);
  }

  mGetPendingSteeringMessageContent(pConversationId: string): string | null {
    return this.vGates.mGetPendingSteeringMessageContent(pConversationId);
  }

  mGetPendingSteeringMessageExtras(pConversationId: string): DatabaseMessageExtra[] | undefined {
    return this.vGates.mGetPendingSteeringMessageExtras(pConversationId);
  }

  mGetSession(pConversationId: string): AgenticSession {
    let vSession = this.vSessions.get(pConversationId);

    if (!vSession) {
      vSession = fCreateDefaultSession();
      this.vSessions.set(pConversationId, vSession);
    }

    return vSession;
  }

  mGetStreamingToolCall(pConversationId: string): { name: string; arguments: string } | null {
    return this.vSessions.get(pConversationId)?.streamingToolCall ?? null;
  }

  mGetTotalToolCalls(pConversationId: string): number {
    return this.vSessions.get(pConversationId)?.totalToolCalls ?? 0;
  }

  mHasPendingSteeringMessage(pConversationId: string): boolean {
    return this.vGates.mHasPendingSteeringMessage(pConversationId);
  }

  /**
   * Queue a steering message. When the current agentic turn completes,
   * the flow exits and the caller re-sends the message as a normal chat message.
   */
  mInjectSteeringMessage(
    pConversationId: string,
    pContent: string,
    pExtras?: DatabaseMessageExtra[]
  ): void {
    this.vGates.mInjectSteeringMessage(pConversationId, pContent, pExtras);
  }

  mIsRunning(pConversationId: string): boolean {
    return this.vSessions.get(pConversationId)?.isRunning ?? false;
  }

  mResolveContinue(pConversationId: string, pShouldContinue: boolean): void {
    this.vGates.mResolveContinue(pConversationId, pShouldContinue);
  }

  mResolvePermission(pConversationId: string, pDecision: ToolPermissionDecision): void {
    this.vGates.mResolvePermission(pConversationId, pDecision);
  }

  async mRunAgenticFlow(pParams: AgenticFlowParams): Promise<AgenticFlowResult> {
    const {
      callbacks: vCallbacks,
      conversationId: vConversationId,
      flowRootMessageId: vFlowRootMessageId,
      messages: aMessages,
      options: vOptions = {},
      signal: vSignal,
      toolPolicy: vToolPolicy
    } = pParams;

    // Clear any pending permissions/continue requests for this conversation when starting a new flow
    this.vGates.mClear(vConversationId);

    // Ensure server tools are fetched before checking if agentic is enabled.
    // A disabled /tools endpoint stays disabled for the life of the server,
    // so the tools panel is the only place that probes it again.
    if (
      cToolsStore.serverTools.length === 0 &&
      !cToolsStore.loading &&
      !cToolsStore.isToolsEndpointUnreachable
    ) {
      await cToolsStore.mFetchServerTools();
    }

    const cAgenticConfig = this.mGetConfig(cSettingsStore.vConfig);

    if (!cAgenticConfig.enabled) return { handled: false };

    // callers without an explicit policy fall back to the global defaults
    const sDisabledTools = new Set(vToolPolicy?.disabledTools ?? cToolsStore.disabledTools);
    const sDisabledToolCategories = new Set(
      vToolPolicy?.disabledToolCategories ?? cToolsStore.disabledToolCategories
    );
    // initialize every settings-enabled server; tool collection filters by this
    // flow's policy, so switching policies never re-initializes connections
    const cHasMcpServers = cConversationsStore.preferences.mPolicyEnabledServerIds().length > 0;

    if (cHasMcpServers) {
      const cInitialized = await cMcpStore.mEnsureInitialized();

      if (!cInitialized) {
        console.log('[AgenticStore] MCP not initialized');
      }
    }

    const aTools = cToolsStore.mGetEnabledToolsForLLM(sDisabledTools, sDisabledToolCategories);

    if (aTools.length === 0) {
      return { handled: false };
    }

    console.log(`[AgenticStore] Starting agentic flow with ${aTools.length} tools`);

    const aNormalizedMessages: ApiChatMessageData[] =
      await ChatService.mNormalizeMessagesForApi(aMessages);

    this.mUpdateSession(vConversationId, {
      currentTurn: 0,
      flowRootMessageId: vFlowRootMessageId ?? null,
      isRunning: true,
      lastError: null,
      liveLlm: null,
      totalToolCalls: 0
    });

    if (cHasMcpServers) cMcpStore.mAcquireConnection();

    try {
      await this.mExecuteAgenticLoop({
        agenticConfig: cAgenticConfig,
        callbacks: vCallbacks,
        conversationId: vConversationId,
        messages: aNormalizedMessages,
        options: vOptions,
        signal: vSignal,
        tools: aTools
      });

      return { handled: true };
    } catch (error) {
      const cNormalizedError = error instanceof Error ? error : new Error(String(error));

      this.mUpdateSession(vConversationId, { lastError: cNormalizedError });
      vCallbacks.onError?.(cNormalizedError);

      return { error: cNormalizedError, handled: true };
    } finally {
      this.mUpdateSession(vConversationId, {
        flowRootMessageId: null,
        isRunning: false,
        liveLlm: null
      });

      if (cHasMcpServers) {
        await cMcpStore
          .mReleaseConnection()
          .catch((pErr: unknown) =>
            console.warn('[AgenticStore] Failed to release MCP connection:', pErr)
          );
      }
    }
  }

  private mBuildAttachmentName(pMimeType: string, pIndex: number): string {
    const cExtension = pMimeType.startsWith(MimeTypePrefix.AUDIO)
      ? (dAudioMimeToExtension[pMimeType] ?? cDefaultAudioExtension)
      : (dImageMimeToExtension[pMimeType] ?? cDefaultImageExtension);

    return `${cMcpAttachmentNamePrefix}-${Date.now()}-${pIndex}.${cExtension}`;
  }

  private mBuildFinalTimings(
    pCapturedTimings: ChatMessageTimings | undefined,
    pAgenticTimings: ChatMessageAgenticTimings
  ): ChatMessageTimings | undefined {
    if (pAgenticTimings.toolCallsCount === 0) return pCapturedTimings;

    return {
      agentic: pAgenticTimings,
      cache_n: pCapturedTimings?.cache_n,
      predicted_ms: pCapturedTimings?.predicted_ms,
      predicted_n: pCapturedTimings?.predicted_n,
      prompt_ms: pCapturedTimings?.prompt_ms,
      prompt_n: pCapturedTimings?.prompt_n
    };
  }

  private async mExecuteAgenticLoop(pParams: {
    conversationId: string;
    messages: ApiChatMessageData[];
    options: AgenticFlowOptions;
    tools: ReturnType<typeof cToolsStore.mGetEnabledToolsForLLM>;
    agenticConfig: AgenticConfig;
    callbacks: AgenticFlowCallbacks;
    signal?: AbortSignal;
  }): Promise<void> {
    const {
      agenticConfig: vAgenticConfig,
      callbacks: vCallbacks,
      conversationId: vConversationId,
      messages: aMessages,
      options: vOptions,
      signal: vSignal,
      tools: aTools
    } = pParams;
    const {
      createAssistantMessage: vCreateAssistantMessage,
      createToolResultMessage: vCreateToolResultMessage,
      onAssistantTurnComplete: vOnAssistantTurnComplete,
      onAttachments: vOnAttachments,
      onChunk: vOnChunk,
      onCompletionId: vOnCompletionId,
      onFlowComplete: vOnFlowComplete,
      onModel: vOnModel,
      onReasoningChunk: vOnReasoningChunk,
      onTimings: vOnTimings,
      onToolCallsStreaming: vOnToolCallsStreaming,
      onTurnComplete: vOnTurnComplete,
      updateToolResultMessage: vUpdateToolResultMessage
    } = vCallbacks;
    const aSessionMessages: AgenticMessage[] = fToAgenticMessages(aMessages);

    let vCapturedTimings: ChatMessageTimings | undefined;
    let vTotalToolCallCount = 0;

    const cAgenticTimings: ChatMessageAgenticTimings = {
      llm: { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 },
      perTurn: [],
      toolCalls: [],
      toolCallsCount: 0,
      toolsMs: 0,
      turns: 0
    };
    const cMaxTurns = vAgenticConfig.maxTurns;
    const cEffectiveModel = vOptions.model || cModelsStore.aModels[0]?.model || '';

    let vTurn = 0;

    while (true) {
      if (vTurn >= cMaxTurns) {
        // Turn limit reached - ask user whether to continue
        const cShouldContinue = await this.vGates.mRequestContinue(vConversationId, vSignal);

        // Yield to allow Svelte to flush the UI update
        await new Promise((pR) => setTimeout(pR, 0));

        if (!cShouldContinue || vSignal?.aborted) {
          vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

          return;
        }

        // User chose to continue - extend the limit
        vTurn = 0;
      }

      this.mUpdateSession(vConversationId, { currentTurn: vTurn + 1 });
      cAgenticTimings.turns = vTurn + 1;

      if (vSignal?.aborted) {
        vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

        return;
      }

      // For turns > 0, create a new assistant message via callback
      if (vTurn > 0 && vCreateAssistantMessage) {
        await vCreateAssistantMessage();
      }

      let vTurnContent = '';
      let vTurnReasoningContent = '';
      let aTurnToolCalls: ApiChatCompletionToolCall[] = [];
      let vLastStreamingToolCallName = '';
      let vLastStreamingToolCallArgsLength = 0;
      let vTurnTimings: ChatMessageTimings | undefined;

      const cTurnStats: ChatMessageAgenticTurnStats = {
        llm: { predicted_ms: 0, predicted_n: 0, prompt_ms: 0, prompt_n: 0 },
        toolCalls: [],
        toolsMs: 0,
        turn: vTurn + 1
      };

      try {
        await ChatService.mSendMessage(
          aSessionMessages as ApiChatMessageData[],
          {
            ...vOptions,
            onChunk: (pChunk: string) => {
              vTurnContent += pChunk;
              vOnChunk?.(pChunk);
            },
            onComplete: () => {
              /* Completion handled after sendMessage resolves */
            },
            onCompletionId: vOnCompletionId,
            onError: (pError: Error) => {
              throw pError;
            },
            onModel: vOnModel,
            onReasoningChunk: (pChunk: string) => {
              vTurnReasoningContent += pChunk;
              vOnReasoningChunk?.(pChunk);
            },
            onTimings: (pTimings?: ChatMessageTimings, pProgress?: ChatMessagePromptProgress) => {
              vOnTimings?.(pTimings, pProgress);

              if (pTimings) {
                vCapturedTimings = pTimings;
                vTurnTimings = pTimings;

                // completed turns + in-flight turn live counts
                this.mUpdateSession(vConversationId, {
                  liveLlm: {
                    predicted_ms: cAgenticTimings.llm.predicted_ms + (pTimings.predicted_ms ?? 0),
                    predicted_n: cAgenticTimings.llm.predicted_n + (pTimings.predicted_n ?? 0),
                    prompt_ms: cAgenticTimings.llm.prompt_ms + (pTimings.prompt_ms ?? 0),
                    prompt_n: cAgenticTimings.llm.prompt_n + (pTimings.prompt_n ?? 0)
                  }
                });
              }
            },
            onToolCallChunk: (pSerialized: string) => {
              try {
                aTurnToolCalls = JSON.parse(pSerialized) as ApiChatCompletionToolCall[];

                vOnToolCallsStreaming?.(aTurnToolCalls);

                if (aTurnToolCalls.length > 0 && aTurnToolCalls[0]?.function) {
                  const name = aTurnToolCalls[0].function.name || '';
                  const cArgs = aTurnToolCalls[0].function.arguments || '';
                  const cArgsLengthBucket = Math.floor(cArgs.length / 100);

                  if (
                    name !== vLastStreamingToolCallName ||
                    cArgsLengthBucket !== vLastStreamingToolCallArgsLength
                  ) {
                    vLastStreamingToolCallName = name;
                    vLastStreamingToolCallArgsLength = cArgsLengthBucket;
                    this.mUpdateSession(vConversationId, {
                      streamingToolCall: { arguments: cArgs, name }
                    });
                  }
                }
              } catch {
                /* Ignore parse errors during streaming */
              }
            },
            stream: true,
            tools: aTools.length > 0 ? aTools : undefined
          },
          vConversationId,
          vSignal
        );

        this.mUpdateSession(vConversationId, { streamingToolCall: null });

        if (vTurnTimings) {
          cAgenticTimings.llm.predicted_n += vTurnTimings.predicted_n || 0;
          cAgenticTimings.llm.predicted_ms += vTurnTimings.predicted_ms || 0;
          cAgenticTimings.llm.prompt_n += vTurnTimings.prompt_n || 0;
          cAgenticTimings.llm.prompt_ms += vTurnTimings.prompt_ms || 0;
          cTurnStats.llm.predicted_n = vTurnTimings.predicted_n || 0;
          cTurnStats.llm.predicted_ms = vTurnTimings.predicted_ms || 0;
          cTurnStats.llm.prompt_n = vTurnTimings.prompt_n || 0;
          cTurnStats.llm.prompt_ms = vTurnTimings.prompt_ms || 0;
        }
      } catch (error) {
        if (vSignal?.aborted) {
          // Save whatever we have for this turn before exiting
          await vOnAssistantTurnComplete?.(
            vTurnContent,
            vTurnReasoningContent || undefined,
            this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings),
            undefined
          );
          vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

          return;
        }

        const cNormalizedError =
          error instanceof Error ? error : new Error(fTranslate('text_80c39c3fcd37'));

        // preserve partial output as is, the outer error dialog informs the user separately
        await vOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings),
          undefined
        );
        vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

        throw cNormalizedError;
      }

      // If the abort landed while ChatService.sendMessage was still resolving, the
      // outer catch above never fires because ChatService swallows the AbortError
      // and returns normally. Bail out here so a half-received tool_call (truncated
      // arguments JSON) is not persisted as if it were complete.
      if (vSignal?.aborted) {
        await vOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings),
          undefined
        );
        vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

        return;
      }

      // === Steering check: if a user message was queued during this turn, exit the flow.
      // The caller (chatStore) will consume the pending message and re-send it normally.
      if (this.vGates.mHasPendingSteeringMessage(vConversationId)) {
        console.log('[AgenticStore] Steering message detected after turn, exiting agentic flow');
        await vOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings),
          aTurnToolCalls.length > 0 ? this.mNormalizeToolCalls(aTurnToolCalls) : undefined
        );
        vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

        return;
      }

      // No tool calls = final turn, save and complete
      if (aTurnToolCalls.length === 0) {
        cAgenticTimings.perTurn!.push(cTurnStats);

        const cFinalTimings = this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings);

        await vOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          cFinalTimings,
          undefined
        );

        if (cFinalTimings) vOnTurnComplete?.(cFinalTimings);

        vOnFlowComplete?.(cFinalTimings);

        return;
      }

      // Normalize and save assistant turn with tool calls
      const aNormalizedCalls = this.mNormalizeToolCalls(aTurnToolCalls);

      if (aNormalizedCalls.length === 0) {
        await vOnAssistantTurnComplete?.(
          vTurnContent,
          vTurnReasoningContent || undefined,
          this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings),
          undefined
        );
        vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

        return;
      }

      vTotalToolCallCount += aNormalizedCalls.length;
      this.mUpdateSession(vConversationId, { totalToolCalls: vTotalToolCallCount });

      // Save the assistant message with its tool calls
      await vOnAssistantTurnComplete?.(
        vTurnContent,
        vTurnReasoningContent || undefined,
        vTurnTimings,
        aNormalizedCalls
      );

      // Add assistant message to session history
      aSessionMessages.push({
        content: vTurnContent || undefined,
        reasoning_content: vTurnReasoningContent || undefined,
        role: MessageRole.cAssistant,
        tool_calls: aNormalizedCalls
      });

      // Execute each tool call and create result messages
      for (let vI = 0; vI < aNormalizedCalls.length; vI++) {
        const toolCall = aNormalizedCalls[vI];

        if (vSignal?.aborted) {
          vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

          return;
        }

        // Check for pending steering message - skip remaining tool calls
        if (this.vGates.mHasPendingSteeringMessage(vConversationId)) {
          console.log(
            `[AgenticStore] Steering message detected, skipping ${aNormalizedCalls.length - vI} remaining tool call(s)`
          );
          for (let vJ = vI; vJ < aNormalizedCalls.length; vJ++) {
            const cRemainingCall = aNormalizedCalls[vJ];
            const cInterruptedContent = 'Tool execution was interrupted by a new user message.';

            if (vCreateToolResultMessage) {
              await vCreateToolResultMessage(cRemainingCall.id, cInterruptedContent);
            }

            aSessionMessages.push({
              content: cInterruptedContent,
              role: MessageRole.TOOL,
              tool_call_id: cRemainingCall.id
            });
          }

          break;
        }

        const cToolName = toolCall.function.name;
        const cServerLabel = cToolsStore.mGetToolServerLabel(cToolName);
        // Ask for permission before executing the tool
        const cPermission = await this.vGates.mRequestPermission(
          vConversationId,
          cToolName,
          cServerLabel,
          vSignal
        );

        // Yield to allow Svelte to flush the UI update (hide permission dialog)
        await new Promise((pR) => setTimeout(pR, 0));

        if (vSignal?.aborted) {
          vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

          return;
        }

        const cToolStartTime = performance.now();
        const cToolSource = cToolsStore.mGetToolSource(cToolName);

        let vResult = '';
        let vToolSuccess = true;
        let vCreatedToolResultMessageId: string | null = null;

        // Streaming tools (currently only exec_shell_command): mark
        // the session so the matching renderer can switch to live mode.
        // Cleared unconditionally below.
        this.mUpdateSession(vConversationId, { executingToolCallId: toolCall.id });

        if (cPermission === ToolPermissionDecision.cDeny) {
          vResult = 'Tool execution was denied by the user.';
          vToolSuccess = false;
        } else {
          try {
            if (
              cToolSource === ToolSource.SERVER &&
              cToolName === BuiltInTool.cServerExecShellCommand &&
              vCreateToolResultMessage &&
              vUpdateToolResultMessage
            ) {
              const dArgs = this.mParseToolArguments(toolCall.function.arguments);
              const cwd = cConversationsStore.vActiveConversation?.cwd;
              const cMsg = await vCreateToolResultMessage(toolCall.id, '', undefined, cwd);

              vCreatedToolResultMessageId = cMsg.id;

              let vAccumulated = '';

              for await (const cEv of ToolsService.mStreamTool(cToolName, dArgs, vSignal, cwd)) {
                if (cEv.chunk !== null) {
                  vAccumulated += cEv.chunk;
                  await vUpdateToolResultMessage(cMsg.id, vAccumulated);
                }

                if (cEv.done) {
                  if (cEv.error) {
                    vAccumulated = vAccumulated
                      ? `${vAccumulated}\nError: ${cEv.error}`
                      : `Error: ${cEv.error}`;
                    await vUpdateToolResultMessage(cMsg.id, vAccumulated);
                    vToolSuccess = false;
                  }

                  break;
                }
              }
              vResult = vAccumulated;
            } else if (cToolSource === ToolSource.SERVER) {
              const dArgs = this.mParseToolArguments(toolCall.function.arguments);
              const cwd = cConversationsStore.vActiveConversation?.cwd;
              const cExecutionResult = await ToolsService.mExecuteTool(
                cToolName,
                dArgs,
                vSignal,
                cwd
              );

              vResult = cExecutionResult.content;

              if (cExecutionResult.isError) vToolSuccess = false;
            } else if (cToolSource === ToolSource.cBrowser) {
              const dArgs = this.mParseToolArguments(toolCall.function.arguments);

              let vExecutionResult: ToolExecutionResult;

              if (cToolName === BuiltInTool.cBrowserGetDatetime) {
                vExecutionResult = fExecuteGetDatetimeTool();
              } else if (cToolName === BuiltInTool.cServerGetInfo) {
                vExecutionResult = fExecuteBrowserInfoTool();
              } else if (cToolName === BuiltInTool.cBrowserReadMedia) {
                vExecutionResult = await ReadMediaService.mExecuteTool(
                  dArgs,
                  {
                    audio: cModelsStore.props.mModelSupportsAudio(cEffectiveModel),
                    vision: cModelsStore.props.mModelSupportsVision(cEffectiveModel)
                  },
                  vSignal,
                  cConversationsStore.vActiveConversation?.cwd
                );
              } else {
                vExecutionResult = await SandboxService.mExecuteTool(cToolName, dArgs, vSignal);
              }

              vResult = vExecutionResult.content;

              if (vExecutionResult.isError) vToolSuccess = false;
            } else {
              const cMcpCall: MCPToolCall = {
                function: { arguments: toolCall.function.arguments, name: cToolName },
                id: toolCall.id
              };
              const cExecutionResult = await cMcpStore.mExecuteTool(cMcpCall, vSignal);

              vResult = cExecutionResult.content;
            }
          } catch (error) {
            if (fIsAbortError(error)) {
              this.mUpdateSession(vConversationId, { executingToolCallId: null });
              vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

              return;
            }

            // Carry the partial stream contents already mirrored to the UI -
            // they show up as live output even if the stream broke off mid-run.
            vResult = vResult
              ? `${vResult}\nError: ${error instanceof Error ? error.message : String(error)}`
              : `Error: ${error instanceof Error ? error.message : String(error)}`;
            vToolSuccess = false;

            if (vCreatedToolResultMessageId && vUpdateToolResultMessage) {
              await vUpdateToolResultMessage(vCreatedToolResultMessageId, vResult);
            }
          }
        }

        this.mUpdateSession(vConversationId, { executingToolCallId: null });

        const cToolDurationMs = performance.now() - cToolStartTime;
        const cToolTiming: ChatMessageToolCallTiming = {
          duration_ms: Math.round(cToolDurationMs),
          name: toolCall.function.name,
          success: vToolSuccess
        };

        cAgenticTimings.toolCalls!.push(cToolTiming);
        cAgenticTimings.toolCallsCount++;
        cAgenticTimings.toolsMs += Math.round(cToolDurationMs);
        cTurnStats.toolCalls.push(cToolTiming);
        cTurnStats.toolsMs += Math.round(cToolDurationMs);

        if (vSignal?.aborted) {
          vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

          return;
        }

        const { attachments, cleanedResult: vCleanedResult } =
          this.mExtractBase64Attachments(vResult);

        // For streaming tools the result message was created empty
        // at the start of execution and updated in place as chunks
        // arrived via updateToolResultMessage. Skip the second
        // create call - just attach any base64 attachments found in
        // the final accumulator (rare, since chunks usually don't
        // carry image data URIs) and emit the attachments callback.
        let vToolResultMessage: DatabaseMessage | undefined;

        if (vCreatedToolResultMessageId) {
          vToolResultMessage = { id: vCreatedToolResultMessageId } as DatabaseMessage;

          if (attachments.length > 0 && vUpdateToolResultMessage) {
            await vUpdateToolResultMessage(
              vCreatedToolResultMessageId,
              vCleanedResult,
              attachments
            );
          }
        } else if (vCreateToolResultMessage) {
          vToolResultMessage = await vCreateToolResultMessage(
            toolCall.id,
            vCleanedResult,
            attachments.length > 0 ? attachments : undefined
          );
        }

        if (attachments.length > 0 && vToolResultMessage) {
          vOnAttachments?.(vToolResultMessage.id, attachments);
        }

        // Build content parts for session history (including images for vision models)
        const aContentParts: ApiChatMessageContentPart[] = [
          { text: vCleanedResult, type: ContentPartType.TEXT }
        ];

        for (const attachment of attachments) {
          if (attachment.type === AttachmentType.AUDIO) {
            if (cModelsStore.props.mModelSupportsAudio(cEffectiveModel)) {
              aContentParts.push({
                input_audio: {
                  data: (attachment as DatabaseMessageExtraAudioFile).base64Data,
                  format: fGetAudioInputFormat(
                    (attachment as DatabaseMessageExtraAudioFile).mimeType
                  )
                },
                type: ContentPartType.cInputAudio
              });
            }
          } else if (attachment.type === AttachmentType.IMAGE) {
            if (cModelsStore.props.mModelSupportsVision(cEffectiveModel)) {
              aContentParts.push({
                image_url: {
                  url: (attachment as DatabaseMessageExtraImageFile).base64Url
                },
                type: ContentPartType.cImageUrl
              });
            } else {
              console.info(
                `[AgenticStore] Skipping image attachment (model "${cEffectiveModel}" does not support vision)`
              );
            }
          }
        }

        aSessionMessages.push({
          content: aContentParts.length === 1 ? vCleanedResult : aContentParts,
          role: MessageRole.TOOL,
          tool_call_id: toolCall.id
        });
      }

      if (cTurnStats.toolCalls.length > 0) {
        cAgenticTimings.perTurn!.push(cTurnStats);

        const cIntermediateTimings = this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings);

        if (cIntermediateTimings) vOnTurnComplete?.(cIntermediateTimings);
      }

      // If tools were interrupted by a steering message, exit now instead of starting another LLM turn
      if (this.vGates.mHasPendingSteeringMessage(vConversationId)) {
        console.log(
          '[AgenticStore] Steering message detected after tool execution, exiting agentic flow'
        );
        vOnFlowComplete?.(this.mBuildFinalTimings(vCapturedTimings, cAgenticTimings));

        return;
      }

      vTurn++;
    }
  }

  private mExtractBase64Attachments(pResult: string): {
    cleanedResult: string;
    attachments: DatabaseMessageExtra[];
  } {
    if (!pResult.trim()) {
      return { attachments: [], cleanedResult: pResult };
    }

    const aLines = pResult.split(cNewline);
    const attachments: DatabaseMessageExtra[] = [];

    let vAttachmentIndex = 0;

    const aCleanedLines = aLines.map((pLine) => {
      const cTrimmedLine = pLine.trim();
      const cMatch = cTrimmedLine.match(cDataUriBase64Regex);

      if (!cMatch) {
        return pLine;
      }

      const cMimeType = cMatch[1].toLowerCase();
      const cBase64Data = cMatch[2];

      if (!cBase64Data) {
        return pLine;
      }

      vAttachmentIndex += 1;
      const name = this.mBuildAttachmentName(cMimeType, vAttachmentIndex);

      if (cMimeType.startsWith(cMimeTypePrefixes.IMAGE)) {
        attachments.push({ base64Url: cTrimmedLine, name, type: AttachmentType.IMAGE });

        return `[Attachment saved: ${name}]`;
      }

      if (cMimeType.startsWith(MimeTypePrefix.AUDIO)) {
        // audio extras hold the bare base64, the input_audio part has no room for a data URI
        attachments.push({
          base64Data: cBase64Data,
          mimeType: cMimeType,
          name,
          type: AttachmentType.AUDIO
        });

        return `[Attachment saved: ${name}]`;
      }

      return pLine;
    });

    return { attachments, cleanedResult: aCleanedLines.join(cNewline) };
  }

  private mNormalizeToolCalls(pToolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
    if (!pToolCalls) return [];

    return pToolCalls.map((pCall, pIndex) => ({
      function: {
        arguments: pCall?.function?.arguments ?? '',
        name: pCall?.function?.name ?? ''
      },
      id: pCall?.id ?? `tool_${pIndex}`,
      type: (pCall?.type as ToolCallType.cFunction) ?? ToolCallType.cFunction
    }));
  }

  private mParseToolArguments(pArgs: string | Record<string, unknown>): Record<string, unknown> {
    if (typeof pArgs === 'object') return pArgs;

    const cTrimmed = pArgs.trim();

    if (cTrimmed === '') return {};

    return JSON.parse(cTrimmed) as Record<string, unknown>;
  }

  private mUpdateSession(pConversationId: string, pUpdate: Partial<AgenticSession>): void {
    const session = this.mGetSession(pConversationId);

    this.vSessions.set(pConversationId, { ...session, ...pUpdate });
  }
}

export const cAgenticStore = new AgenticStore();
