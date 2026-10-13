/**
 * ChatMessageFlows - Message-level flows for the active conversation
 *
 * Owns the operations that mutate chat history and (re)stream a response:
 * editing, regeneration, continuation and deletion of messages. Created and
 * owned by chatStore; the host exposes the streaming core and the
 * per-conversation state setters these flows drive.
 */

import {
  ContinueIntentKind,
  ErrorDialogType,
  MessageRole,
  MessageType,
  StreamConnectionState
} from '$lib/enums';
import { ChatService } from '$lib/services/chat.service';
import { DatabaseService } from '$lib/services/database.service';
import type { ChatProcessingStore } from '$lib/stores/chat/processing.svelte';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cConversationsStore } from '$lib/stores/conversations/index.svelte';
import type {
  ChatMessagePromptProgress,
  ChatMessageTimings,
  DatabaseMessage,
  DatabaseMessageExtra,
  ErrorDialogState
} from '$lib/types';
import {
  fClassifyContinueIntent,
  fFilterByLeafNodeId,
  fFindDescendantMessages,
  fFindLeafNode,
  fFindMessageById,
  fIsAbortError
} from '$lib/utils';

/**
 * The slice of chatStore the flows drive. Kept narrow on purpose so the flows
 * cannot reach around the host's full surface; chatStore implements this
 * structurally.
 */
export interface ChatFlowsHost {
  processing: ChatProcessingStore;
  vStreamConnectionState: StreamConnectionState;
  mCancelPreEncode(): void;
  mClearChatStreaming(pConvId: string, pMessageId?: string): void;
  mCleanupStreaming(pConvId: string): void;
  mCreateAssistantMessage(pParentId?: string): Promise<DatabaseMessage>;
  mGetApiOptions(): Record<string, unknown>;
  mGetOrCreateAbortController(pConvId: string): AbortController;
  mIsChatLoadingInternal(pConvId: string): boolean;
  mSetChatLoading(pConvId: string, pLoading: boolean): void;
  mSetChatReasoning(pConvId: string, pReasoning: boolean): void;
  mSetChatStreaming(
    pConvId: string,
    pResponse: string,
    pMessageId: string,
    pModel?: string | null
  ): void;
  mShowErrorDialog(pState: ErrorDialogState | null): void;
  mStopGeneration(): Promise<void>;
  mStreamChatCompletion(
    pAllMessages: DatabaseMessage[],
    pAssistantMessage: DatabaseMessage,
    pOnComplete?: (pContent: string) => Promise<void>,
    pOnError?: (pError: Error) => void,
    pModelOverride?: string | null,
    pFirstUserMessageContent?: string
  ): Promise<void>;
}

export class ChatMessageFlows {
  constructor(private pHost: ChatFlowsHost) {}

  async mContinueAssistantMessage(pMessageId: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv || this.pHost.mIsChatLoadingInternal(cActiveConv.id)) return;

    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.cAssistant);

    if (!cResult) return;

    const { index: vIdx, message: vMsg } = cResult;
    // Decide which resume path applies. tool_calls without tool results can
    // not be resumed mid sequence by continue_final_message, branch instead.
    // tool_calls already paired with tool results need a fresh next turn,
    // not a token level continuation of the target assistant.
    const cIntent = fClassifyContinueIntent(cConversationsStore.aActiveMessages, vIdx);

    if (cIntent.kind === ContinueIntentKind.cRerunTurn) {
      return this.mRegenerateMessageWithBranching(pMessageId);
    }

    if (cIntent.kind === ContinueIntentKind.cNextTurn) {
      return this.mContinueAsNextAgenticTurn(cIntent.truncateAfter);
    }

    try {
      this.pHost.mShowErrorDialog(null);
      this.pHost.mSetChatLoading(cActiveConv.id, true);
      this.pHost.mClearChatStreaming(cActiveConv.id);

      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cDbMessage = fFindMessageById(aAllMessages, pMessageId);

      if (!cDbMessage) {
        this.pHost.mSetChatLoading(cActiveConv.id, false);

        return;
      }

      const cOriginalContent = cDbMessage.content;
      const cOriginalReasoning = cDbMessage.reasoningContent || '';
      // Hand the persisted DatabaseMessage straight to sendMessage so its
      // internal converter preserves tool_calls and extras when present.
      // Reconstructing a bare {role, content} here would drop those fields
      // and break continue_final_message for messages with tool calls.
      const aContextWithContinue = cConversationsStore.aActiveMessages.slice(0, vIdx + 1);

      let vAppendedContent = '';
      let vAppendedReasoning = '';
      let vHasReceivedContent = false;

      const fUpdateStreamingContent = (pFullContent: string) => {
        this.pHost.mSetChatStreaming(vMsg.convId, pFullContent, vMsg.id);
        // resolve the row by id on every write, switching to another conv mid continue makes
        // this a no op instead of writing positionally into the now displayed conversation
        cConversationsStore.mUpdateMessageAtIndex(cConversationsStore.mFindMessageIndex(vMsg.id), {
          content: pFullContent
        });
      };
      const cAbortController = this.pHost.mGetOrCreateAbortController(vMsg.convId);

      await ChatService.mSendMessage(
        aContextWithContinue,
        {
          ...this.pHost.mGetApiOptions(),
          continueFinalMessage: true,
          onChunk: (pChunk: string) => {
            vAppendedContent += pChunk;
            vHasReceivedContent = true;
            fUpdateStreamingContent(cOriginalContent + vAppendedContent);
            this.pHost.mSetChatReasoning(vMsg.convId, false);
          },
          onComplete: async (
            pFinalContent?: string,
            pReasoningContent?: string,
            pTimings?: ChatMessageTimings
          ) => {
            const cFinalAppendedContent = vHasReceivedContent
              ? vAppendedContent
              : pFinalContent || '';
            const cFinalAppendedReasoning = vHasReceivedContent
              ? vAppendedReasoning
              : pReasoningContent || '';
            const cFullContent = cOriginalContent + cFinalAppendedContent;
            const cFullReasoning = cOriginalReasoning + cFinalAppendedReasoning || undefined;

            await DatabaseService.mUpdateMessage(vMsg.id, {
              content: cFullContent,
              reasoningContent: cFullReasoning,
              timestamp: Date.now(),
              timings: pTimings
            });

            cConversationsStore.mUpdateMessageAtIndex(
              cConversationsStore.mFindMessageIndex(vMsg.id),
              {
                content: cFullContent,
                reasoningContent: cFullReasoning,
                timestamp: Date.now(),
                timings: pTimings
              }
            );

            cConversationsStore.mUpdateConversationTimestamp(vMsg.convId);

            this.pHost.mCleanupStreaming(vMsg.convId);
          },
          onCompletionId: (pId: string) => {
            if (!pId) return;

            // refresh the message id so a later skip targets the live slot after a continue
            cConversationsStore.mUpdateMessageAtIndex(
              cConversationsStore.mFindMessageIndex(vMsg.id),
              {
                completionId: pId
              }
            );
            DatabaseService.mUpdateMessage(vMsg.id, { completionId: pId }).catch(() => {});
          },
          onConnectionState: (pState: StreamConnectionState) => {
            if (vMsg.convId === cConversationsStore.vActiveConversation?.id) {
              this.pHost.vStreamConnectionState = pState;
            }
          },
          onError: async (pError: Error) => {
            if (fIsAbortError(pError)) {
              if (vHasReceivedContent && vAppendedContent) {
                await DatabaseService.mUpdateMessage(vMsg.id, {
                  content: cOriginalContent + vAppendedContent,
                  reasoningContent: cOriginalReasoning + vAppendedReasoning || undefined,
                  timestamp: Date.now()
                });

                cConversationsStore.mUpdateMessageAtIndex(
                  cConversationsStore.mFindMessageIndex(vMsg.id),
                  {
                    content: cOriginalContent + vAppendedContent,
                    reasoningContent: cOriginalReasoning + vAppendedReasoning || undefined,
                    timestamp: Date.now()
                  }
                );
              }

              this.pHost.mCleanupStreaming(vMsg.convId);

              return;
            }

            console.error('Continue generation error:', pError);
            // keep whatever was appended so far, the message stays in memory and in DB
            await DatabaseService.mUpdateMessage(vMsg.id, {
              content: cOriginalContent + vAppendedContent,
              reasoningContent: cOriginalReasoning + vAppendedReasoning || undefined,
              timestamp: Date.now()
            });
            cConversationsStore.mUpdateMessageAtIndex(
              cConversationsStore.mFindMessageIndex(vMsg.id),
              {
                content: cOriginalContent + vAppendedContent,
                reasoningContent: cOriginalReasoning + vAppendedReasoning || undefined,
                timestamp: Date.now()
              }
            );

            this.pHost.mCleanupStreaming(vMsg.convId);
            this.pHost.mShowErrorDialog({
              message: pError.message,
              type:
                pError.name === 'TimeoutError' ? ErrorDialogType.cTimeout : ErrorDialogType.SERVER
            });
          },
          onReasoningChunk: (pChunk: string) => {
            vAppendedReasoning += pChunk;
            vHasReceivedContent = true;
            // mark streaming state so a stop mid-thinking can persist the partial reasoning
            this.pHost.mSetChatStreaming(vMsg.convId, cOriginalContent + vAppendedContent, vMsg.id);
            cConversationsStore.mUpdateMessageAtIndex(
              cConversationsStore.mFindMessageIndex(vMsg.id),
              {
                reasoningContent: cOriginalReasoning + vAppendedReasoning
              }
            );
            this.pHost.mSetChatReasoning(vMsg.convId, true);
          },
          onTimings: (
            pTimings?: ChatMessageTimings,
            pPromptProgress?: ChatMessagePromptProgress
          ) => {
            this.pHost.processing.mApplyStreamTimings(pTimings, pPromptProgress, vMsg.convId);
          }
        },

        vMsg.convId,
        cAbortController.signal
      );
    } catch (error) {
      if (!fIsAbortError(error)) console.error('Failed to continue message:', error);

      if (cActiveConv) this.pHost.mSetChatLoading(cActiveConv.id, false);
    }
  }

  async mDeleteMessage(pMessageId: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) return;

    try {
      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cMessageToDelete = fFindMessageById(aAllMessages, pMessageId);

      if (!cMessageToDelete) return;

      const aCurrentPath = fFilterByLeafNodeId(aAllMessages, cActiveConv.currNode || '', false);
      const cIsInCurrentPath = aCurrentPath.some((pM) => pM.id === pMessageId);

      if (cIsInCurrentPath && cMessageToDelete.parent) {
        const aSiblings = aAllMessages.filter(
          (pM) => pM.parent === cMessageToDelete.parent && pM.id !== pMessageId
        );

        if (aSiblings.length > 0) {
          const cLatestSibling = aSiblings.reduce((pLatest, pSibling) =>
            pSibling.timestamp > pLatest.timestamp ? pSibling : pLatest
          );

          await cConversationsStore.mUpdateCurrentNode(
            fFindLeafNode(aAllMessages, cLatestSibling.id)
          );
        } else if (cMessageToDelete.parent) {
          await cConversationsStore.mUpdateCurrentNode(
            fFindLeafNode(aAllMessages, cMessageToDelete.parent)
          );
        }
      }

      await DatabaseService.mDeleteMessageCascading(cActiveConv.id, pMessageId);
      await cConversationsStore.mRefreshActiveMessages();

      cConversationsStore.mUpdateConversationTimestamp();
    } catch (error) {
      console.error('Failed to delete message:', error);
    }
  }

  async mEditAssistantMessage(
    pMessageId: string,
    pNewContent: string,
    pShouldBranch: boolean
  ): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv || this.pHost.mIsChatLoadingInternal(cActiveConv.id)) return;

    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.cAssistant);

    if (!cResult) return;

    const { index: vIdx, message: vMsg } = cResult;

    try {
      if (pShouldBranch) {
        const cNewMessage = await DatabaseService.mCreateMessageBranch(
          {
            children: [],
            content: pNewContent,
            convId: vMsg.convId,
            model: vMsg.model,
            role: vMsg.role,
            timestamp: Date.now(),
            toolCalls: vMsg.toolCalls || '',
            type: vMsg.type
          },
          vMsg.parent!
        );

        await cConversationsStore.mUpdateCurrentNode(cNewMessage.id);
      } else {
        await DatabaseService.mUpdateMessage(vMsg.id, { content: pNewContent });
        cConversationsStore.mUpdateMessageAtIndex(vIdx, { content: pNewContent });
      }

      cConversationsStore.mUpdateConversationTimestamp();

      await cConversationsStore.mRefreshActiveMessages();
    } catch (error) {
      console.error('Failed to edit assistant message:', error);
    }
  }

  async mEditMessageWithBranching(
    pMessageId: string,
    pNewContent: string,
    pNewExtras?: DatabaseMessageExtra[]
  ): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv || this.pHost.mIsChatLoadingInternal(cActiveConv.id)) return;

    let vResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.cUser);

    if (!vResult) vResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.cSystem);

    if (!vResult) return;

    const { index: vIdx, message: vMsg } = vResult;

    try {
      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cRootMessage = aAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
      const cIsFirstUserMessage =
        vMsg.role === MessageRole.cUser && cRootMessage && vMsg.parent === cRootMessage.id;
      const cExtrasToUse =
        pNewExtras !== undefined
          ? JSON.parse(JSON.stringify(pNewExtras))
          : vMsg.extra
            ? JSON.parse(JSON.stringify(vMsg.extra))
            : undefined;

      let vMessageIdForResponse: string;

      const cDbMsg = fFindMessageById(aAllMessages, vMsg.id);
      const cHasChildren = cDbMsg ? cDbMsg.children.length > 0 : vMsg.children.length > 0;

      if (!cHasChildren) {
        // No responses after this message - update in place instead of branching
        const cUpdates: Partial<DatabaseMessage> = {
          content: pNewContent,
          extra: cExtrasToUse,
          timestamp: Date.now()
        };

        await DatabaseService.mUpdateMessage(vMsg.id, cUpdates);
        cConversationsStore.mUpdateMessageAtIndex(vIdx, cUpdates);
        vMessageIdForResponse = vMsg.id;
      } else {
        // Has children - create a new branch as sibling
        const cParentId = vMsg.parent || cRootMessage?.id;

        if (!cParentId) return;

        const cNewMessage = await DatabaseService.mCreateMessageBranch(
          {
            children: [],
            content: pNewContent,
            convId: vMsg.convId,
            extra: cExtrasToUse,
            model: vMsg.model,
            role: vMsg.role,
            timestamp: Date.now(),
            toolCalls: vMsg.toolCalls || '',
            type: vMsg.type
          },
          cParentId
        );

        await cConversationsStore.mUpdateCurrentNode(cNewMessage.id);
        vMessageIdForResponse = cNewMessage.id;
      }

      cConversationsStore.mUpdateConversationTimestamp();

      if (cIsFirstUserMessage && pNewContent.trim())
        await cConversationsStore.mApplyTitleFromContent(cActiveConv.id, pNewContent);

      await cConversationsStore.mRefreshActiveMessages();

      if (vMsg.role === MessageRole.cUser)
        await this.mGenerateResponseForMessage(vMessageIdForResponse);
    } catch (error) {
      console.error('Failed to edit message with branching:', error);
    }
  }

  async mEditUserMessagePreserveResponses(
    pMessageId: string,
    pNewContent: string,
    pNewExtras?: DatabaseMessageExtra[]
  ): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) return;

    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.cUser);

    if (!cResult) return;

    const { index: vIdx, message: vMsg } = cResult;

    try {
      const cUpdateData: Partial<DatabaseMessage> = { content: pNewContent };

      if (pNewExtras !== undefined) cUpdateData.extra = JSON.parse(JSON.stringify(pNewExtras));

      await DatabaseService.mUpdateMessage(pMessageId, cUpdateData);

      cConversationsStore.mUpdateMessageAtIndex(vIdx, cUpdateData);

      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cRootMessage = aAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);

      if (cRootMessage && vMsg.parent === cRootMessage.id && pNewContent.trim()) {
        await cConversationsStore.mApplyTitleFromContent(cActiveConv.id, pNewContent);
      }

      cConversationsStore.mUpdateConversationTimestamp();
    } catch (error) {
      console.error('Failed to edit user message:', error);
    }
  }

  async mGetDeletionInfo(pMessageId: string): Promise<{
    totalCount: number;
    userMessages: number;
    assistantMessages: number;
    messageTypes: string[];
  }> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv)
      return { assistantMessages: 0, messageTypes: [], totalCount: 0, userMessages: 0 };

    const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
    const cMessageToDelete = fFindMessageById(aAllMessages, pMessageId);

    // For system messages, don't count descendants as they will be preserved (reparented to root)
    if (cMessageToDelete?.role === MessageRole.cSystem) {
      const aMessagesToDelete = aAllMessages.filter((pM) => pM.id === pMessageId);

      let vAssistantMessages = 0,
        vUserMessages = 0;

      const aMessageTypes: string[] = [];

      for (const cMsg of aMessagesToDelete) {
        if (cMsg.role === MessageRole.cUser) {
          vUserMessages++;

          if (!aMessageTypes.includes('user message')) aMessageTypes.push('user message');
        } else if (cMsg.role === MessageRole.cAssistant) {
          vAssistantMessages++;

          if (!aMessageTypes.includes('assistant response'))
            aMessageTypes.push('assistant response');
        }
      }

      return {
        assistantMessages: vAssistantMessages,
        messageTypes: aMessageTypes,
        totalCount: 1,
        userMessages: vUserMessages
      };
    }

    const aDescendants = fFindDescendantMessages(aAllMessages, pMessageId);
    const aAllToDelete = [pMessageId, ...aDescendants];
    const aMessagesToDelete = aAllMessages.filter((pM) => aAllToDelete.includes(pM.id));

    let vAssistantMessages = 0,
      vUserMessages = 0;

    const aMessageTypes: string[] = [];

    for (const cMsg of aMessagesToDelete) {
      if (cMsg.role === MessageRole.cUser) {
        vUserMessages++;

        if (!aMessageTypes.includes('user message')) aMessageTypes.push('user message');
      } else if (cMsg.role === MessageRole.cAssistant) {
        vAssistantMessages++;

        if (!aMessageTypes.includes('assistant response')) aMessageTypes.push('assistant response');
      }
    }

    return {
      assistantMessages: vAssistantMessages,
      messageTypes: aMessageTypes,
      totalCount: aAllToDelete.length,
      userMessages: vUserMessages
    };
  }

  async mRegenerateMessage(pMessageId: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv || this.pHost.mIsChatLoadingInternal(cActiveConv.id)) return;

    this.pHost.mCancelPreEncode();
    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.cAssistant);

    if (!cResult) return;

    const { index: vMessageIndex } = cResult;

    try {
      const aMessagesToRemove = cConversationsStore.aActiveMessages.slice(vMessageIndex);

      await DatabaseService.mDeleteMessageCascading(cActiveConv.id, aMessagesToRemove[0].id);
      cConversationsStore.mSliceActiveMessages(vMessageIndex);
      cConversationsStore.mUpdateConversationTimestamp();
      this.pHost.mSetChatLoading(cActiveConv.id, true);
      this.pHost.mClearChatStreaming(cActiveConv.id);
      const cParentMessageId =
        cConversationsStore.aActiveMessages.length > 0
          ? cConversationsStore.aActiveMessages[cConversationsStore.aActiveMessages.length - 1].id
          : undefined;
      const cAssistantMessage = await this.pHost.mCreateAssistantMessage(cParentMessageId);

      cConversationsStore.mAddMessageToActive(cAssistantMessage);
      await this.pHost.mStreamChatCompletion(
        cConversationsStore.aActiveMessages.slice(0, -1),
        cAssistantMessage
      );
    } catch (error) {
      if (!fIsAbortError(error)) console.error('Failed to regenerate message:', error);

      this.pHost.mSetChatLoading(cActiveConv?.id || '', false);
    }
  }

  async mRegenerateMessageWithBranching(
    pMessageId: string,
    pModelOverride?: string
  ): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv || this.pHost.mIsChatLoadingInternal(cActiveConv.id)) return;

    this.pHost.mCancelPreEncode();
    try {
      const cIdx = cConversationsStore.mFindMessageIndex(pMessageId);

      if (cIdx === -1) return;

      const cMsg = cConversationsStore.aActiveMessages[cIdx];

      if (cMsg.role !== MessageRole.cAssistant) return;

      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cParentMessage = fFindMessageById(aAllMessages, cMsg.parent);

      if (!cParentMessage) return;

      this.pHost.mSetChatLoading(cActiveConv.id, true);
      this.pHost.mClearChatStreaming(cActiveConv.id);
      const cNewAssistantMessage = await DatabaseService.mCreateMessageBranch(
        {
          children: [],
          content: '',
          convId: cMsg.convId,
          model: null,
          role: cMsg.role,
          timestamp: Date.now(),
          toolCalls: '',
          type: cMsg.type
        },
        cParentMessage.id
      );

      await cConversationsStore.mUpdateCurrentNode(cNewAssistantMessage.id);
      cConversationsStore.mUpdateConversationTimestamp();
      await cConversationsStore.mRefreshActiveMessages();
      const aConversationPath = fFilterByLeafNodeId(
        aAllMessages,
        cParentMessage.id,
        false
      ) as DatabaseMessage[];
      const cModelToUse = pModelOverride || cMsg.model || undefined;

      await this.pHost.mStreamChatCompletion(
        aConversationPath,
        cNewAssistantMessage,
        undefined,
        undefined,
        cModelToUse
      );
    } catch (error) {
      if (!fIsAbortError(error))
        console.error('Failed to regenerate message with branching:', error);

      this.pHost.mSetChatLoading(cActiveConv?.id || '', false);
    }
  }

  async mUpdateMessage(pMessageId: string, pNewContent: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) return;

    if (this.pHost.mIsChatLoadingInternal(cActiveConv.id)) await this.pHost.mStopGeneration();

    const cResult = this.mGetMessageByIdWithRole(pMessageId, MessageRole.cUser);

    if (!cResult) return;

    const { index: vMessageIndex, message: vMessageToUpdate } = cResult;
    const cOriginalContent = vMessageToUpdate.content;

    try {
      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cRootMessage = aAllMessages.find((pM) => pM.type === 'root' && pM.parent === null);
      const cIsFirstUserMessage = cRootMessage && vMessageToUpdate.parent === cRootMessage.id;

      cConversationsStore.mUpdateMessageAtIndex(vMessageIndex, { content: pNewContent });
      await DatabaseService.mUpdateMessage(pMessageId, { content: pNewContent });

      if (cIsFirstUserMessage && pNewContent.trim())
        await cConversationsStore.mApplyTitleFromContent(cActiveConv.id, pNewContent);

      const aMessagesToRemove = cConversationsStore.aActiveMessages.slice(vMessageIndex + 1);

      if (aMessagesToRemove.length > 0)
        await DatabaseService.mDeleteMessageCascading(cActiveConv.id, aMessagesToRemove[0].id);

      cConversationsStore.mSliceActiveMessages(vMessageIndex + 1);
      cConversationsStore.mUpdateConversationTimestamp();
      this.pHost.mSetChatLoading(cActiveConv.id, true);
      this.pHost.mClearChatStreaming(cActiveConv.id);
      const cAssistantMessage = await this.pHost.mCreateAssistantMessage();

      cConversationsStore.mAddMessageToActive(cAssistantMessage);
      await cConversationsStore.mUpdateCurrentNode(cAssistantMessage.id);
      await this.pHost.mStreamChatCompletion(
        cConversationsStore.aActiveMessages.slice(0, -1),
        cAssistantMessage,
        undefined,
        () => {
          cConversationsStore.mUpdateMessageAtIndex(
            cConversationsStore.mFindMessageIndex(pMessageId),
            {
              content: cOriginalContent
            }
          );
        }
      );
    } catch (error) {
      if (!fIsAbortError(error)) console.error('Failed to update message:', error);
    }
  }

  /**
   * Open a fresh assistant turn anchored at the last tool result of a resolved
   * agentic round and let streamChatCompletion route through runAgenticFlow.
   * Used by continueAssistantMessage when classifyContinueIntent returns
   * next_turn, meaning the target assistant already has its tool_calls paired
   * with trailing tool results and the next thing to generate is a brand new
   * turn rather than a token level continuation.
   */
  private async mContinueAsNextAgenticTurn(pAnchorIndex: number): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) return;

    const cAnchor = cConversationsStore.aActiveMessages[pAnchorIndex];

    if (!cAnchor) return;

    this.pHost.mCancelPreEncode();
    this.pHost.mSetChatLoading(cActiveConv.id, true);
    this.pHost.mClearChatStreaming(cActiveConv.id);
    try {
      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const cAnchorMessage = fFindMessageById(aAllMessages, cAnchor.id);

      if (!cAnchorMessage) {
        this.pHost.mSetChatLoading(cActiveConv.id, false);

        return;
      }

      const cNewAssistantMessage = await DatabaseService.mCreateMessageBranch(
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
        cAnchorMessage.id
      );

      await cConversationsStore.mUpdateCurrentNode(cNewAssistantMessage.id);
      cConversationsStore.mUpdateConversationTimestamp();
      await cConversationsStore.mRefreshActiveMessages();
      const aConversationPath = fFilterByLeafNodeId(
        aAllMessages,
        cAnchorMessage.id,
        false
      ) as DatabaseMessage[];

      await this.pHost.mStreamChatCompletion(aConversationPath, cNewAssistantMessage);
    } catch (error) {
      if (!fIsAbortError(error)) console.error('Failed to continue agentic turn:', error);

      this.pHost.mSetChatLoading(cActiveConv.id, false);
    }
  }

  private async mGenerateResponseForMessage(pUserMessageId: string): Promise<void> {
    const cActiveConv = cConversationsStore.vActiveConversation;

    if (!cActiveConv) return;

    this.pHost.mShowErrorDialog(null);
    this.pHost.mSetChatLoading(cActiveConv.id, true);
    this.pHost.mClearChatStreaming(cActiveConv.id);

    try {
      const aAllMessages = await cConversationsStore.mGetConversationMessages(cActiveConv.id);
      const aConversationPath = fFilterByLeafNodeId(
        aAllMessages,
        pUserMessageId,
        false
      ) as DatabaseMessage[];
      const cAssistantMessage = await DatabaseService.mCreateMessageBranch(
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
        pUserMessageId
      );

      cConversationsStore.mAddMessageToActive(cAssistantMessage);

      await this.pHost.mStreamChatCompletion(aConversationPath, cAssistantMessage);
    } catch (error) {
      console.error('Failed to generate response:', error);
      this.pHost.mSetChatLoading(cActiveConv.id, false);
    }
  }

  private mGetMessageByIdWithRole(
    pMessageId: string,
    pExpectedRole?: MessageRole
  ): { message: DatabaseMessage; index: number } | null {
    const cIndex = cConversationsStore.mFindMessageIndex(pMessageId);

    if (cIndex === -1) return null;

    const message = cConversationsStore.aActiveMessages[cIndex];

    if (pExpectedRole && message.role !== pExpectedRole) return null;

    return { index: cIndex, message };
  }
}
