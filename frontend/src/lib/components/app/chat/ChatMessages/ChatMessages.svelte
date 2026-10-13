<script lang="ts">
  import LazyChatMessage from './LazyChatMessage.svelte';
  import { ChatMessageUserPending } from '$lib/components/app';
  import { MessageRole } from '$lib/enums';
  import { cAgenticStore, cChatStore, cConversationsStore, cSettingsStore } from '$lib/stores';
  import type { ChatMessageActions } from '$lib/types';
  import {
    fBuildSiblingInfoMap,
    fCopyToClipboard,
    fFormatMessageForClipboard,
    fHasAgenticContent
  } from '$lib/utils';

  interface Props {
    messages?: DatabaseMessage[];
    onUserAction?: () => void;
    onMessagesReady?: (pMessageCount: number) => void;
  }

  let { messages = [], onMessagesReady, onUserAction }: Props = $props();

  let aAllConversationMessages = $state<DatabaseMessage[]>([]);

  const cCurrentConfig = cSettingsStore.vConfig;

  const cChatActions: ChatMessageActions = {
    continueAssistantMessage: async (pMessage: DatabaseMessage) => {
      onUserAction?.();
      await cChatStore.mContinueAssistantMessage(pMessage.id);
      fRefreshAllMessages();
    },

    copy: async (pMessage: DatabaseMessage, pContentOverride?: string) => {
      const cAsPlainText = Boolean(cCurrentConfig.copyTextAttachmentsAsPlainText);
      const cClipboardContent = fFormatMessageForClipboard(
        pContentOverride ?? pMessage.content,
        pMessage.extra,
        cAsPlainText
      );

      await fCopyToClipboard(cClipboardContent, 'Message copied to clipboard');
    },

    delete: async (pMessage: DatabaseMessage) => {
      await cChatStore.mDeleteMessage(pMessage.id);
      fRefreshAllMessages();
    },

    editUserMessagePreserveResponses: async (
      pMessage: DatabaseMessage,
      pNewContent: string,
      pNewExtras?: DatabaseMessageExtra[]
    ) => {
      onUserAction?.();
      // in-place edit: the store already updated activeMessages and no
      // branch is created, so sibling info stays valid without a refetch
      await cChatStore.mEditUserMessagePreserveResponses(pMessage.id, pNewContent, pNewExtras);
    },

    editWithBranching: async (
      pMessage: DatabaseMessage,
      pNewContent: string,
      pNewExtras?: DatabaseMessageExtra[]
    ) => {
      onUserAction?.();
      await cChatStore.mEditMessageWithBranching(pMessage.id, pNewContent, pNewExtras);
      fRefreshAllMessages();
    },

    editWithReplacement: async (
      pMessage: DatabaseMessage,
      pNewContent: string,
      pShouldBranch: boolean
    ) => {
      onUserAction?.();
      await cChatStore.mEditAssistantMessage(pMessage.id, pNewContent, pShouldBranch);

      // only a branch changes sibling info; an in-place edit already
      // landed in activeMessages
      if (pShouldBranch) fRefreshAllMessages();
    },

    forkConversation: async (
      pMessage: DatabaseMessage,
      pOptions: { name: string; includeAttachments: boolean }
    ) => {
      await cConversationsStore.mForkConversation(pMessage.id, pOptions);
    },

    navigateToSibling: async (pSiblingId: string) => {
      await cConversationsStore.mNavigateToSibling(pSiblingId);
    },

    regenerateWithBranching: async (pMessage: DatabaseMessage, pModelOverride?: string) => {
      onUserAction?.();
      await cChatStore.mRegenerateMessageWithBranching(pMessage.id, pModelOverride);
      fRefreshAllMessages();
    }
  };

  function fRefreshAllMessages() {
    const cConversation = cConversationsStore.vActiveConversation;

    if (cConversation) {
      // reuse the array loadConversation just read, when present; branch
      // actions fall through to a fresh fetch
      const cPreloaded = cConversationsStore.mConsumeLastLoadedMessages(cConversation.id);

      if (cPreloaded) {
        aAllConversationMessages = cPreloaded;
      } else {
        cConversationsStore.mGetConversationMessages(cConversation.id).then((pMessages) => {
          aAllConversationMessages = pMessages;
        });
      }
    } else {
      aAllConversationMessages = [];
    }
  }

  // Refresh messages whenever the active conversation changes
  $effect(() => {
    if (cConversationsStore.vActiveConversation) {
      fRefreshAllMessages();
    }
  });

  $effect(() => {
    void aAllConversationMessages;

    onMessagesReady?.(aDisplayMessages.length);
  });

  let dSiblingInfoByMessageId = $derived(fBuildSiblingInfoMap(aAllConversationMessages));

  let aDisplayMessages = $derived.by(() => {
    if (!messages.length) {
      return [];
    }

    const aFilteredMessages = cCurrentConfig.showSystemMessage
      ? messages
      : messages.filter((pMsg) => pMsg.type !== MessageRole.cSystem);
    // Build display entries, grouping agentic sessions into single entries.
    // An agentic session = assistant(with tool_calls) → tool → assistant → tool → ... → assistant(final)
    const aResult: Array<{
      message: DatabaseMessage;
      toolMessages: DatabaseMessage[];
      isLastAssistantMessage: boolean;
      isLastUserMessage: boolean;
      nextAssistantMessage: DatabaseMessage | null;
      siblingInfo: ChatMessageSiblingInfo;
    }> = [];

    for (let vI = 0; vI < aFilteredMessages.length; vI++) {
      const cMsg = aFilteredMessages[vI];

      // Skip tool messages - they're grouped with preceding assistant
      if (cMsg.role === MessageRole.TOOL) continue;

      const aToolMessages: DatabaseMessage[] = [];

      if (cMsg.role === MessageRole.cAssistant && fHasAgenticContent(cMsg)) {
        let vJ = vI + 1;

        while (vJ < aFilteredMessages.length) {
          const cNext = aFilteredMessages[vJ];

          if (cNext.role === MessageRole.TOOL) {
            aToolMessages.push(cNext);

            vJ++;
          } else if (cNext.role === MessageRole.cAssistant) {
            aToolMessages.push(cNext);

            vJ++;
          } else {
            break;
          }
        }

        vI = vJ - 1;
      } else if (cMsg.role === MessageRole.cAssistant) {
        let vJ = vI + 1;

        while (vJ < aFilteredMessages.length && aFilteredMessages[vJ].role === MessageRole.TOOL) {
          aToolMessages.push(aFilteredMessages[vJ]);
          vJ++;
        }
      }

      const cSiblingInfo = dSiblingInfoByMessageId.get(cMsg.id) ?? {
        currentIndex: 0,
        message: cMsg,
        siblingIds: [cMsg.id],
        totalSiblings: 1
      };

      aResult.push({
        isLastAssistantMessage: false,
        isLastUserMessage: false,
        message: cMsg,
        nextAssistantMessage: null,
        siblingInfo: cSiblingInfo,
        toolMessages: aToolMessages
      });
    }

    let vLastAssistantIdx = -1;

    for (let vI = aResult.length - 1; vI >= 0; vI--) {
      if (aResult[vI].message.role === MessageRole.cAssistant) {
        aResult[vI].isLastAssistantMessage = true;
        vLastAssistantIdx = vI;

        break;
      }
    }

    if (
      vLastAssistantIdx > 0 &&
      aResult[vLastAssistantIdx - 1].message.role === MessageRole.cUser
    ) {
      aResult[vLastAssistantIdx - 1].isLastUserMessage = true;
    }

    for (let vI = 0; vI < aResult.length; vI++) {
      if (aResult[vI].message.role !== MessageRole.cUser) continue;

      for (let vJ = vI + 1; vJ < aResult.length; vJ++) {
        if (aResult[vJ].message.role === MessageRole.cAssistant) {
          aResult[vI].nextAssistantMessage = aResult[vJ].message;

          break;
        }
      }
    }

    return aResult;
  });
</script>

<!-- Re-created per conversation, so the CSS fade-in below plays on every
     navigation into a chat route. -->
{#key cConversationsStore.vActiveConversation?.id ?? 'new'}
  <div class="chat-messages">
    {#each aDisplayMessages as { isLastAssistantMessage: vIsLastAssistantMessage, isLastUserMessage: vIsLastUserMessage, message: vMessage, nextAssistantMessage: vNextAssistantMessage, siblingInfo: vSiblingInfo, toolMessages: aToolMessages } (vMessage.id)}
      <LazyChatMessage
        chatActions={cChatActions}
        class="mx-auto mt-12 w-full max-w-3xl"
        isLastAssistantMessage={vIsLastAssistantMessage}
        isLastUserMessage={vIsLastUserMessage}
        message={vMessage}
        nextAssistantMessage={vNextAssistantMessage}
        siblingInfo={vSiblingInfo}
        toolMessages={aToolMessages}
      />
    {/each}

    {#if cConversationsStore.vActiveConversation && cAgenticStore.mGetPendingSteeringMessageContent(cConversationsStore.vActiveConversation!.id)}
      {@const cConvId = cConversationsStore.vActiveConversation!.id}
      {@const cPendingContent = cAgenticStore.mGetPendingSteeringMessageContent(cConvId)}

      {#if cPendingContent}
        <ChatMessageUserPending
          class="mx-auto mt-12 w-full max-w-[48rem]"
          content={cPendingContent}
          extras={cAgenticStore.mGetPendingSteeringMessageExtras(cConvId)}
          onDelete={() => cAgenticStore.mClearSteeringMessage(cConvId)}
          onEdit={(pNewContent, pExtras) =>
            cAgenticStore.mInjectSteeringMessage(cConvId, pNewContent, pExtras)}
          onSendImmediately={() => cChatStore.mAbortCurrentFlow(cConvId)}
        />
      {/if}
    {:else if cConversationsStore.vActiveConversation && cChatStore.mGetPendingMessageContent(cConversationsStore.vActiveConversation!.id)}
      {@const cConvId = cConversationsStore.vActiveConversation!.id}
      {@const cPendingContent = cChatStore.mGetPendingMessageContent(cConvId)}

      {#if cPendingContent}
        <ChatMessageUserPending
          class="mx-auto mt-12 w-full max-w-[48rem]"
          content={cPendingContent}
          extras={cChatStore.mGetPendingMessageExtras(cConvId)}
          onDelete={() => cChatStore.mClearPendingMessage(cConvId)}
          onEdit={(pNewContent, pExtras) =>
            cChatStore.mInjectPendingMessage(cConvId, pNewContent, pExtras)}
          onSendImmediately={() => cChatStore.mAbortCurrentFlow(cConvId)}
        />
      {/if}
    {/if}
  </div>
{/key}

<style>
  /* Compositor-friendly opacity fade; the keyed block re-creates the list per
	 * conversation, so the animation plays on every navigation into a chat. */
  .chat-messages {
    animation: chat-messages-fade-in 150ms ease-out;
  }

  @keyframes chat-messages-fade-in {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .chat-messages {
      animation: none;
    }
  }
</style>
