<script lang="ts">
  import { goto } from '$app/navigation';
  import {
    ChatMessageAssistant,
    ChatMessageMcpPrompt,
    ChatMessageSynthetic,
    ChatMessageSystem,
    ChatMessageUser
  } from '$lib/components/app/chat';
  import {
    cAgenticTextCopySeparator,
    cReasoningTags,
    cRoutes,
    cSystemMessagePlaceholder
  } from '$lib/constants';
  import { fSetChatMessageActionsContext, fSetChatMessageEditContext } from '$lib/contexts';
  import { AgenticSectionType, AttachmentType, MessageRole } from '$lib/enums';
  import { DatabaseService } from '$lib/services/database.service';
  import { cChatStore, cConversationsStore, cDeviceStore } from '$lib/stores';
  import type {
    ChatMessageActions,
    ChatMessageDeletionInfo,
    DatabaseMessageExtraMcpPrompt
  } from '$lib/types';
  import { fDeriveAgenticSections } from '$lib/utils';
  import { fParseFilesToMessageExtras } from '$lib/utils/browser-only';

  interface Props {
    class?: string;
    chatActions: ChatMessageActions;
    message: DatabaseMessage;
    toolMessages?: DatabaseMessage[];
    isLastAssistantMessage?: boolean;
    isLastUserMessage?: boolean;
    nextAssistantMessage?: DatabaseMessage | null;
    siblingInfo?: ChatMessageSiblingInfo | null;
  }

  let {
    chatActions,
    class: className = '',
    isLastAssistantMessage = false,
    isLastUserMessage = false,
    message,
    nextAssistantMessage = null,
    siblingInfo = null,
    toolMessages = []
  }: Props = $props();

  let vDeletionInfo = $state<ChatMessageDeletionInfo | null>(null);
  // The system message placeholder must never surface as editable content; keeping
  // it in the derived (not just in handleEdit) guards against prop invalidation
  // reverting the override while editing
  let vEditedContent = $derived(
    message.role === MessageRole.cSystem && message.content === cSystemMessagePlaceholder
      ? ''
      : message.content
  );

  // Synthetic cwd-change messages render with the folder-row UI instead
  // of a user bubble. The persisted flag is the single source of truth.
  let vIsSynthetic = $derived(Boolean(message.isSynthetic));

  let vRawEditContent = $derived.by(() => {
    if (message.role !== MessageRole.cAssistant) return undefined;

    const aSections = fDeriveAgenticSections(message, toolMessages, [], false);
    const aParts: string[] = [];

    for (const cSection of aSections) {
      switch (cSection.type) {
        case AgenticSectionType.REASONING:
        case AgenticSectionType.cReasoningPending:
          aParts.push(`${cReasoningTags.START}\n${cSection.content}\n${cReasoningTags.END}`);

          break;

        case AgenticSectionType.TEXT:
          aParts.push(cSection.content);

          break;

        case AgenticSectionType.TOOL_CALL:
        case AgenticSectionType.cToolCallPending:
        case AgenticSectionType.cToolCallStreaming: {
          const dCallObj: Record<string, unknown> = { name: cSection.toolName };

          if (cSection.toolArgs) {
            try {
              dCallObj.arguments = JSON.parse(cSection.toolArgs);
            } catch {
              dCallObj.arguments = cSection.toolArgs;
            }
          }

          aParts.push(JSON.stringify(dCallObj, null, 2));

          if (cSection.toolResult) {
            aParts.push(`[Tool Result]\n${cSection.toolResult}`);
          }

          break;
        }
      }
    }

    return aParts.join('\n\n\n');
  });
  let aEditedExtras = $derived<DatabaseMessageExtra[]>(message.extra ? [...message.extra] : []);
  let aEditedUploadedFiles = $state<ChatUploadedFile[]>([]);
  let vIsEditing = $state(false);
  let vShowDeleteDialog = $state(false);
  let vShouldBranchAfterEdit = $state(false);
  let vTextareaElement: HTMLTextAreaElement | undefined = $state();

  let vShowSaveOnlyOption = $derived(message.role === MessageRole.cUser);
  let vShowBranchAfterEditOption = $derived(message.role === MessageRole.cAssistant);

  fSetChatMessageEditContext({
    cancel: fHandleCancelEdit,
    get editedContent() {
      return vEditedContent;
    },
    get editedExtras() {
      return aEditedExtras;
    },
    get editedUploadedFiles() {
      return aEditedUploadedFiles;
    },
    get isEditing() {
      return vIsEditing;
    },
    get messageRole() {
      return message.role;
    },
    get originalContent() {
      return message.role === MessageRole.cAssistant
        ? (vRawEditContent ?? message.content)
        : message.content;
    },
    get originalExtras() {
      return message.extra || [];
    },
    get rawEditContent() {
      return vRawEditContent;
    },
    save: fHandleSaveEdit,
    saveOnly: fHandleSaveEditOnly,
    setContent: (pContent: string) => {
      vEditedContent = pContent;
    },
    setExtras: (pExtras: DatabaseMessageExtra[]) => {
      aEditedExtras = pExtras;
    },
    setShouldBranchAfterEdit: (pValue: boolean) => {
      vShouldBranchAfterEdit = pValue;
    },
    setUploadedFiles: (pFiles: ChatUploadedFile[]) => {
      aEditedUploadedFiles = pFiles;
    },
    get shouldBranchAfterEdit() {
      return vShouldBranchAfterEdit;
    },
    get showBranchAfterEditOption() {
      return vShowBranchAfterEditOption;
    },
    get showSaveOnlyOption() {
      return vShowSaveOnlyOption;
    },
    startEdit: fHandleEdit
  });

  fSetChatMessageActionsContext({
    confirmDelete: fHandleConfirmDelete,
    copy: fHandleCopy,
    get deletionInfo() {
      return vDeletionInfo;
    },
    get forkConversation() {
      const cIsForkableUser = message.role === MessageRole.cUser && !vMcpPromptExtra;

      return cIsForkableUser || message.role === MessageRole.cAssistant
        ? fHandleForkConversation
        : undefined;
    },
    navigateToSibling: fHandleNavigateToSibling,
    requestDelete: fHandleDelete,
    setShowDeleteDialog: fHandleShowDeleteDialogChange,
    get showDeleteDialog() {
      return vShowDeleteDialog;
    },
    get siblingInfo() {
      return siblingInfo;
    }
  });

  let vMcpPromptExtra = $derived.by(() => {
    if (message.role !== MessageRole.cUser) return null;

    if (message.content.trim()) return null;

    if (!message.extra || message.extra.length !== 1) return null;

    const cExtra = message.extra[0];

    if (cExtra.type === AttachmentType.MCP_PROMPT) {
      return cExtra as DatabaseMessageExtraMcpPrompt;
    }

    return null;
  });

  $effect(() => {
    const cPendingId = cChatStore.vPendingEditMessageId;

    if (cPendingId && cPendingId === message.id && !vIsEditing) {
      fHandleEdit();
      cChatStore.mClearPendingEditMessageId();
    }
  });

  async function fHandleCancelEdit() {
    vIsEditing = false;

    // If canceling a new system message with placeholder content, remove it without deleting children
    if (message.role === MessageRole.cSystem && message.content === cSystemMessagePlaceholder) {
      const cConversationDeleted = await cChatStore.mRemoveSystemPromptPlaceholder(message.id);

      if (cConversationDeleted) {
        goto(cRoutes.START);
      }

      return;
    }

    vEditedContent =
      message.role === MessageRole.cAssistant
        ? vRawEditContent || message.content || ''
        : message.content;
    aEditedExtras = message.extra ? [...message.extra] : [];
    aEditedUploadedFiles = [];
  }

  function fHandleCopy() {
    // Agentic sessions render as a single entry anchored on the first assistant
    // turn, whose own content is typically just the first tool call. Copy the
    // text sections of the whole session so the clipboard matches the visible
    // response instead of the anchor turn.
    if (message.role === MessageRole.cAssistant) {
      const aSections = fDeriveAgenticSections(message, toolMessages, [], false);
      const text = aSections
        .filter((pSection) => pSection.type === AgenticSectionType.TEXT)
        .map((pSection) => pSection.content)
        .join(cAgenticTextCopySeparator);

      if (text) {
        chatActions.copy(message, text);

        return;
      }
    }

    chatActions.copy(message);
  }

  async function fHandleConfirmDelete() {
    if (message.role === MessageRole.cSystem) {
      const cConversationDeleted = await cChatStore.mRemoveSystemPromptPlaceholder(message.id);

      if (cConversationDeleted) {
        goto(cRoutes.START);
      }
    } else {
      chatActions.delete(message);
    }

    vShowDeleteDialog = false;
  }

  async function fHandleDelete() {
    vDeletionInfo = await cChatStore.mGetDeletionInfo(message.id);
    vShowDeleteDialog = true;
  }

  function fHandleEdit() {
    vIsEditing = true;

    // Clear temporary placeholder content for system messages
    if (message.role === MessageRole.cSystem && message.content === cSystemMessagePlaceholder) {
      vEditedContent = '';
    } else if (message.role === MessageRole.cAssistant) {
      vEditedContent = vRawEditContent || message.content || '';
    } else {
      vEditedContent = message.content;
    }

    vTextareaElement?.focus({ preventScroll: true });
    aEditedExtras = message.extra ? [...message.extra] : [];
    aEditedUploadedFiles = [];

    setTimeout(() => {
      if (vTextareaElement) {
        vTextareaElement.focus();
        vTextareaElement.setSelectionRange(
          vTextareaElement.value.length,
          vTextareaElement.value.length
        );
      }
    }, 0);
  }

  function fHandleRegenerate(pModelOverride?: string) {
    chatActions.regenerateWithBranching(message, pModelOverride);
  }

  function fHandleContinue() {
    chatActions.continueAssistantMessage(message);
  }

  function fHandleForkConversation(pOptions: { name: string; includeAttachments: boolean }) {
    chatActions.forkConversation(message, pOptions);
  }

  function fHandleNavigateToSibling(pSiblingId: string) {
    chatActions.navigateToSibling(pSiblingId);
  }

  // After the system message flow ends, hand focus to the main chat form
  function fFocusMainChatForm() {
    if (cDeviceStore.isMobile) return;

    document.querySelector<HTMLTextAreaElement>('.chat-screen-form-wrapper textarea')?.focus();
  }

  async function fHandleSaveEdit() {
    if (message.role === MessageRole.cSystem) {
      // System messages: update in place without branching
      const cNewContent = vEditedContent.trim();

      // If content is empty, remove without deleting children
      if (!cNewContent) {
        const cConversationDeleted = await cChatStore.mRemoveSystemPromptPlaceholder(message.id);

        vIsEditing = false;

        if (cConversationDeleted) {
          goto(cRoutes.START);
        } else {
          fFocusMainChatForm();
        }

        return;
      }

      await DatabaseService.mUpdateMessage(message.id, { content: cNewContent });
      const cIndex = cConversationsStore.mFindMessageIndex(message.id);

      if (cIndex !== -1) {
        cConversationsStore.mUpdateMessageAtIndex(cIndex, { content: cNewContent });
      }

      fFocusMainChatForm();
    } else if (message.role === MessageRole.cUser) {
      const aFinalExtras = await fGetMergedExtras();

      chatActions.editWithBranching(message, vEditedContent.trim(), aFinalExtras);
    } else {
      // For assistant messages, preserve exact content including trailing whitespace
      // This is important for the Continue feature to work properly
      chatActions.editWithReplacement(message, vEditedContent, vShouldBranchAfterEdit);
    }

    vIsEditing = false;
    vShouldBranchAfterEdit = false;
    aEditedUploadedFiles = [];
  }

  async function fHandleSaveEditOnly() {
    if (message.role === MessageRole.cUser) {
      // For user messages, trim to avoid accidental whitespace
      const aFinalExtras = await fGetMergedExtras();

      chatActions.editUserMessagePreserveResponses(message, vEditedContent.trim(), aFinalExtras);
    }

    vIsEditing = false;
    aEditedUploadedFiles = [];
  }

  async function fGetMergedExtras(): Promise<DatabaseMessageExtra[]> {
    if (aEditedUploadedFiles.length === 0) {
      return aEditedExtras;
    }

    const cPlainFiles = $state.snapshot(aEditedUploadedFiles);
    const cResult = await fParseFilesToMessageExtras(cPlainFiles);
    const aNewExtras = cResult?.extras || [];

    return [...aEditedExtras, ...aNewExtras];
  }

  function fHandleShowDeleteDialogChange(pShow: boolean) {
    vShowDeleteDialog = pShow;
  }
</script>

<div>
  {#if message.role === MessageRole.cSystem}
    <ChatMessageSystem bind:textareaElement={vTextareaElement} class={className} {message} />
  {:else if vMcpPromptExtra}
    <ChatMessageMcpPrompt class={className} mcpPrompt={vMcpPromptExtra} {message} />
  {:else if vIsSynthetic}
    <ChatMessageSynthetic class={className} {message} />
  {:else if message.role === MessageRole.cUser}
    <ChatMessageUser class={className} {isLastUserMessage} {message} {nextAssistantMessage} />
  {:else}
    <ChatMessageAssistant
      bind:textareaElement={vTextareaElement}
      class={className}
      {isLastAssistantMessage}
      {message}
      onContinue={fHandleContinue}
      onRegenerate={fHandleRegenerate}
      {toolMessages}
    />
  {/if}
</div>
