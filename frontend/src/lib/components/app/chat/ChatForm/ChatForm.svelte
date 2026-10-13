<script lang="ts">
  import ContextGaugePopup from './ChatFormContextGauge/ContextGaugePopup.svelte';
  import {
    ChatAttachmentsList,
    ChatFormActions,
    ChatFormCurrentWorkingDirectory,
    ChatFormInput,
    ChatFormInputFileInputInvisible,
    ChatFormMcpResourcesList,
    ChatFormPickers,
    DialogMcpResourcesBrowser
  } from '$lib/components/app';
  import {
    cClipboardContentQuotePrefix,
    cInitialFileSize,
    cInputClasses,
    cPromptContentSeparator,
    dSettingConfigDefault
  } from '$lib/constants';
  import {
    ContentPartType,
    FileExtensionText,
    KeyboardKey,
    MimeTypeText,
    SpecialFileType,
    ToolSource
  } from '$lib/enums';
  import { fUseChatFormPickers } from '$lib/hooks/use-chat-form-pickers.svelte';
  import {
    cChatStore,
    cConversationsStore,
    cMcpStore,
    cModelsStore,
    cServerStore,
    cSettingsStore,
    cToolsStore,
    cUiStore
  } from '$lib/stores';
  import type {
    FileMentionEntry,
    GetPromptResult,
    MCPPromptInfo,
    MCPResourceInfo,
    PromptMessage
  } from '$lib/types';
  import {
    fBuildMentionInsertion,
    fContainsCodeSpan,
    fContainsFileMentionLink,
    fFindCommandToken,
    fFindMentionToken,
    fGetConversationModel,
    fIsIMEComposing,
    fIsOffsetInCodeBlock,
    fParseClipboardContent,
    fUuid
  } from '$lib/utils';
  import {
    AudioRecorder,
    fConvertToWav,
    fCreateAudioFile,
    fIsAudioRecordingSupported
  } from '$lib/utils/browser-only';
  import { onMount } from 'svelte';

  interface Props {
    // Data
    attachments?: DatabaseMessageExtra[];
    uploadedFiles?: ChatUploadedFile[];
    value?: string;

    // UI State
    class?: string;
    disabled?: boolean;
    isLoading?: boolean;
    placeholder?: string;
    showAddButton?: boolean;
    showModelSelector?: boolean;

    // Event Handlers
    onAttachmentRemove?: (pIndex: number) => void;
    onFilesAdd?: (pFiles: File[]) => void;
    onStop?: () => void;
    onSubmit?: () => void;
    onSystemPromptClick?: (pDraft: { message: string; files: ChatUploadedFile[] }) => void;
    onUploadedFileRemove?: (pFileId: string) => void;
    onUploadedFilesChange?: (pFiles: ChatUploadedFile[]) => void;
    onValueChange?: (pValue: string) => void;
  }

  let {
    attachments = [],
    class: className = '',
    disabled = false,
    isLoading = false,
    onAttachmentRemove,
    onFilesAdd,
    onStop,
    onSubmit,
    onSystemPromptClick,
    onUploadedFileRemove,
    onUploadedFilesChange,
    onValueChange,
    placeholder = 'Type a message...',
    showAddButton = true,
    showModelSelector = true,
    uploadedFiles = $bindable([]),
    value = $bindable('')
  }: Props = $props();

  // Component References
  // Shared handle of the two input renderers (plain textarea + rich chat form input).
  type ChatInputHandle = {
    focus(): void;
    resetHeight(): void;
    getElement(): HTMLElement | undefined;
    getCaretOffset(): number;
    setCaretOffset(pOffset: number): void;
  };

  let vAudioRecorder: AudioRecorder | undefined;
  let vChatFormActionsRef: ChatFormActions | undefined = $state(undefined);
  let vFileInputRef: ChatFormInputFileInputInvisible | undefined = $state(undefined);
  let vPickersRef: { handleKeydown: (pEvent: KeyboardEvent) => boolean } | undefined =
    $state(undefined);
  let vInputRef: ChatInputHandle | undefined = $state(undefined);

  // Render-mode gate: the plain textarea by default, the rich chat form input
  // while the buffer carries a `file://` mention link or a complete code
  // span (badges and code chips need a DOM the textarea cannot provide).
  // Demotes back once neither remains.
  let vUseRichInput = $state(false);

  // Audio Recording State
  let vIsRecording = $state(false);
  let vRecordingSupported = $state(false);

  // Invisible anchor at the form's top edge so the mention/WD popovers
  // float above the box.
  let vMentionAnchor: HTMLDivElement | null = $state(null);

  let vCwd = $derived(
    cConversationsStore.vActiveConversation?.cwd ?? cConversationsStore.preferences.vPendingCwd
  );

  const cPickers = fUseChatFormPickers({
    focusInput: fRefocusInput,
    getCaretOffset: () => vInputRef?.getCaretOffset(),
    getCwd: () => vCwd,
    getPickersRef: () => vPickersRef,
    getServerHome: () => cToolsStore.serverHome ?? null,
    getShowModelSelector: () => showModelSelector,
    getValue: () => value,
    hasCwdTools: () => cConversationsStore.preferences.mHasEnabledCwdTools(),
    // policy-aware, same rule as the agentic flow: MCP category on and at
    // least one globally-enabled server whose group key is not disabled
    hasPrompts: () => {
      const cPrefs = cConversationsStore.preferences;

      if (!cPrefs.mIsCategoryEnabled(ToolSource.cMcp)) return false;

      return cMcpStore
        .mGetServers()
        .some((pS) => pS.enabled && cPrefs.mIsServerToolsEnabled(pS.id) && pS.url.trim());
    },
    openModelSelector: () => vChatFormActionsRef?.openModelSelector(),
    setCaretOffset: (pOffset) => vInputRef?.setCaretOffset(pOffset),
    setValue: (pV) => {
      value = pV;
      onValueChange?.(pV);
    }
  });

  async function fHandleWorkingDirectoryChange(pNewDir: string | null) {
    // Committing a directory consumes the `/cwd` token; the chip's
    // clear-X path has no token to consume.
    const cToken = fFindCommandToken(value);

    if (cToken && cToken.name === 'cwd') {
      value = '';
      onValueChange?.('');
    }

    await cConversationsStore.preferences.mSetCwd(pNewDir);

    if (cConversationsStore.vActiveConversation) {
      await cChatStore.mRecordCwdChange(pNewDir?.trim() || null);
    }
  }

  // Resource Dialog State
  let vIsResourceDialogOpen = $state(false);
  let vPreSelectedResourceUri = $state<string | undefined>(undefined);

  // MCP Servers Dialog State

  let vCurrentConfig = $derived(cSettingsStore.vConfig);

  let vPasteLongTextToFileLength = $derived.by(() => {
    const cN = Number(vCurrentConfig.pasteLongTextToFileLen);

    return Number.isNaN(cN) ? Number(dSettingConfigDefault.pasteLongTextToFileLen) : cN;
  });

  let vIsRouter = $derived(cServerStore.isRouterMode);
  let vConversationModel = $derived(
    fGetConversationModel(cConversationsStore.aActiveMessages as DatabaseMessage[])
  );
  let vActiveModelId = $derived(cModelsStore.activeModelId);

  let vHasModelSelected = $derived(
    !vIsRouter || !!vConversationModel || !!cModelsStore.vSelectedModelId
  );
  let vHasLoadingAttachments = $derived(uploadedFiles.some((pF) => pF.isLoading));
  let vHasAttachments = $derived(
    (attachments && attachments.length > 0) || (uploadedFiles && uploadedFiles.length > 0)
  );
  let vCanSubmit = $derived(value.trim().length > 0 || vHasAttachments);

  // Caret offset restored after a renderer swap. Callers that mutate
  // `value` themselves (e.g. the mention picker) pin the target offset
  // BEFORE the assignment; otherwise the swap effect snapshots the
  // current caret.
  let vPendingCaretOffset = 0;
  let vCaretOffsetPinned = false;

  function fQueueCaretRestore() {
    queueMicrotask(() => {
      vInputRef?.focus();
      vInputRef?.setCaretOffset(vPendingCaretOffset);
      vCaretOffsetPinned = false;
    });
  }

  $effect(() => {
    const cWantRichInput = fContainsFileMentionLink(value ?? '') || fContainsCodeSpan(value ?? '');

    if (vUseRichInput === cWantRichInput) return;

    if (!vCaretOffsetPinned) {
      vPendingCaretOffset = vInputRef?.getCaretOffset() ?? (value ?? '').length;
    }

    vUseRichInput = cWantRichInput;
    fQueueCaretRestore();
  });

  onMount(() => {
    vRecordingSupported = fIsAudioRecordingSupported();
    vAudioRecorder = new AudioRecorder();
  });

  export function focus() {
    vInputRef?.focus();
  }

  export function resetTextareaHeight() {
    vInputRef?.resetHeight();
  }

  export function openModelSelector() {
    vChatFormActionsRef?.openModelSelector();
  }

  export function checkModelSelected(): boolean {
    if (!vHasModelSelected) {
      vChatFormActionsRef?.openModelSelector();

      return false;
    }

    return true;
  }

  function fHandleFileSelect(pFiles: File[]) {
    onFilesAdd?.(pFiles);
  }

  function fHandleFileUpload() {
    vFileInputRef?.click();
  }

  function fHandleFileRemove(pFileId: string) {
    if (pFileId.startsWith('attachment-')) {
      const cIndex = parseInt(pFileId.replace('attachment-', ''), 10);

      if (!isNaN(cIndex) && cIndex >= 0 && cIndex < attachments.length) {
        onAttachmentRemove?.(cIndex);
      }
    } else {
      onUploadedFileRemove?.(pFileId);
    }
  }

  function fHandleKeydown(pEvent: KeyboardEvent) {
    // Pickers consume navigation/escape keys first; when consumed, skip
    // the enter-to-submit logic below.
    if (cPickers.handleKeydown(pEvent)) {
      return;
    }

    if (pEvent.key === KeyboardKey.cEnter && !pEvent.shiftKey && !fIsIMEComposing(pEvent)) {
      const cIsModifier = pEvent.ctrlKey || pEvent.metaKey;
      const cSendOnEnter = vCurrentConfig.sendOnEnter !== false;

      // Caret inside a fenced code block (closed, or still open
      // while being typed): Enter adds a line, never submits. The
      // rich chat form input consumes this case locally; this gate
      // covers the plain textarea, where skipping submit lets the
      // native newline through.
      if (!cIsModifier && fIsOffsetInCodeBlock(value ?? '', vInputRef?.getCaretOffset() ?? 0)) {
        return;
      }

      if (cSendOnEnter || cIsModifier) {
        pEvent.preventDefault();

        if (!vCanSubmit || disabled || vHasLoadingAttachments) return;

        onSubmit?.();
      }
    }
  }

  function fHandlePaste(pEvent: ClipboardEvent) {
    if (!pEvent.clipboardData) return;

    const aFiles = Array.from(pEvent.clipboardData.items)
      .filter((pItem) => pItem.kind === 'file')
      .map((pItem) => pItem.getAsFile())
      .filter((pFile): pFile is File => pFile !== null);

    if (aFiles.length > 0) {
      pEvent.preventDefault();
      onFilesAdd?.(aFiles);

      return;
    }

    const text = pEvent.clipboardData.getData(MimeTypeText.cPlain);

    if (text.startsWith(cClipboardContentQuotePrefix)) {
      const cParsed = fParseClipboardContent(text);

      if (cParsed.textAttachments.length > 0 || cParsed.mcpPromptAttachments.length > 0) {
        pEvent.preventDefault();
        value = cParsed.message;
        onValueChange?.(cParsed.message);

        // Handle text attachments as files
        if (cParsed.textAttachments.length > 0) {
          const aAttachmentFiles = cParsed.textAttachments.map(
            (pAtt) =>
              new File([pAtt.content], pAtt.name, {
                type: MimeTypeText.cPlain
              })
          );

          onFilesAdd?.(aAttachmentFiles);
        }

        // Handle MCP prompt attachments as ChatUploadedFile with mcpPrompt data
        if (cParsed.mcpPromptAttachments.length > 0) {
          const aMcpPromptFiles: ChatUploadedFile[] = cParsed.mcpPromptAttachments.map((pAtt) => ({
            file: new File([pAtt.content], `${pAtt.name}${FileExtensionText.cTxt}`, {
              type: MimeTypeText.cPlain
            }),
            id: fUuid(),
            isLoading: false,
            mcpPrompt: {
              arguments: pAtt.arguments,
              promptName: pAtt.promptName,
              serverName: pAtt.serverName
            },
            name: pAtt.name,
            size: pAtt.content.length,
            textContent: pAtt.content,
            type: SpecialFileType.MCP_PROMPT
          }));

          uploadedFiles = [...uploadedFiles, ...aMcpPromptFiles];
          onUploadedFilesChange?.(uploadedFiles);
        }

        setTimeout(() => {
          vInputRef?.focus();
        }, 10);

        return;
      }
    }

    if (
      text.length > 0 &&
      vPasteLongTextToFileLength > 0 &&
      text.length > vPasteLongTextToFileLength
    ) {
      pEvent.preventDefault();

      const cTextFile = new File([text], 'Pasted', {
        type: MimeTypeText.cPlain
      });

      onFilesAdd?.([cTextFile]);
    }
  }

  function fHandlePromptLoadStart(
    pPlaceholderId: string,
    pPromptInfo: MCPPromptInfo,
    pArgs?: Record<string, string>
  ) {
    cPickers.mClosePromptPicker();

    const cPromptName = pPromptInfo.title || pPromptInfo.name;
    const cPlaceholder: ChatUploadedFile = {
      file: new File([], 'loading'),
      id: pPlaceholderId,
      isLoading: true,
      mcpPrompt: {
        arguments: pArgs ? { ...pArgs } : undefined,
        promptName: pPromptInfo.name,
        serverName: pPromptInfo.serverName
      },
      name: cPromptName,
      size: cInitialFileSize,
      type: SpecialFileType.MCP_PROMPT
    };

    uploadedFiles = [...uploadedFiles, cPlaceholder];
    onUploadedFilesChange?.(uploadedFiles);
    vInputRef?.focus();
  }

  function fHandlePromptLoadComplete(pPlaceholderId: string, pResult: GetPromptResult) {
    const cPromptText = pResult.messages
      ?.map((pMsg: PromptMessage) => {
        if (typeof pMsg.content === 'string') {
          return pMsg.content;
        }

        if (pMsg.content.type === ContentPartType.TEXT) {
          return pMsg.content.text;
        }

        return '';
      })
      .filter(Boolean)
      .join(cPromptContentSeparator);

    uploadedFiles = uploadedFiles.map((pF) =>
      pF.id === pPlaceholderId
        ? {
            ...pF,
            file: new File([cPromptText], `${pF.name}${FileExtensionText.cTxt}`, {
              type: MimeTypeText.cPlain
            }),
            isLoading: false,
            size: cPromptText.length,
            textContent: cPromptText
          }
        : pF
    );
    onUploadedFilesChange?.(uploadedFiles);
  }

  function fHandlePromptLoadError(pPlaceholderId: string, pError: string) {
    uploadedFiles = uploadedFiles.map((pF) =>
      pF.id === pPlaceholderId ? { ...pF, isLoading: false, loadError: pError } : pF
    );
    onUploadedFilesChange?.(uploadedFiles);
  }

  // Deferred so the closing popover's focus scope tears down first -
  // bits-ui yanks a synchronous focus() back into the still-mounted popover.
  function fRefocusInput() {
    queueMicrotask(() => vInputRef?.focus());
  }

  // A flow outside the chat, e.g. the models manager closing onto it, asks for the
  // composer back. The dialog's focus scope is still tearing down, so focus on a
  // frame of its own.
  $effect(() => {
    if (cUiStore.vManageModelsOpen) return;

    if (!cUiStore.vComposerFocusRequested) return;

    cUiStore.vComposerFocusRequested = false;

    setTimeout(() => vInputRef?.focus(), 0);
  });

  // Splice the mention link in place of the `@<query>` token. Uses the
  // live cursor, not a stale snapshot - the token may have been edited.
  function fHandleMentionSelect(pEntry: FileMentionEntry) {
    const cCursor = vInputRef?.getCaretOffset() ?? value.length;
    const cToken = fFindMentionToken(value, cCursor);

    if (!cToken) return;

    const cBuilt = fBuildMentionInsertion(pEntry, value, cToken);

    if (!cBuilt) return;

    // Pin the post-insertion caret BEFORE the swap effect runs;
    // otherwise the effect clobbers it with the textarea's selection
    // at promotion time (browser-dependent: usually reset to 0).
    vPendingCaretOffset = cBuilt.caretOffset;
    vCaretOffsetPinned = true;

    value = cBuilt.newValue;
    onValueChange?.(cBuilt.newValue);

    // Already in rich chat form input mode: no renderer flip, so the swap
    // effect's caret restore never runs.
    if (vUseRichInput) {
      fQueueCaretRestore();
    }
  }

  async function fHandleMicClick() {
    if (!vAudioRecorder || !vRecordingSupported) {
      console.warn('Audio recording not supported');

      return;
    }

    if (vIsRecording) {
      vIsRecording = false;
      try {
        const cAudioBlob = await vAudioRecorder.mStopRecording();
        const cWavBlob = await fConvertToWav(cAudioBlob);
        const cAudioFile = fCreateAudioFile(cWavBlob);

        onFilesAdd?.([cAudioFile]);
      } catch (error) {
        console.error('Failed to stop recording:', error);
      }
    } else {
      try {
        await vAudioRecorder.mStartRecording();
        vIsRecording = true;
      } catch (error) {
        console.error('Failed to start recording:', error);
      }
    }
  }
</script>

<ChatFormInputFileInputInvisible bind:this={vFileInputRef} onFileSelect={fHandleFileSelect} />

<form
  class="relative grid {className}"
  onsubmit={(pEvent) => {
    pEvent.preventDefault();

    if (!vCanSubmit || disabled || vHasLoadingAttachments) return;

    onSubmit?.();
  }}
>
  <ChatFormPickers
    bind:this={vPickersRef}
    commandQuery={cPickers.commandQuery}
    commands={cPickers.availableCommands}
    isCommandPickerOpen={cPickers.isCommandPickerOpen}
    isMentionPickerOpen={cPickers.isMentionPickerOpen}
    isPromptPickerOpen={cPickers.isPromptPickerOpen}
    mentionAnchor={vMentionAnchor}
    mentionQuery={cPickers.mentionQuery}
    onCommandPickerClose={cPickers.handleCommandPickerClose}
    onCommandSelect={cPickers.handleCommandSelect}
    onMentionOpened={() => vInputRef?.focus()}
    onMentionPickerClose={cPickers.handleMentionPickerClose}
    onMentionSelect={fHandleMentionSelect}
    onPromptLoadComplete={fHandlePromptLoadComplete}
    onPromptLoadError={fHandlePromptLoadError}
    onPromptLoadStart={fHandlePromptLoadStart}
    onPromptPickerClose={cPickers.handlePromptPickerClose}
    promptSearchQuery={cPickers.promptSearchQuery}
    scopePath={cPickers.mentionScopePath}
  />

  <div
    bind:this={vMentionAnchor}
    aria-hidden="true"
    class="pointer-events-none absolute top-0 right-0 left-0 h-px"
  ></div>

  <div
    class="{cInputClasses} overflow-hidden rounded-4xl md:rounded-3xl backdrop-blur-md {disabled
      ? 'cursor-not-allowed opacity-60'
      : ''}"
    data-slot="input-area"
  >
    <ChatAttachmentsList
      bind:uploadedFiles
      activeModelId={vActiveModelId ?? undefined}
      {attachments}
      class="py-5"
      limitToSingleRow
      onFileRemove={fHandleFileRemove}
      style="scroll-padding: 1rem;"
    />

    <div
      class="flex-column relative min-h-12 items-center rounded-4xl md:rounded-3xl py-2 pb-2.25 shadow-sm transition-all focus-within:shadow-md md:py-3!"
    >
      <ChatFormInput
        bind:this={vInputRef}
        bind:value
        class="px-5 py-1.5 md:pt-0"
        {disabled}
        onInput={() => {
          cPickers.handleInput();
          onValueChange?.(value);
        }}
        onKeydown={fHandleKeydown}
        onPaste={fHandlePaste}
        {placeholder}
        useRichInput={vUseRichInput}
      />

      {#if cMcpStore.resources.hasAttachments}
        <ChatFormMcpResourcesList
          class="mb-3"
          onResourceClick={(pUri) => {
            vPreSelectedResourceUri = pUri;
            vIsResourceDialogOpen = true;
          }}
        />
      {/if}

      <ChatFormActions
        bind:this={vChatFormActionsRef}
        canSend={vCanSubmit}
        class="px-3"
        {disabled}
        {isLoading}
        isReasoning={cChatStore.vIsReasoning}
        isRecording={vIsRecording}
        onFileUpload={fHandleFileUpload}
        onMicClick={fHandleMicClick}
        {onStop}
        onSystemPromptClick={() => onSystemPromptClick?.({ files: uploadedFiles, message: value })}
        {showAddButton}
        {showModelSelector}
        {uploadedFiles}
      />
    </div>
  </div>

  <ContextGaugePopup />

  {#if cConversationsStore.preferences.mHasEnabledCwdTools()}
    <ChatFormCurrentWorkingDirectory
      bind:query={cPickers.workingDirectoryQuery}
      customAnchor={vMentionAnchor}
      directory={vCwd}
      {disabled}
      isOpen={cPickers.isWorkingDirectoryPickerOpen}
      onChange={fHandleWorkingDirectoryChange}
      onClose={cPickers.handleWorkingDirectoryClose}
      onOpen={cPickers.handleWorkingDirectoryOpen}
    />
  {/if}
</form>

<DialogMcpResourcesBrowser
  bind:open={vIsResourceDialogOpen}
  onAttach={(pResource: MCPResourceInfo) => {
    cMcpStore.mAttachResource(pResource.uri);
  }}
  onOpenChange={(pNewOpen: boolean) => {
    if (!pNewOpen) {
      vPreSelectedResourceUri = undefined;
    }
  }}
  preSelectedUri={vPreSelectedResourceUri}
/>
