<script lang="ts">
  import ChatScreenActionScrollDown from './ChatScreenActionScrollDown.svelte';
  import ChatScreenDialogsAndAlerts from './ChatScreenDialogsAndAlerts.svelte';
  import ChatScreenGreeting from './ChatScreenGreeting.svelte';
  import { page } from '$app/state';
  import {
    ChatMessages,
    ChatScreenDragOverlay,
    ChatScreenForm,
    ChatScreenServerError,
    ChatScreenStreamResumeStatus,
    ServerLoadingSplash
  } from '$lib/components/app';
  import { cLandingSettleMaxMs, cLandingStableFrames, cRoutes } from '$lib/constants';
  import { fCreateAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
  import { fUseChatScreenActiveModel } from '$lib/hooks/use-chat-screen-active-model.svelte';
  import { fUseChatScreenDragAndDrop } from '$lib/hooks/use-chat-screen-drag-and-drop.svelte';
  import { fUseChatScreenFileUpload } from '$lib/hooks/use-chat-screen-file-upload.svelte';
  import { fUseChatScreenScroll } from '$lib/hooks/use-chat-screen-scroll.svelte';
  import { fUseKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
  import {
    cChatStore,
    cConversationsStore,
    cDeviceStore,
    cServerStore,
    cSettingsStore
  } from '$lib/stores';
  import { fParseFilesToMessageExtras } from '$lib/utils/browser-only';
  import { onDestroy, onMount, tick } from 'svelte';

  let { showCenteredEmpty = false } = $props();

  let vDisableAutoScroll = $derived(
    Boolean(cSettingsStore.vConfig.disableAutoScroll) || cDeviceStore.isMobile
  );
  let vIsMobileUserScrolledUp = $state(false);
  let vMobileScrollDownHint = $state(false);
  let vMobileScrollDownHintLockedUntil = $state(0);
  let aEmptyFileNames = $state<string[]>([]);
  let vInitialMessage = $state('');
  let vShowDeleteDialog = $state(false);
  let vShowEmptyFileDialog = $state(false);
  let vIsEmpty = $derived(
    showCenteredEmpty && cConversationsStore.aActiveMessages.length === 0 && !cChatStore.isLoading
  );
  let vActiveErrorDialog = $derived(cChatStore.vErrorDialogState);
  let vIsServerLoading = $derived(cServerStore.loading);
  let vHasPropsError = $derived(!!cServerStore.error);
  let vIsCurrentConversationLoading = $derived(cChatStore.isLoading || cChatStore.mIsStreaming());
  let vChatFormBottomPosition = $derived.by(() => {
    if (!cDeviceStore.isMobile) return '1rem';

    if (cDeviceStore.vIsStandalone) return '1.5rem';

    if (cDeviceStore.vIsIOSSafari) return '0.25rem';

    return '0.5rem';
  });

  const cAutoScroll = fCreateAutoScrollController();
  const cScroll = fUseChatScreenScroll(cAutoScroll);
  const cActiveModel = fUseChatScreenActiveModel();
  const fileUpload = fUseChatScreenFileUpload({
    activeModelId: () => cActiveModel.activeModelId,
    capabilities: () => ({
      hasAudio: cActiveModel.hasAudioModality,
      hasVideo: cActiveModel.hasVideoModality,
      hasVision: cActiveModel.hasVisionModality
    })
  });
  const cDragAndDrop = fUseChatScreenDragAndDrop({
    onDrop: fileUpload.handleFileUpload
  });
  const { handleKeydown: fHandleKeydown } = fUseKeyboardShortcuts({
    deleteActiveConversation: () => {
      if (cConversationsStore.vActiveConversation) {
        vShowDeleteDialog = true;
      }
    }
  });

  function fHandleMobileScroll() {
    if (!cDeviceStore.isMobile) return;

    const cContainer = cScroll.chatScrollContainer;

    if (!cContainer) return;

    const cDistanceFromBottom =
      cContainer.scrollHeight - cContainer.clientHeight - cContainer.scrollTop;

    vIsMobileUserScrolledUp = cDistanceFromBottom > 300;
  }

  async function fHandleDeleteConfirm() {
    const cConversation = cConversationsStore.vActiveConversation;

    if (cConversation) {
      await cConversationsStore.mDeleteConversation(cConversation.id);
    }

    vShowDeleteDialog = false;
  }

  async function fHandleSendMessage(
    pMessage: string,
    pFiles?: ChatUploadedFile[]
  ): Promise<boolean> {
    const cPlainFiles = pFiles ? $state.snapshot(pFiles) : undefined;
    const cResult = cPlainFiles
      ? await fParseFilesToMessageExtras(cPlainFiles, cActiveModel.activeModelId ?? undefined)
      : undefined;

    if (cResult?.emptyFiles && cResult.emptyFiles.length > 0) {
      aEmptyFileNames = cResult.emptyFiles;
      vShowEmptyFileDialog = true;

      if (pFiles) {
        const sEmptyFileNamesSet = new Set(cResult.emptyFiles);

        fileUpload.uploadedFiles = fileUpload.uploadedFiles.filter(
          (pFile) => !sEmptyFileNamesSet.has(pFile.name)
        );
      }

      return false;
    }

    fHandleSendLikeScroll();

    await cChatStore.mSendMessage(pMessage, cResult?.extras);

    return true;
  }

  let vLastScrolledConversationId: string | null = null;

  // Lands at the bottom of a conversation the first time its messages
  // render, whether the route comes from another conversation or from a
  // non-conversation route. The page keeps growing after the first pin
  // without DOM mutations (content-visibility size realizations, syntax
  // highlight passes), so the instant pin repeats every frame until the
  // height settles, bailing out on user scroll or conversation change.
  async function fHandleMessagesReady(pMessageCount: number) {
    if (pMessageCount === 0) return;

    const cId = cConversationsStore.vActiveConversation?.id ?? null;

    if (!cId || cId === vLastScrolledConversationId) return;

    vLastScrolledConversationId = cId;
    await tick();
    cAutoScroll.mScrollToBottom();

    const cContainer = cScroll.chatScrollContainer;

    if (!cContainer) return;

    const cStarted = performance.now();

    let vStableFrames = 0;
    let vLastHeight = cContainer.scrollHeight;

    const fSettle = () => {
      if (cAutoScroll.userScrolledUp) return;

      if (cConversationsStore.vActiveConversation?.id !== cId) return;

      cAutoScroll.mScrollToBottom();
      const cHeight = cContainer.scrollHeight;

      vStableFrames = cHeight === vLastHeight ? vStableFrames + 1 : 0;
      vLastHeight = cHeight;

      if (vStableFrames >= cLandingStableFrames) return;

      if (performance.now() - cStarted > cLandingSettleMaxMs) return;

      requestAnimationFrame(fSettle);
    };

    requestAnimationFrame(fSettle);
  }

  function fHandleSendLikeScroll() {
    if (!cDeviceStore.isMobile) {
      cAutoScroll.mEnable();
    }

    setTimeout(() => {
      const cContainer = cScroll.chatScrollContainer;

      if (!cContainer) return;

      const cLastUserBubble = cContainer.querySelector(
        '.chat-message:nth-last-child(2) .chat-message-user .chat-message-user-bubble'
      ) as HTMLElement | null;

      if (cDeviceStore.isMobile) {
        // Keep the last user message bubble just above the input on mobile
        const cBubbleHeight = cLastUserBubble?.scrollHeight ?? 0;
        const cBaseHeight = cContainer.scrollHeight - innerHeight;

        cContainer.scrollTo({
          behavior: 'smooth',
          top: cBubbleHeight > 0 ? cBaseHeight - cBubbleHeight : cBaseHeight
        });
      } else if (cLastUserBubble) {
        // On desktop, place the last user message near the top of the viewport
        const cTopPadding = 24;
        const cBubbleRect = cLastUserBubble.getBoundingClientRect();

        cContainer.scrollTo({
          behavior: 'smooth',
          top: Math.max(0, cContainer.scrollTop + cBubbleRect.top - cTopPadding)
        });
      } else {
        cAutoScroll.mScrollToBottom();
      }
    }, 100);

    if (cDeviceStore.isMobile) {
      cAutoScroll.mSetDisabled(vDisableAutoScroll);
      vMobileScrollDownHint = true;
      vMobileScrollDownHintLockedUntil = Date.now() + 500;
    }
  }

  function fHandleErrorDialogOpenChange(pOpen: boolean) {
    if (!pOpen) {
      cChatStore.mDismissErrorDialog();
    }
  }

  async function fHandleSystemPromptAdd(pDraft: { message: string; files: ChatUploadedFile[] }) {
    if (pDraft.message || pDraft.files.length > 0) {
      cChatStore.mSavePendingDraft(pDraft.message, pDraft.files);
    }

    await cChatStore.mAddSystemPrompt();
  }

  $effect(() => {
    const cShouldDisableAutoScroll =
      cSettingsStore.vConfig.disableAutoScroll ||
      (cDeviceStore.isMobile && vIsCurrentConversationLoading);

    cAutoScroll.mSetDisabled(cShouldDisableAutoScroll);

    if (!cShouldDisableAutoScroll) {
      cAutoScroll.mEnable();
    }
  });

  onMount(() => {
    const cPendingDraft = cChatStore.mConsumePendingDraft();

    if (cPendingDraft) {
      vInitialMessage = cPendingDraft.message;
      fileUpload.uploadedFiles = cPendingDraft.files;
    }

    cAutoScroll.mStartObserving();

    if (!vDisableAutoScroll) {
      cAutoScroll.mEnable();
    }

    if (cDeviceStore.isMobile && vIsCurrentConversationLoading) {
      vMobileScrollDownHint = true;
      vMobileScrollDownHintLockedUntil = Date.now() + 500;
    }

    fHandleMobileScroll();
  });

  onDestroy(() => cAutoScroll.mDestroy());
</script>

{#if cDragAndDrop.isDragOver}
  <ChatScreenDragOverlay />
{/if}

<svelte:window
  onkeydown={fHandleKeydown}
  onscroll={(pE) => {
    cScroll.handleScroll(pE);
    fHandleMobileScroll();

    if (pE.isTrusted && Date.now() > vMobileScrollDownHintLockedUntil) {
      vMobileScrollDownHint = false;
    }
  }}
/>

{#if vIsServerLoading}
  <ServerLoadingSplash />
{:else}
  <div
    style:--chat-form-bottom-position={vChatFormBottomPosition}
    class="chat-screen flex grow flex-col min-h-[calc(100dvh-var(--status-bar-height,0px)-1rem)] md:min-h-[calc(100dvh-var(--status-bar-height,0px)-1rem-var(--chat-tabs-offset,0px))] px-4 md:py-0 pt-12 pb-48 md:pb-4"
    ondragenter={cDragAndDrop.dragHandlers.dragenter}
    ondragleave={cDragAndDrop.dragHandlers.dragleave}
    ondragover={cDragAndDrop.dragHandlers.dragover}
    ondrop={cDragAndDrop.dragHandlers.drop}
    role="main"
  >
    {#if !vIsEmpty}
      <ChatMessages
        messages={cConversationsStore.aActiveMessages}
        onMessagesReady={fHandleMessagesReady}
        onUserAction={() => {
          fHandleSendLikeScroll();
        }}
      />
    {/if}

    <div
      style:padding-top={!vIsEmpty ? 'var(--chat-form-padding-top)' : undefined}
      class={[
        // animate the centered->bottomed move with transform, not bottom:
        // layout-property transitions need the main thread every frame and
        // stutter while a long conversation loads; transform transitions
        // run on the compositor and stay smooth
        'pointer-events-none md:sticky fixed  mt-auto transition-transform duration-200',
        cDeviceStore.vIsStandalone
          ? 'bottom-[calc(1.5rem+var(--status-bar-height,0px))] right-4 left-4'
          : cDeviceStore.vIsIOSSafari
            ? 'bottom-[calc(0.25rem+var(--status-bar-height,0px))] left-2 right-2'
            : 'bottom-[calc(0.5rem+var(--status-bar-height,0px))] right-2 left-2',
        'md:bottom-[calc(1rem+var(--status-bar-height,0px))]',
        vIsEmpty ? 'md:translate-y-[calc(-50dvh+8rem)] 2xl:translate-y-[calc(-50dvh+5rem)]' : ''
      ]}
    >
      <ChatScreenGreeting isEmpty={vIsEmpty} />

      <ChatScreenServerError />

      {#if page.params.id}
        <ChatScreenStreamResumeStatus />
      {/if}

      <div class="pointer-events-none flex flex-col gap-6 items-center w-full">
        {#if (cDeviceStore.isMobile ? vMobileScrollDownHint || vIsMobileUserScrolledUp : cAutoScroll.userScrolledUp) && page.url.hash.includes(cRoutes.CHAT) && page.params.id}
          <ChatScreenActionScrollDown
            onclick={() => {
              vMobileScrollDownHint = false;
              cScroll.chatScrollContainer?.scrollTo({
                behavior: 'smooth',
                top: cScroll.chatScrollContainer.scrollHeight
              });
            }}
          />
        {/if}
      </div>

      <ChatScreenForm
        bind:uploadedFiles={fileUpload.uploadedFiles}
        class="pointer-events-auto conversation-chat-form"
        disabled={vHasPropsError || cChatStore.mIsEditing()}
        initialMessage={vInitialMessage}
        isLoading={vIsCurrentConversationLoading}
        onFileRemove={fileUpload.handleFileRemove}
        onFileUpload={fileUpload.handleFileUpload}
        onSend={fHandleSendMessage}
        onStop={() => cChatStore.mStopGeneration()}
        onSystemPromptAdd={fHandleSystemPromptAdd}
      />
    </div>
  </div>
{/if}

<ChatScreenDialogsAndAlerts
  activeErrorDialog={vActiveErrorDialog}
  emptyFileNames={aEmptyFileNames}
  {fileUpload}
  handleDeleteConfirm={fHandleDeleteConfirm}
  handleErrorDialogOpenChange={fHandleErrorDialogOpenChange}
  showDeleteDialog={vShowDeleteDialog}
  showEmptyFileDialog={vShowEmptyFileDialog}
/>
