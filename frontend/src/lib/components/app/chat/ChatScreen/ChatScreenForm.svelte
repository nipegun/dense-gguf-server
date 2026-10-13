<script lang="ts">
  import { afterNavigate } from '$app/navigation';
  import { page } from '$app/state';
  import { ChatForm } from '$lib/components/app';
  import { fUseDraftMessages } from '$lib/hooks/use-draft-messages.svelte';
  import { cDeviceStore } from '$lib/stores';
  import { onMount } from 'svelte';

  interface Props {
    class?: string;
    disabled?: boolean;
    initialMessage?: string;
    isLoading?: boolean;
    onFileRemove?: (pFileId: string) => void;
    onFileUpload?: (pFiles: File[]) => void;
    onSend?: (pMessage: string, pFiles?: ChatUploadedFile[]) => Promise<boolean>;
    onStop?: () => void;
    onSystemPromptAdd?: (pDraft: { message: string; files: ChatUploadedFile[] }) => void;
    uploadedFiles?: ChatUploadedFile[];
  }

  let {
    class: className,
    disabled = false,
    initialMessage = '',
    isLoading = false,
    onFileRemove,
    onFileUpload,
    onSend,
    onStop,
    onSystemPromptAdd,
    uploadedFiles = $bindable([])
  }: Props = $props();

  let vChatFormRef: ChatForm | undefined = $state(undefined);

  let vFormWrapperEl: HTMLDivElement | undefined = $state();
  let vChatId = $derived(page.params.id as string | undefined);

  $effect(() => {
    if (!vFormWrapperEl) return;

    const cFormEl = vFormWrapperEl.querySelector('form') as HTMLElement | null;

    if (!cFormEl) return;

    const fUpdateHeight = () => {
      const cHeight = Math.round(cFormEl.getBoundingClientRect().height);

      document.documentElement.style.setProperty('--chat-form-height', `${cHeight}px`);
    };

    fUpdateHeight();

    const cResizeObserver = new ResizeObserver(fUpdateHeight);

    cResizeObserver.observe(cFormEl);

    return () => {
      cResizeObserver.disconnect();
      document.documentElement.style.removeProperty('--chat-form-height');
    };
  });
  let vHasLoadingAttachments = $derived(uploadedFiles.some((pF) => pF.isLoading));
  let vMessage = $derived(initialMessage);
  let vPreviousIsLoading = $derived(isLoading);
  let vPreviousInitialMessage = $derived(initialMessage);

  const { clearDraft: fClearDraft } = fUseDraftMessages({
    getChatId: () => vChatId,
    getFiles: () => uploadedFiles,
    getInitialMessage: () => initialMessage,
    getMessage: () => vMessage,
    setFiles: (pF) => (uploadedFiles = pF),
    setMessage: (pM) => (vMessage = pM)
  });

  function fHandleFilesAdd(pFiles: File[]) {
    onFileUpload?.(pFiles);
  }

  async function fHandleSubmit() {
    if ((!vMessage.trim() && uploadedFiles.length === 0) || disabled || vHasLoadingAttachments)
      return;

    if (!vChatFormRef?.checkModelSelected()) return;

    const cMessageToSend = vMessage.trim();
    const aFilesToSend = [...uploadedFiles];

    vMessage = '';
    uploadedFiles = [];
    fClearDraft();

    vChatFormRef?.resetTextareaHeight();

    const success = await onSend?.(cMessageToSend, aFilesToSend);

    if (!success) {
      vMessage = cMessageToSend;
      uploadedFiles = aFilesToSend;
    }
  }

  function fHandleSystemPromptClick() {
    onSystemPromptAdd?.({ files: uploadedFiles, message: vMessage });
  }

  function fHandleUploadedFileRemove(pFileId: string) {
    onFileRemove?.(pFileId);
  }

  // Auto-focus must not steal focus already claimed elsewhere (e.g. the system
  // message editor opened just before a navigation)
  function fFocusFormUnlessCaptured() {
    const cActive = document.activeElement;

    if (cActive instanceof HTMLTextAreaElement || cActive instanceof HTMLInputElement) return;

    vChatFormRef?.focus();
  }

  onMount(() => {
    if (!cDeviceStore.isMobile) {
      setTimeout(fFocusFormUnlessCaptured, 100);
    }
  });

  afterNavigate((pNavigation) => {
    if (pNavigation?.from != null && !cDeviceStore.isMobile) {
      setTimeout(fFocusFormUnlessCaptured, 100);
    }
  });

  $effect(() => {
    if (initialMessage !== vPreviousInitialMessage) {
      vMessage = initialMessage;
      vPreviousInitialMessage = initialMessage;
    }
  });

  $effect(() => {
    if (vPreviousIsLoading && !isLoading) {
      setTimeout(fFocusFormUnlessCaptured, 10);
    }

    vPreviousIsLoading = isLoading;
  });
</script>

<div bind:this={vFormWrapperEl} class="chat-screen-form-wrapper">
  <ChatForm
    bind:this={vChatFormRef}
    bind:uploadedFiles
    bind:value={vMessage}
    class="mx-auto max-w-3xl {className}"
    {disabled}
    {isLoading}
    onFilesAdd={fHandleFilesAdd}
    {onStop}
    onSubmit={fHandleSubmit}
    onSystemPromptClick={fHandleSystemPromptClick}
    onUploadedFileRemove={fHandleUploadedFileRemove}
  />
</div>
