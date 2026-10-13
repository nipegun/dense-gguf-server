<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { X } from '@lucide/svelte';
  import { ChatAttachmentsPreview } from '$lib/components/app';
  import * as DialogUI from '$lib/components/ui/dialog';
  import { KeyboardKey } from '$lib/enums';
  import { Dialog } from 'bits-ui';

  interface Props {
    open: boolean;
    uploadedFiles?: ChatUploadedFile[];
    attachments?: DatabaseMessageExtra[];
    activeModelId?: string;
    previewFocusIndex?: number;
  }

  let {
    activeModelId,
    attachments = [],
    open = $bindable(false),
    previewFocusIndex = 0,
    uploadedFiles = []
  }: Props = $props();

  function fHandleClose() {
    open = false;
  }

  $effect(() => {
    if (!open) return;

    function fHandleKeyDown(pEvent: KeyboardEvent) {
      const cTarget = pEvent.target as HTMLElement;

      if (cTarget.tagName === 'INPUT' || cTarget.tagName === 'TEXTAREA') return;

      switch (pEvent.key) {
        case KeyboardKey.cArrowLeft:
          pEvent.preventDefault();
          pEvent.stopPropagation();

          document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: -1 }));

          break;
        case KeyboardKey.cArrowRight:
          pEvent.preventDefault();
          pEvent.stopPropagation();

          document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: 1 }));

          break;
        case KeyboardKey.cSpace:
          pEvent.preventDefault();
          pEvent.stopPropagation();

          document.dispatchEvent(new CustomEvent('chat-attachments-nav', { detail: 1 }));

          break;
      }
    }

    document.addEventListener('keydown', fHandleKeyDown);

    return () => document.removeEventListener('keydown', fHandleKeyDown);
  });
</script>

<Dialog.Root bind:open>
  <Dialog.Portal>
    <DialogUI.Overlay class="bg-black/85" style="z-index: 1000" />

    <Dialog.Content class="fixed inset-0 z-[1000] flex flex-col bg-transparent outline-none">
      <Dialog.Close
        aria-label={fTranslate('text_7d9eb7acb13e')}
        class="absolute top-4 right-4 z-10 cursor-pointer text-white hover:text-gray-400"
        onclick={fHandleClose}
      >
        <X class="size-4" />
      </Dialog.Close>

      <ChatAttachmentsPreview
        {activeModelId}
        {attachments}
        class="min-h-0 flex-1"
        {previewFocusIndex}
        {uploadedFiles}
      />
    </Dialog.Content>
  </Dialog.Portal>
</Dialog.Root>
