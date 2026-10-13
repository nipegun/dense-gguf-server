<script lang="ts">
  import {
    ChatAttachmentsPreviewCurrentItem,
    ChatAttachmentsPreviewFileInfo,
    ChatAttachmentsPreviewNavButtons,
    ChatAttachmentsPreviewThumbnailStrip
  } from '$lib/components/app';
  import { cUiDataAttrs } from '$lib/constants';
  import { cModelsStore } from '$lib/stores';
  import {
    fCreateBase64DataUrl,
    fFormatFileSize,
    fGetAttachmentDisplayItems,
    fGetLanguageFromFilename,
    fIsAudioFile,
    fIsImageFile,
    fIsMcpPrompt,
    fIsMcpResource,
    fIsPdfFile,
    fIsTextFile,
    fIsVideoFile
  } from '$lib/utils';

  interface PreviewItem {
    id: string;
    name: string;
    size?: number;
    preview?: string;
    uploadedFile?: ChatUploadedFile;
    attachment?: DatabaseMessageExtra;
    textContent?: string;
    isImage: boolean;
    isAudio: boolean;
    isVideo: boolean;
  }

  interface Props {
    uploadedFiles?: ChatUploadedFile[];
    attachments?: DatabaseMessageExtra[];
    activeModelId?: string;
    class?: string;
    previewFocusIndex?: number;
  }

  let {
    activeModelId,
    attachments = [],
    class: className = '',
    previewFocusIndex = 0,
    uploadedFiles = []
  }: Props = $props();

  let aAllItems = $derived(
    fGetAttachmentDisplayItems({ attachments, uploadedFiles })
      .filter((pItem) => !fIsMcpPrompt(pItem) && !fIsMcpResource(pItem))
      .map(
        (pItem): PreviewItem => ({
          ...pItem,
          isAudio: fIsAudioFile(pItem.attachment, pItem.uploadedFile),
          isImage: fIsImageFile(pItem.attachment, pItem.uploadedFile),
          isVideo: fIsVideoFile(pItem.attachment, pItem.uploadedFile)
        })
      )
  );

  let vCurrentIndex = $state(0);

  $effect(() => {
    if (previewFocusIndex >= 0 && previewFocusIndex < aAllItems.length) {
      vCurrentIndex = previewFocusIndex;
    }
  });

  $effect(() => {
    const fHandler = (pE: Event) => {
      const cDelta = (pE as CustomEvent).detail;

      if (cDelta < 0) {
        vCurrentIndex = vCurrentIndex > 0 ? vCurrentIndex - 1 : aAllItems.length - 1;
      } else {
        vCurrentIndex = vCurrentIndex < aAllItems.length - 1 ? vCurrentIndex + 1 : 0;
      }
    };

    document.addEventListener('chat-attachments-nav', fHandler);

    return () => document.removeEventListener('chat-attachments-nav', fHandler);
  });

  $effect(() => {
    const cIndex = vCurrentIndex;

    setTimeout(() => {
      const cThumbnail = document.querySelector(`[${cUiDataAttrs.THUMBNAIL_INDEX}="${cIndex}"]`);

      cThumbnail?.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }, 0);
  });

  let vCurrentItem = $derived(aAllItems[vCurrentIndex] ?? null);
  let vDisplayName = $derived(
    vCurrentItem?.name ||
      vCurrentItem?.uploadedFile?.name ||
      vCurrentItem?.attachment?.name ||
      'Unknown File'
  );
  let vIsAudio = $derived(
    vCurrentItem ? fIsAudioFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );
  let vIsVideo = $derived(
    vCurrentItem ? fIsVideoFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );
  let vIsImage = $derived(
    vCurrentItem ? fIsImageFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );
  let vIsPdf = $derived(
    vCurrentItem ? fIsPdfFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );
  let vIsText = $derived(
    vCurrentItem ? fIsTextFile(vCurrentItem.attachment, vCurrentItem.uploadedFile) : false
  );

  let vDisplayPreview = $derived(
    vCurrentItem?.uploadedFile?.preview ||
      (vIsImage && vCurrentItem?.attachment && 'base64Url' in vCurrentItem.attachment
        ? vCurrentItem.attachment.base64Url
        : vCurrentItem?.preview)
  );

  let vDisplayTextContent = $derived(
    vCurrentItem?.uploadedFile?.textContent ||
      (vCurrentItem?.attachment && 'content' in vCurrentItem.attachment
        ? vCurrentItem.attachment.content
        : vCurrentItem?.textContent)
  );

  let vLanguage = $derived(fGetLanguageFromFilename(vDisplayName));

  let vFileSize = $derived(vCurrentItem?.size ? fFormatFileSize(vCurrentItem.size) : '');

  let vHasVisionModality = $derived(
    vCurrentItem && activeModelId ? cModelsStore.props.mModelSupportsVision(activeModelId) : false
  );

  let vAudioSrc = $derived(
    vIsAudio && vCurrentItem
      ? (vCurrentItem.uploadedFile?.preview ??
          (vCurrentItem.attachment &&
          'mimeType' in vCurrentItem.attachment &&
          'base64Data' in vCurrentItem.attachment
            ? fCreateBase64DataUrl(
                vCurrentItem.attachment.mimeType,
                vCurrentItem.attachment.base64Data
              )
            : null))
      : null
  );

  let vVideoSrc = $derived(
    vIsVideo && vCurrentItem
      ? (vCurrentItem.uploadedFile?.preview ??
          (vCurrentItem.attachment &&
          'mimeType' in vCurrentItem.attachment &&
          'base64Data' in vCurrentItem.attachment
            ? fCreateBase64DataUrl(
                vCurrentItem.attachment.mimeType,
                vCurrentItem.attachment.base64Data
              )
            : null))
      : null
  );

  export function prev() {
    vCurrentIndex = vCurrentIndex > 0 ? vCurrentIndex - 1 : aAllItems.length - 1;
  }

  export function next() {
    vCurrentIndex = vCurrentIndex < aAllItems.length - 1 ? vCurrentIndex + 1 : 0;
  }

  function fOnNavigate(pIndex: number) {
    vCurrentIndex = pIndex;
  }
</script>

<div class="{className} flex flex-col text-white">
  <div class="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden">
    <ChatAttachmentsPreviewNavButtons onNext={next} onPrev={prev} show={aAllItems.length > 1} />

    <div class="flex h-full w-full flex-col items-center justify-start overflow-auto py-4">
      {#if vCurrentItem}
        <ChatAttachmentsPreviewFileInfo displayName={vDisplayName} fileSize={vFileSize} />

        <ChatAttachmentsPreviewCurrentItem
          {activeModelId}
          audioSrc={vAudioSrc}
          currentItem={vCurrentItem}
          displayPreview={vDisplayPreview}
          displayTextContent={vDisplayTextContent}
          hasVisionModality={vHasVisionModality}
          isAudio={vIsAudio}
          isImage={vIsImage}
          isPdf={vIsPdf}
          isText={vIsText}
          isVideo={vIsVideo}
          language={vLanguage}
          videoSrc={vVideoSrc}
        />
      {/if}

      <ChatAttachmentsPreviewThumbnailStrip
        currentIndex={vCurrentIndex}
        items={aAllItems}
        onNavigate={fOnNavigate}
      />
    </div>
  </div>
</div>
