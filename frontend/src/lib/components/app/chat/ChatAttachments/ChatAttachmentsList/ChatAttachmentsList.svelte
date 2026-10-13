<script lang="ts">
  import {
    ChatAttachmentsListItem,
    DialogChatAttachmentsPreview,
    DialogMcpResourcePreview,
    ScrollCarousel
  } from '$lib/components/app';
  import { ScrollCarouselVariant } from '$lib/enums';
  import type { DatabaseMessageExtraMcpResource } from '$lib/types';
  import { fGetAttachmentDisplayItems, fIsMcpPrompt, fIsMcpResource } from '$lib/utils';

  interface Props {
    class?: string;
    style?: string;
    // For ChatMessage - stored attachments
    attachments?: DatabaseMessageExtra[];
    readonly?: boolean;
    // For ChatForm - pending uploads
    onFileRemove?: (pFileId: string) => void;
    uploadedFiles?: ChatUploadedFile[];
    // Image size customization
    imageClass?: string;
    imageHeight?: string;
    imageWidth?: string;
    // Limit display to single row with "+ X more" button
    limitToSingleRow?: boolean;
    // For vision modality check
    activeModelId?: string;
  }

  let {
    activeModelId,
    attachments = [],
    class: className = '',
    // Default to small size for form previews
    imageClass = '',
    imageHeight = 'h-24',
    imageWidth = 'w-auto',
    limitToSingleRow = false,
    onFileRemove,
    readonly = false,
    style = '',
    uploadedFiles = $bindable([])
  }: Props = $props();

  let vCarouselRef: ScrollCarousel | undefined = $state();
  let vMcpResourcePreviewOpen = $state(false);
  let vMcpResourcePreviewExtra = $state<DatabaseMessageExtraMcpResource | null>(null);
  let vPreviewFocusIndex = $state(0);
  let vViewAllDialogOpen = $state(false);

  let aDisplayItems = $derived(fGetAttachmentDisplayItems({ attachments, uploadedFiles }));

  function fOpenPreview(pItem: ChatAttachmentDisplayItem, pEvent?: MouseEvent) {
    pEvent?.stopPropagation();
    pEvent?.preventDefault();

    // Find the index of the clicked item among non-MCP attachments
    const aNonMcpItems = aDisplayItems.filter((pI) => !fIsMcpPrompt(pI) && !fIsMcpResource(pI));
    const cIndex = aNonMcpItems.findIndex((pI) => pI.id === pItem.id);

    vPreviewFocusIndex = cIndex >= 0 ? cIndex : 0;
    vViewAllDialogOpen = true;
  }

  function fOpenMcpResourcePreview(pExtra: DatabaseMessageExtraMcpResource) {
    vMcpResourcePreviewExtra = pExtra;
    vMcpResourcePreviewOpen = true;
  }

  $effect(() => {
    if (vCarouselRef && aDisplayItems.length) {
      vCarouselRef.resetScroll();
    }
  });
</script>

{#snippet attachmentitem(pItem: ChatAttachmentDisplayItem)}
  <ChatAttachmentsListItem
    {imageClass}
    {imageHeight}
    {imageWidth}
    item={pItem}
    {limitToSingleRow}
    {onFileRemove}
    onMcpResourcePreview={fOpenMcpResourcePreview}
    onPreview={(pI: ChatAttachmentDisplayItem, pEvent?: MouseEvent) => fOpenPreview(pI, pEvent)}
    {readonly}
  />
{/snippet}

{#if aDisplayItems.length > 0}
  <div class={className} {style}>
    {#if limitToSingleRow}
      <ScrollCarousel bind:this={vCarouselRef} variant={ScrollCarouselVariant.CENTER}>
        {#each aDisplayItems as vItem (vItem.id)}
          {@render attachmentitem(vItem)}
        {/each}
      </ScrollCarousel>
    {:else}
      <div class="flex flex-wrap items-start justify-end gap-3">
        {#each aDisplayItems as vItem (vItem.id)}
          {@render attachmentitem(vItem)}
        {/each}
      </div>
    {/if}
  </div>
{/if}

<DialogChatAttachmentsPreview
  bind:open={vViewAllDialogOpen}
  {activeModelId}
  {attachments}
  previewFocusIndex={vPreviewFocusIndex}
  {uploadedFiles}
/>

{#if vMcpResourcePreviewExtra}
  <DialogMcpResourcePreview bind:open={vMcpResourcePreviewOpen} extra={vMcpResourcePreviewExtra} />
{/if}
