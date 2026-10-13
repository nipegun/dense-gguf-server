<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Eye, FileText, Info } from '@lucide/svelte';
  import { SyntaxHighlightedCode } from '$lib/components/app';
  import * as Alert from '$lib/components/ui/alert';
  import { Button } from '$lib/components/ui/button';
  import { cIconClassDefault } from '$lib/constants';
  import { PdfViewMode } from '$lib/enums';
  import type { ChatAttachmentDisplayItem } from '$lib/types';
  import { fGetLanguageFromFilename } from '$lib/utils';
  import { fConvertPDFToImage } from '$lib/utils/browser-only';

  interface Props {
    currentItem: ChatAttachmentDisplayItem | null;
    displayName: string;
    displayTextContent: string | undefined;
    hasVisionModality: boolean;
    activeModelId?: string;
  }

  let { activeModelId, currentItem, displayName, displayTextContent, hasVisionModality }: Props =
    $props();

  let vPdfViewMode = $state<PdfViewMode>(PdfViewMode.cPages);
  let aPdfImages = $state<string[]>([]);
  let vPdfImagesLoading = $state(false);
  let vPdfImagesError = $state<string | null>(null);

  let vLanguage = $derived(fGetLanguageFromFilename(displayName));

  async function fLoadPdfImages() {
    if (aPdfImages.length > 0 || vPdfImagesLoading || !currentItem) return;

    vPdfImagesLoading = true;
    vPdfImagesError = null;

    try {
      let vFile: File | null = null;

      if (currentItem.uploadedFile?.file) {
        vFile = currentItem.uploadedFile.file;
      } else if (currentItem.attachment) {
        // Check if we have pre-processed images
        if (
          'images' in currentItem.attachment &&
          currentItem.attachment.images &&
          Array.isArray(currentItem.attachment.images) &&
          currentItem.attachment.images.length > 0
        ) {
          aPdfImages = currentItem.attachment.images;

          return;
        }

        // Convert base64 back to File for processing
        if ('base64Data' in currentItem.attachment && currentItem.attachment.base64Data) {
          const cBase64Data = currentItem.attachment.base64Data;
          const cByteCharacters = atob(cBase64Data);
          const aByteNumbers = new Array(cByteCharacters.length);

          for (let vI = 0; vI < cByteCharacters.length; vI++) {
            aByteNumbers[vI] = cByteCharacters.charCodeAt(vI);
          }
          const cByteArray = new Uint8Array(aByteNumbers);

          vFile = new File([cByteArray], displayName, { type: 'application/pdf' });
        }
      }

      if (vFile) {
        aPdfImages = await fConvertPDFToImage(vFile);
      } else {
        throw new Error(fTranslate('text_4291e9544b36'));
      }
    } catch (error) {
      vPdfImagesError = error instanceof Error ? error.message : 'Failed to load PDF images';
    } finally {
      vPdfImagesLoading = false;
    }
  }

  $effect(() => {
    if (vPdfViewMode === PdfViewMode.cPages) {
      fLoadPdfImages();
    }
  });
</script>

<div class="mb-4 flex items-center justify-end gap-2">
  <Button
    disabled={vPdfImagesLoading}
    onclick={() => (vPdfViewMode = PdfViewMode.TEXT)}
    size="sm"
    variant={vPdfViewMode === PdfViewMode.TEXT ? 'default' : 'outline'}
  >
    <FileText class="mr-1 {cIconClassDefault}" />
    {fTranslate('text_71988c4d8e08')}
  </Button>

  <Button
    disabled={vPdfImagesLoading}
    onclick={() => (vPdfViewMode = PdfViewMode.cPages)}
    size="sm"
    variant={vPdfViewMode === PdfViewMode.cPages ? 'default' : 'outline'}
  >
    {#if vPdfImagesLoading}
      <div
        class="mr-1 {cIconClassDefault} animate-spin rounded-full border-2 border-current border-t-transparent"
      ></div>
    {:else}
      <Eye class="mr-1 {cIconClassDefault}" />
    {/if}
    {fTranslate('text_9046da16aea9')}
  </Button>
</div>

{#if !hasVisionModality && activeModelId && currentItem}
  <Alert.Root class="mb-4 max-w-4xl">
    <Info class={cIconClassDefault} />

    <Alert.Title>{fTranslate('text_3673f9c181ba')}</Alert.Title>

    <Alert.Description>
      <span class="inline-flex">
        {fTranslate('text_7d2b2a1b79ad')}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <span
          class="mx-1 cursor-pointer underline"
          onclick={() => (vPdfViewMode = PdfViewMode.TEXT)}
        >
          {fTranslate('text_982d9e3eb996')}
        </span>
        {fTranslate('text_030cfd60beee')}
      </span>
    </Alert.Description>
  </Alert.Root>
{/if}

{#if vPdfImagesLoading}
  <div class="flex flex-1 items-center justify-center p-8">
    <div class="text-center">
      <div
        class="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-4 border-white border-t-transparent"
      ></div>

      <p class="text-white/70">{fTranslate('text_cef551a7cb0f')}</p>
    </div>
  </div>
{:else if vPdfImagesError}
  <div class="flex flex-1 items-center justify-center p-8">
    <div class="text-center">
      <FileText class="mx-auto mb-4 h-16 w-16 text-white/50" />

      <p class="mb-4 text-white/70">{fTranslate('text_e6728ce51717')}</p>

      <p class="text-sm text-white/50">{vPdfImagesError}</p>
    </div>
  </div>
{:else if aPdfImages.length > 0}
  {#each aPdfImages as vImage, vIndex (vImage)}
    <p class="mb-2 text-sm text-white/50">{fTranslate('text_0a30a815d67d')} {vIndex + 1}</p>

    <img
      alt="PDF Page {vIndex + 1}"
      class="mx-auto max-w-[85vw] rounded-lg shadow-lg"
      src={vImage}
    />

    <div class="h-4"></div>
  {/each}
{:else}
  <div class="flex flex-1 items-center justify-center p-8">
    <div class="text-center">
      <FileText class="mx-auto mb-4 h-16 w-16 text-white/50" />

      <p class="text-white/70">{fTranslate('text_35ccf4acfc40')}</p>
    </div>
  </div>
{/if}

{#if vPdfViewMode === PdfViewMode.TEXT && displayTextContent}
  <div class="px-4 pb-4">
    <SyntaxHighlightedCode
      class="max-w-4xl"
      code={displayTextContent}
      language={vLanguage}
      maxHeight="none"
    />
  </div>
{/if}
