<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { AlertCircle, Download, FileText, Loader2 } from '@lucide/svelte';
  import { ActionIconCopyToClipboard } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import { cIconClassDefault } from '$lib/constants';
  import { MimeTypeApplication, MimeTypeText } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';
  import type { MCPResourceContent, MCPResourceInfo } from '$lib/types';
  import {
    fCreateBase64DataUrl,
    fDownloadResourceContent,
    fGetResourceBlobContent,
    fGetResourceTextContent,
    fIsImageMimeType
  } from '$lib/utils';

  interface Props {
    resource: MCPResourceInfo | null;
    /** Pre-loaded content (e.g., from template resolution). Skips store fetch when provided. */
    preloadedContent?: MCPResourceContent[] | null;
    class?: string;
  }

  let { class: className, preloadedContent, resource }: Props = $props();

  let vContent = $state<MCPResourceContent[] | null>(null);
  let isLoading = $state(false);
  let error = $state<string | null>(null);

  $effect(() => {
    if (resource) {
      if (preloadedContent) {
        vContent = preloadedContent;
        isLoading = false;
        error = null;
      } else {
        fLoadContent(resource.uri);
      }
    } else {
      vContent = null;
      error = null;
    }
  });

  async function fLoadContent(pUri: string) {
    isLoading = true;
    error = null;

    try {
      const cResult = await cMcpStore.mReadResource(pUri);

      if (cResult) {
        vContent = cResult;
      } else {
        error = 'Failed to load resource content';
      }
    } catch (vE) {
      error = vE instanceof Error ? vE.message : 'Unknown error';
    } finally {
      isLoading = false;
    }
  }

  function fHandleDownload() {
    const text = fGetResourceTextContent(vContent);

    if (!text || !resource) return;

    fDownloadResourceContent(
      text,
      resource.mimeType || MimeTypeText.cPlain,
      resource.name || 'resource.txt'
    );
  }
</script>

<div class={['flex flex-col gap-3', className]}>
  {#if !resource}
    <div class="flex flex-col items-center justify-center gap-2 py-8 text-muted-foreground">
      <FileText class="h-8 w-8 opacity-50" />

      <span class="text-sm">{fTranslate('text_a09ca1a10c95')}</span>
    </div>
  {:else}
    <div class="flex items-start justify-between gap-2">
      <div class="min-w-0 flex-1">
        <h3 class="truncate font-medium">{resource.title || resource.name}</h3>

        <p class="truncate text-xs text-muted-foreground">{resource.uri}</p>

        {#if resource.description}
          <p class="mt-1 text-sm text-muted-foreground">{resource.description}</p>
        {/if}
      </div>

      <div class="flex items-center gap-1">
        <ActionIconCopyToClipboard
          ariaLabel="Copy content"
          canCopy={!isLoading && !!fGetResourceTextContent(vContent)}
          text={fGetResourceTextContent(vContent)}
        />

        <Button
          class="h-7 w-7 p-0"
          disabled={isLoading || !fGetResourceTextContent(vContent)}
          onclick={fHandleDownload}
          size="sm"
          title={fTranslate('text_6eca010826ea')}
          variant="ghost"
        >
          <Download class="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>

    <div class="min-h-[200px] overflow-auto rounded-md border bg-muted/30 p-3 break-all">
      {#if isLoading}
        <div class="flex items-center justify-center py-8">
          <Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
      {:else if error}
        <div class="flex flex-col items-center justify-center gap-2 py-8 text-red-500">
          <AlertCircle class="h-6 w-6" />

          <span class="text-sm">{error}</span>
        </div>
      {:else if vContent}
        {@const cTextContent = fGetResourceTextContent(vContent)}
        {@const aBlobContent = fGetResourceBlobContent(vContent)}

        {#if cTextContent}
          <pre class="font-mono text-xs break-words whitespace-pre-wrap">{cTextContent}</pre>
        {/if}

        {#each aBlobContent as vBlob (vBlob.uri)}
          {#if fIsImageMimeType(vBlob.mimeType ?? MimeTypeApplication.cOctetStream)}
            <img
              alt={fTranslate('text_5c80d1708bf1')}
              class="max-w-full rounded"
              src={fCreateBase64DataUrl(
                vBlob.mimeType ?? MimeTypeApplication.cOctetStream,
                vBlob.blob
              )}
            />
          {:else}
            <div class="flex items-center gap-2 rounded bg-muted p-2 text-sm text-muted-foreground">
              <FileText class={cIconClassDefault} />

              <span>{fTranslate('text_78aee1d526dd')}{vBlob.mimeType || 'unknown type'})</span>
            </div>
          {/if}
        {/each}

        {#if !cTextContent && aBlobContent.length === 0}
          <div class="py-4 text-center text-sm text-muted-foreground">
            {fTranslate('text_a7c49ff5b9e2')}
          </div>
        {/if}
      {/if}
    </div>

    {#if resource.mimeType || resource.annotations}
      <div class="flex flex-wrap gap-2 text-xs text-muted-foreground">
        {#if resource.mimeType}
          <span class="rounded bg-muted px-1.5 py-0.5">{resource.mimeType}</span>
        {/if}

        {#if resource.annotations?.priority !== undefined}
          <span class="rounded bg-muted px-1.5 py-0.5">
            {fTranslate('text_34574af1e694')}
            {resource.annotations.priority}
          </span>
        {/if}

        <span class="rounded bg-muted px-1.5 py-0.5">
          {fTranslate('text_087fbe073393')}
          {resource.serverName}
        </span>
      </div>
    {/if}
  {/if}
</div>
