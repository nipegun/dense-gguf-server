<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Download } from '@lucide/svelte';
  import { ActionIconCopyToClipboard, SyntaxHighlightedCode } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import * as Dialog from '$lib/components/ui/dialog';
  import { cDefaultResourceFilename, cMimeTypeSubstrings } from '$lib/constants';
  import { MimeTypeText } from '$lib/enums';
  import { cDeviceStore, cMcpStore } from '$lib/stores';
  import type { DatabaseMessageExtraMcpResource } from '$lib/types';
  import {
    fDownloadResourceContent,
    fGetLanguageFromFilename,
    fIsCodeResource,
    fIsImageResource
  } from '$lib/utils';

  interface Props {
    open: boolean;
    onOpenChange?: (pOpen: boolean) => void;
    extra: DatabaseMessageExtraMcpResource;
  }

  let { extra, onOpenChange, open = $bindable() }: Props = $props();

  const cServerName = $derived(cMcpStore.mGetServerDisplayName(extra.serverName));
  const cFavicon = $derived(cMcpStore.mGetServerFavicon(extra.serverName));

  function fGetLanguage(): string {
    if (extra.mimeType?.includes(cMimeTypeSubstrings.JSON)) return cMimeTypeSubstrings.JSON;

    if (extra.mimeType?.includes(cMimeTypeSubstrings.JAVASCRIPT))
      return cMimeTypeSubstrings.JAVASCRIPT;

    if (extra.mimeType?.includes(cMimeTypeSubstrings.TYPESCRIPT))
      return cMimeTypeSubstrings.TYPESCRIPT;

    const name = extra.name || extra.uri || '';

    return fGetLanguageFromFilename(name) || 'plaintext';
  }

  function fHandleDownload() {
    if (!extra.content) return;

    fDownloadResourceContent(
      extra.content,
      extra.mimeType || MimeTypeText.cPlain,
      extra.name || cDefaultResourceFilename
    );
  }
</script>

<Dialog.fRoot bind:open {onOpenChange}>
  <Dialog.Content
    class="grid max-h-[90vh] max-w-5xl overflow-hidden p-4 sm:w-auto sm:max-w-6xl"
    onOpenAutoFocus={(pEvent) => cDeviceStore.isMobile && pEvent.preventDefault()}
  >
    <Dialog.Header>
      <Dialog.Title class="pr-8">{extra.name}</Dialog.Title>

      <Dialog.Description>
        <div class="flex items-center gap-2">
          <span class="text-xs text-muted-foreground">{extra.uri}</span>

          {#if cServerName}
            <span class="flex items-center gap-1 text-xs text-muted-foreground">
              ·
              {#if cFavicon}
                <img
                  alt=""
                  class="h-3 w-3 shrink-0 rounded-sm"
                  onerror={(pE) => {
                    (pE.currentTarget as HTMLImageElement).style.display = 'none';
                  }}
                  src={cFavicon}
                />
              {/if}
              {cServerName}
            </span>
          {/if}

          {#if extra.mimeType}
            <span class="rounded bg-muted px-1.5 py-0.5 text-xs">{extra.mimeType}</span>
          {/if}
        </div>
      </Dialog.Description>
    </Dialog.Header>

    <div class="flex items-center justify-end gap-1">
      <ActionIconCopyToClipboard
        ariaLabel="Copy content"
        canCopy={!!extra.content}
        text={extra.content}
      />

      <Button
        class="h-7 w-7 p-0"
        disabled={!extra.content}
        onclick={fHandleDownload}
        size="sm"
        title={fTranslate('text_6eca010826ea')}
        variant="ghost"
      >
        <Download class="h-3.5 w-3.5" />
      </Button>
    </div>

    <div class="overflow-auto">
      {#if fIsImageResource(extra.mimeType, extra.uri) && extra.content}
        <div class="flex items-center justify-center">
          <img
            alt={extra.name}
            class="max-h-[70vh] max-w-full rounded object-contain"
            src={extra.content.startsWith('data:')
              ? extra.content
              : `data:${extra.mimeType || 'image/png'};base64,${extra.content}`}
          />
        </div>
      {:else if fIsCodeResource(extra.mimeType, extra.uri) && extra.content}
        <SyntaxHighlightedCode code={extra.content} language={fGetLanguage()} maxHeight="70vh" />
      {:else if extra.content}
        <pre
          class="max-h-[70vh] overflow-auto rounded-md border bg-muted/30 p-4 font-mono text-sm break-words whitespace-pre-wrap">{extra.content}</pre>
      {:else}
        <div class="py-8 text-center text-sm text-muted-foreground">
          {fTranslate('text_a7c49ff5b9e2')}
        </div>
      {/if}
    </div>
  </Dialog.Content>
</Dialog.fRoot>
