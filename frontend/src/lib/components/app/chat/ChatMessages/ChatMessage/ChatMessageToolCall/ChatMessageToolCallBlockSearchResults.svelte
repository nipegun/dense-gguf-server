<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Globe, Loader2 } from '@lucide/svelte';
  import { CollapsibleContentBlock } from '$lib/components/app';
  import * as HoverCard from '$lib/components/ui/hover-card';
  import { cIconClassDefault, cIconClassSpin } from '$lib/constants';
  import { AgenticSectionType } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';
  import type { AgenticSection, SearchResult } from '$lib/types';
  import {
    fBuildProxiedIconUrl,
    fExtractSearchQuery,
    fExtractSearchResults,
    fFaviconForUrl,
    fSanitizeExternalUrl
  } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    open?: boolean;
    isStreaming?: boolean;
    onToggle?: () => void;
  }

  let { isStreaming = false, onToggle, open = $bindable(false), section }: Props = $props();

  const cIsPending = $derived(section.type === AgenticSectionType.cToolCallPending);
  const cIsStreamingCall = $derived(section.type === AgenticSectionType.cToolCallStreaming);
  const cShowSpinner = $derived(cIsPending || (cIsStreamingCall && isStreaming));

  const aResults = $derived(fExtractSearchResults(section.toolResult));
  const cQuery = $derived(fExtractSearchQuery(section.toolArgs));

  // Same icon-resolution chain as ChatMessageToolCallBlockDefault so
  // MCP-server branding is consistent across both views. Spinner wins
  // while the call is in flight so the user sees execution status.
  const cIconUrl = $derived(
    cShowSpinner ? null : cMcpStore.mGetServerFaviconForTool(section.toolName)
  );
  const cIcon = $derived(cShowSpinner ? Loader2 : undefined);
  const cIconClass = $derived(cShowSpinner ? cIconClassSpin : cIconClassDefault);

  // Verb reflects state: "Searching" while the call is in flight, "Searched"
  // once results (or a definitive empty response) have arrived. Lets the
  // heading read as a live progress indicator rather than a completed
  // retrospective.
  const title = $derived.by(() => {
    const cVerb = cShowSpinner ? 'Searching' : 'Searched';

    return cQuery ? `${cVerb} web for "${cQuery}"` : `${cVerb} web`;
  });

  function fHideBrokenIcon(pEvent: Event) {
    (pEvent.currentTarget as HTMLImageElement).style.display = 'none';
  }

  function fFormatPublishDate(pIso: string | undefined): string | null {
    if (!pIso) return null;

    try {
      const cDate = new Date(pIso);

      if (Number.isNaN(cDate.getTime())) return pIso;

      return cDate.toLocaleDateString(undefined, {
        day: 'numeric',
        month: 'short',
        year: 'numeric'
      });
    } catch {
      return pIso;
    }
  }

  function fHostFor(pUrl: string): string | null {
    try {
      return new URL(pUrl).host;
    } catch {
      return null;
    }
  }

  function fHasDetails(pResult: SearchResult): boolean {
    return Boolean(pResult.highlights || pResult.published || pResult.author);
  }
</script>

{#snippet pill(pResult: SearchResult)}
  {@const cFaviconUrl = fBuildProxiedIconUrl(
    fFaviconForUrl(pResult.url),
    cMcpStore.isProxyAvailable
  )}
  {@const cSafeUrl = fSanitizeExternalUrl(pResult.url)}
  {@const cShowHoverCard = cSafeUrl !== null && fHasDetails(pResult)}
  {#if cSafeUrl}
    <HoverCard.Root closeDelay={100} openDelay={150}>
      <HoverCard.Trigger
        class="hover:bg-muted/80 focus-visible:ring-ring inline-flex max-w-full items-center gap-1.5 rounded-full border bg-muted px-2.5 py-1 text-xs transition-colors outline-none focus-visible:ring-2"
        href={cSafeUrl}
        rel="noopener noreferrer"
        target="_blank"
      >
        {#if cFaviconUrl}
          <img
            alt=""
            class="h-3 w-3 shrink-0 rounded-sm"
            onerror={fHideBrokenIcon}
            src={cFaviconUrl}
          />
        {:else}
          <Globe class="text-muted-foreground/70 h-3 w-3 shrink-0" />
        {/if}

        <span class="truncate font-medium text-foreground/80">{pResult.title}</span>
      </HoverCard.Trigger>

      {#if cShowHoverCard}
        {@const cPublishDate = fFormatPublishDate(pResult.published)}
        {@const cHost = fHostFor(cSafeUrl)}
        <HoverCard.Content
          align="start"
          class="bg-popover text-popover-foreground z-50 w-80 max-w-[90vw] rounded-lg border p-0 shadow-lg"
          side="top"
          sideOffset={6}
        >
          <div class="flex flex-col gap-2 p-3">
            <a
              class="line-clamp-3 text-sm font-medium leading-snug hover:underline"
              href={cSafeUrl}
              rel="noopener noreferrer"
              target="_blank">{pResult.title}</a
            >

            {#if cPublishDate || pResult.author}
              <div class="text-muted-foreground flex items-center gap-1.5 text-[11px]">
                {#if cPublishDate}
                  <span>{cPublishDate}</span>
                {/if}

                {#if cPublishDate && pResult.author}
                  <span class="opacity-50">&middot;</span>
                {/if}

                {#if pResult.author}
                  <span class="truncate">{pResult.author}</span>
                {/if}
              </div>
            {/if}

            {#if pResult.highlights}
              <p
                class="text-popover-foreground/85 line-clamp-5 text-xs leading-relaxed whitespace-pre-line"
              >
                {pResult.highlights}
              </p>
            {/if}

            {#if cHost}
              <div class="text-muted-foreground/80 truncate text-[11px]">{cHost}</div>
            {/if}
          </div>
        </HoverCard.Content>
      {/if}
    </HoverCard.Root>
  {/if}
{/snippet}

<CollapsibleContentBlock
  class="my-2"
  icon={cIcon}
  iconClass={cIconClass}
  iconUrl={cIconUrl}
  {onToggle}
  {open}
  {title}
>
  {#if aResults.length > 0}
    <div class="flex flex-wrap items-center gap-2 pb-1">
      {#each aResults as vResult (vResult.url)}
        {@render pill(vResult)}
      {/each}
    </div>
  {:else if cShowSpinner}
    <div class="text-muted-foreground/70 flex items-center gap-2 py-1 text-xs italic">
      <Loader2 class="h-3 w-3 animate-spin" />

      <span>{fTranslate('text_78c9d9f6ace0')}</span>
    </div>
  {:else}
    <div class="text-muted-foreground/70 py-1 text-xs italic">
      {fTranslate('text_a43619f32117')}
    </div>
  {/if}
</CollapsibleContentBlock>
