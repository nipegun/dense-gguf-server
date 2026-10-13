<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { fParseGrepSearchMeta } from './parsers/grep-search';
  import ToolCallBlock from './ToolCallBlock.svelte';
  import { XCircle } from '@lucide/svelte';
  import { cToolsStore } from '$lib/stores';
  import type { AgenticSection } from '$lib/types';
  import { fAbbreviateHome } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    onToggle?: () => void;
  }

  let { isStreaming, onToggle, open, section }: Props = $props();

  const cGrepMeta = $derived(fParseGrepSearchMeta(section));
  const cHome = $derived(cToolsStore.serverHome);
</script>

<ToolCallBlock {isStreaming} meta={cGrepMeta} {onToggle} {open} {section}>
  {#snippet titleSnippet()}
    {#if cGrepMeta}
      <span class="text-muted-foreground">{fTranslate('text_6ea8f7433c6c')} </span>

      <span class="font-mono">{cGrepMeta.pattern}</span>

      <span class="text-muted-foreground"> {fTranslate('text_582967534d0f')} </span>

      <span class="font-mono" title={cGrepMeta.path}>{fAbbreviateHome(cGrepMeta.path, cHome)}</span>
    {/if}
  {/snippet}

  {#snippet children(meta, ctx)}
    {#if ctx.isPending}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_78c9d9f6ace0')}
      </div>
    {:else if meta?.errorMessage}
      <div
        class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
      >
        <XCircle class="mt-0.5 h-3 w-3 shrink-0" />

        <span>{meta.errorMessage}</span>
      </div>
    {:else if meta && meta.matches.length > 0}
      <div class="max-h-96 overflow-auto">
        {#each meta.matches as vMatch, vMi (vMi)}
          <div class="font-mono text-[11px] leading-relaxed">
            <span class="text-muted-foreground/70">{vMatch.file}</span>

            {#if meta.showLineNumbers && vMatch.line != null}
              <span class="text-muted-foreground/70">:{vMatch.line}</span>
            {/if}

            <span class="text-muted-foreground/70">:</span>

            <span>{vMatch.content}</span>
          </div>
        {/each}
      </div>

      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_9f2dda0efa37')}
        <span class="font-mono">{meta.totalMatches ?? meta.matches.length}</span>

        {#if meta.showLineNumbers}
          &nbsp;<span class="italic">{fTranslate('text_f1edb1862f8d')}</span>
        {/if}
      </div>
    {:else}
      <div class="text-xs text-muted-foreground/70 italic">{fTranslate('text_2df01a03ff43')}</div>

      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_9f2dda0efa37')} <span class="font-mono">{meta?.totalMatches ?? 0}</span>
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
