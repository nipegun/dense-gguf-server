<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { fParseWriteFileMeta, fParseWriteFileTitleMeta } from './parsers/write-file';
  import ToolCallBlock from './ToolCallBlock.svelte';
  import { XCircle } from '@lucide/svelte';
  import { SyntaxHighlightedCode } from '$lib/components/app';
  import { cMaxHeightCodeBlock, cResultStatSeparator } from '$lib/constants';
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

  const cWriteFileMeta = $derived(fParseWriteFileTitleMeta(section));
  // body-only: the full meta parses the embedded file content, and this
  // derived is read solely from the children snippet, which renders only
  // while the block is expanded
  const cWriteFileBody = $derived(fParseWriteFileMeta(section));
  const cHome = $derived(cToolsStore.serverHome);
</script>

<ToolCallBlock {isStreaming} meta={cWriteFileMeta} {onToggle} {open} {section}>
  {#snippet titleSnippet()}
    <span class="flex min-w-0 flex-wrap items-baseline gap-x-1">
      <span class="shrink-0 text-muted-foreground">{fTranslate('text_997357233881')}</span>

      <span class="flex min-w-0 items-baseline gap-1.5">
        <span class="min-w-0 overflow-x-auto font-mono" title={cWriteFileMeta?.filePath}>
          {fAbbreviateHome(cWriteFileMeta?.filePath ?? '', cHome)}
        </span>

        {#if cWriteFileMeta?.errorMessage}
          <span class="shrink-0 text-xs italic text-muted-foreground/70"
            >{fTranslate('text_a1bf75497647')}</span
          >
        {/if}
      </span>
    </span>
  {/snippet}

  {#snippet children(meta, ctx)}
    {#if meta?.errorMessage}
      <div
        class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
      >
        <XCircle class="mt-0.5 h-3 w-3 shrink-0" />

        <span>{meta.errorMessage}</span>
      </div>
    {:else if meta}
      <SyntaxHighlightedCode
        code={cWriteFileBody?.content ?? ''}
        language={meta.language}
        maxHeight={cMaxHeightCodeBlock}
        streaming={ctx.isCodeStreaming}
      />

      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {#if meta.resultMessage}
          {meta.resultMessage}{meta.bytesWritten != null ? cResultStatSeparator : ''}{/if}

        {#if meta.bytesWritten != null}
          <span class="font-mono">{meta.bytesWritten}</span>
          {fTranslate('text_277089d91c0b')}{/if}
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
