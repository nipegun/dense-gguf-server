<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { fParseReadFileMeta } from './parsers/read-file';
  import ToolCallBlock from './ToolCallBlock.svelte';
  import { SyntaxHighlightedCode } from '$lib/components/app';
  import { cCodeBlock, cMaxHeightCodeBlock } from '$lib/constants';
  import type { AgenticSection } from '$lib/types';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    onToggle?: () => void;
  }

  let { isStreaming, onToggle, open, section }: Props = $props();

  const cReadFileMeta = $derived(fParseReadFileMeta(section));
</script>

<ToolCallBlock {isStreaming} meta={cReadFileMeta} {onToggle} {open} {section}>
  {#snippet titleSnippet()}
    <span class="flex min-w-0 flex-wrap items-baseline gap-x-1">
      <span class="shrink-0 text-muted-foreground">{fTranslate('text_47659c3dccc8')}</span>

      <span class="flex min-w-0 items-baseline gap-1.5">
        <span class="min-w-0 overflow-x-auto font-mono">{cReadFileMeta?.fileName}</span>

        {#if cReadFileMeta?.lineRange}
          <span class="shrink-0 text-muted-foreground">
            {fTranslate('text_83f47275bb38')}
            {cReadFileMeta.lineRange.start}-{cReadFileMeta.lineRange.end})
          </span>
        {/if}
      </span>
    </span>
  {/snippet}

  {#snippet children(_meta, _ctx)}
    {#if section.toolResult}
      <SyntaxHighlightedCode
        code={section.toolResult}
        language={cReadFileMeta?.language ?? cCodeBlock.DEFAULT_LANGUAGE}
        maxHeight={cMaxHeightCodeBlock}
      />
    {:else}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_510899fbc0a1')}
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
