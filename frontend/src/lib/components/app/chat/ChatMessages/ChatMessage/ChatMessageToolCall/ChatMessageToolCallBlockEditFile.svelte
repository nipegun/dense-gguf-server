<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { fParseEditFileMeta, fParseEditFileTitleMeta } from './parsers/edit-file';
  import ToolCallBlock from './ToolCallBlock.svelte';
  import { XCircle } from '@lucide/svelte';
  import { cMaxHeightCodeBlock, cResultStatSeparator } from '$lib/constants';
  import { cToolsStore } from '$lib/stores';
  import type { AgenticSection } from '$lib/types';
  import { fAbbreviateHome, fComputeLineDiff, fPrefixFor } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    onToggle?: () => void;
  }

  let { isStreaming, onToggle, open, section }: Props = $props();

  const cEditFileMeta = $derived(fParseEditFileTitleMeta(section));
  // body-only: the full meta parses the embedded edit strings, and these
  // deriveds are read solely from the children snippet, which renders only
  // while the block is expanded
  const cEditFileBody = $derived(fParseEditFileMeta(section));
  const cHome = $derived(cToolsStore.serverHome);
  const aEditDiffs = $derived(
    (cEditFileBody?.edits ?? []).map((pEdit) => fComputeLineDiff(pEdit.oldText, pEdit.newText))
  );
</script>

<ToolCallBlock {isStreaming} meta={cEditFileMeta} {onToggle} {open} {section}>
  {#snippet titleSnippet()}
    <span class="flex min-w-0 flex-wrap items-baseline gap-x-1">
      <span class="shrink-0 text-muted-foreground">{fTranslate('text_9608dd142d9b')}</span>

      <span class="flex min-w-0 items-baseline gap-1.5">
        <span class="min-w-0 overflow-x-auto font-mono" title={cEditFileMeta?.filePath}>
          {fAbbreviateHome(cEditFileMeta?.filePath ?? '', cHome)}
        </span>

        {#if cEditFileMeta?.errorMessage}
          <span class="shrink-0 text-xs italic text-muted-foreground/70"
            >{fTranslate('text_a1bf75497647')}</span
          >
        {/if}
      </span>
    </span>
  {/snippet}

  {#snippet children(meta, _ctx)}
    {#if meta?.errorMessage}
      <div
        class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
      >
        <XCircle class="mt-0.5 h-3 w-3 shrink-0" />

        <span>{meta.errorMessage}</span>
      </div>
    {:else if meta && cEditFileBody && cEditFileBody.edits.length > 0}
      {#each aEditDiffs as aDiffLines, vEi (vEi)}
        <div class={vEi === 0 ? '' : 'mt-3'}>
          <div class="mb-1.5 text-xs text-muted-foreground/70 italic">
            {fTranslate('text_464c4ffd019e')}
            {vEi + 1} {fTranslate('text_28391d3bc64e')} {cEditFileBody.edits.length}
          </div>

          <div style:max-height={cMaxHeightCodeBlock} class="diff-block">
            <div class="diff-pre">
              {#each aDiffLines as vLine, vLi (vLi)}
                <div class="diff-line diff-{vLine.kind}">
                  <span class="diff-old-num">{vLine.oldLine ?? ''}</span>

                  <span class="diff-marker">{fPrefixFor(vLine.kind)}</span>

                  <span class="diff-new-num">{vLine.newLine ?? ''}</span>

                  <span class="diff-text">{vLine.text || ' '}</span>
                </div>
              {/each}
            </div>
          </div>
        </div>
      {/each}

      <div class="mt-1.5 text-xs text-muted-foreground/70 italic">
        {#if meta.resultMessage}
          {meta.resultMessage}{meta.editsApplied != null ? cResultStatSeparator : ''}{/if}

        {#if meta.editsApplied != null}
          <span class="font-mono">{meta.editsApplied}</span>
          {meta.editsApplied === 1 ? 'edit' : 'edits'} {fTranslate('text_b4267ce93cbb')}{/if}
      </div>
    {:else}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_dc7c1c0f6399')}
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>

<style>
  .diff-block {
    overflow: auto;
    border-radius: 0.75rem;
    border-width: 1px;
    border-color: color-mix(in oklch, var(--border) 30%, transparent);
    background: var(--code-background);
    box-shadow: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  }

  :global(.dark) .diff-block {
    border-color: color-mix(in oklch, var(--border) 20%, transparent);
  }

  /* Each row is a 4-column grid: old-line#, marker, new-line#, text.
	 * The gutters stay fixed-width so the text column lines up unversally. */
  .diff-line {
    display: grid;
    grid-template-columns: 3.25rem 1.5rem 3.25rem 1fr;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.65;
    align-items: stretch;
  }

  .diff-old-num,
  .diff-new-num {
    text-align: right;
    padding-right: 0.5rem;
    user-select: none;
    color: color-mix(in oklch, var(--muted-foreground) 70%, transparent);
    font-variant-numeric: tabular-nums;
  }

  .diff-marker {
    text-align: center;
    color: color-mix(in oklch, var(--muted-foreground) 70%, transparent);
    user-select: none;
  }

  .diff-line.diff-add {
    background-color: #f0fff4;
    color: #22863a;
  }
  .diff-line.diff-add .diff-new-num,
  .diff-line.diff-add .diff-marker {
    color: #22863a;
  }

  .diff-line.diff-remove {
    background-color: #ffeef0;
    color: #b31d28;
  }
  .diff-line.diff-remove .diff-old-num,
  .diff-line.diff-remove .diff-marker {
    color: #b31d28;
  }

  .diff-line.diff-add .diff-old-num,
  .diff-line.diff-remove .diff-new-num {
    /* Empty gutter columns for add/remove rows mirror git unification
		 * (added lines don't have an old number, removed lines don't have a
		 * new number). Keep them visible so columns stay aligned across
		 * mixed rows. */
    opacity: 0;
  }

  .diff-text {
    padding-left: 0.4rem;
    padding-right: 0.5rem;
    white-space: pre;
    overflow-x: auto;
    min-width: 0;
  }

  :global(.dark) .diff-line.diff-add {
    background-color: #033a16;
    color: #aff5b4;
  }
  :global(.dark) .diff-line.diff-add .diff-new-num,
  :global(.dark) .diff-line.diff-add .diff-marker {
    color: #aff5b4;
  }
  :global(.dark) .diff-line.diff-remove {
    background-color: #67060c;
    color: #ffdcd7;
  }
  :global(.dark) .diff-line.diff-remove .diff-old-num,
  :global(.dark) .diff-line.diff-remove .diff-marker {
    color: #ffdcd7;
  }
</style>
