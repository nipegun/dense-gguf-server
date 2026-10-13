<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { fParseRunJavascriptMeta } from './parsers/run-javascript';
  import ToolCallBlock from './ToolCallBlock.svelte';
  import { Terminal, XCircle } from '@lucide/svelte';
  import { SyntaxHighlightedCode } from '$lib/components/app';
  import { cMaxHeightCodeBlock } from '$lib/constants';
  import { FileTypeText } from '$lib/enums';
  import type { AgenticSection } from '$lib/types';
  import { fGetToolUi } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    onToggle?: () => void;
  }

  let { isStreaming, onToggle, open, section }: Props = $props();

  const cRunJsMeta = $derived(fParseRunJavascriptMeta(section));
  const title = $derived(fGetToolUi(section.toolName)?.label ?? section.toolName ?? '');
</script>

<ToolCallBlock {isStreaming} meta={cRunJsMeta} {onToggle} {open} {section} {title}>
  {#snippet children(meta, ctx)}
    {#if ctx.isPending}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {fTranslate('text_4977a7e5bb36')}
      </div>
    {:else if meta?.errorMessage}
      <div
        class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
      >
        <XCircle class="mt-0.5 h-3 w-3 shrink-0" />

        <span>{meta.errorMessage}</span>
      </div>

      <div class="mt-3">
        <SyntaxHighlightedCode
          code={meta.code}
          language={FileTypeText.cJavascript}
          maxHeight={cMaxHeightCodeBlock}
          streaming={ctx.isCodeStreaming}
        />
      </div>
    {:else if meta}
      <SyntaxHighlightedCode
        code={meta.code}
        language={FileTypeText.cJavascript}
        maxHeight={cMaxHeightCodeBlock}
        streaming={ctx.isCodeStreaming}
      />

      <div class="mb-2 mt-3 flex items-center gap-2 text-xs text-muted-foreground/70">
        <Terminal class="h-3 w-3" />

        <span>{fTranslate('text_29a40861bafe')}</span>

        {#if meta.timeoutMs != null}
          <span class="font-mono"
            >{fTranslate('dgs_4308b469c94a')} {meta.timeoutMs} {fTranslate(
              'text_f785c3ce1d58'
            )}</span
          >
        {/if}
      </div>

      {#if section.toolResult}
        <div class="mt-1">
          <SyntaxHighlightedCode
            code={section.toolResult}
            language={FileTypeText.cJavascript}
            maxHeight={cMaxHeightCodeBlock}
          />
        </div>
      {:else}
        <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
          {fTranslate('text_f7e31759b202')}
        </div>
      {/if}
    {/if}
  {/snippet}
</ToolCallBlock>
