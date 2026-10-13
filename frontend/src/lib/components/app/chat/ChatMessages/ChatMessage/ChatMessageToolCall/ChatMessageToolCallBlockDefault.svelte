<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  // Fall-through renderer for tool calls without a dedicated block.
  // Renders section.toolArgs / section.toolResult directly using the
  // shared chrome shell.

  import ToolCallBlock from './ToolCallBlock.svelte';
  import { Loader2 } from '@lucide/svelte';
  import { MarkdownContent, SyntaxHighlightedCode } from '$lib/components/app';
  import { cMaxHeightCodeBlock } from '$lib/constants';
  import { AttachmentType, FileTypeText, MimeTypeAudio, ToolResultKind } from '$lib/enums';
  import type { AgenticSection, DatabaseMessageExtra, ToolResultLine } from '$lib/types';
  import {
    fClassifyToolResult,
    fFormatJsonPretty,
    fGetToolUi,
    fParseToolResultWithMedia
  } from '$lib/utils';
  import { fCreateBase64DataUrl } from '$lib/utils/data-url';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    attachments?: DatabaseMessageExtra[];
    onToggle?: () => void;
  }

  let { attachments, isStreaming, onToggle, open, section }: Props = $props();

  const title = $derived(fGetToolUi(section.toolName)?.label ?? section.toolName ?? '');
  const cOutputKind = $derived(fClassifyToolResult(section.toolResult));
  const aParsedLines: ToolResultLine[] = $derived(
    section.toolResult ? fParseToolResultWithMedia(section.toolResult, attachments) : []
  );
</script>

<ToolCallBlock {isStreaming} meta={null} {onToggle} {open} {section} {title}>
  {#snippet children(_meta, ctx)}
    {#if ctx.isStreamingCall}
      <div class="mb-2 flex items-center gap-2 text-xs text-muted-foreground/70">
        <span>{fTranslate('text_36ecb4f86691')}</span>

        {#if ctx.isStreaming}
          <Loader2 class="h-3 w-3 animate-spin" />
        {/if}
      </div>

      {#if section.toolArgs}
        <SyntaxHighlightedCode
          code={fFormatJsonPretty(section.toolArgs)}
          language={FileTypeText.cJson}
          maxHeight={cMaxHeightCodeBlock}
          streaming={ctx.isCodeStreaming}
        />
      {:else if ctx.isStreaming}
        <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
          {fTranslate('text_75a15599785c')}
        </div>
      {:else}
        <div
          class="rounded bg-yellow-500/10 p-2 text-xs text-yellow-600 italic dark:text-yellow-400"
        >
          {fTranslate('text_4494ff030139')}
        </div>
      {/if}
    {:else}
      {@const cShowInput = Boolean(section.toolArgs)}
      {#if cShowInput}
        <div class="mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70">
          <span>{fTranslate('text_36ecb4f86691')}</span>
        </div>

        <SyntaxHighlightedCode
          code={fFormatJsonPretty(section.toolArgs ?? '')}
          language={FileTypeText.cJson}
          maxHeight={cMaxHeightCodeBlock}
          streaming={ctx.isCodeStreaming}
        />
      {/if}

      <div
        class={cShowInput
          ? 'mt-4 mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'
          : 'mb-1.5 flex items-center gap-2 text-xs text-muted-foreground/70'}
      >
        <span>{fTranslate('text_b2439bcb8dee')}</span>

        {#if ctx.isPending}
          <Loader2 class="h-3 w-3 animate-spin" />
        {/if}
      </div>

      {#if ctx.isPending}
        <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
          {fTranslate('text_3baf0a61c565')}
        </div>
      {:else if section.toolResult}
        {#if cOutputKind === ToolResultKind.cJson}
          <SyntaxHighlightedCode
            code={fFormatJsonPretty(section.toolResult)}
            language={FileTypeText.cJson}
            maxHeight={cMaxHeightCodeBlock}
          />
        {:else if cOutputKind === ToolResultKind.MARKDOWN}
          <MarkdownContent {attachments} content={section.toolResult} />
        {:else}
          <div class="overflow-auto">
            {#each aParsedLines as vLine, vI (vI)}
              <div class="font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
                {vLine.text}
              </div>

              {#if vLine.media}
                {#if vLine.media.type === AttachmentType.AUDIO}
                  {@const cAudioMimeType = vLine.media.mimeType ?? MimeTypeAudio.cMp3Mpeg}
                  <div class="mt-2 mb-2">
                    <audio class="w-full rounded-lg" controls>
                      <source
                        src={fCreateBase64DataUrl(cAudioMimeType, vLine.media.base64Data)}
                        type={cAudioMimeType}
                      />
                      {fTranslate('text_a21b58ba4c2f')}
                    </audio>
                  </div>
                {:else}
                  <img
                    alt={vLine.media.name}
                    class="mt-2 mb-2 h-auto max-w-full rounded-lg"
                    loading="lazy"
                    src={vLine.media.base64Url}
                  />
                {/if}
              {/if}
            {/each}
          </div>
        {/if}
      {:else}
        <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
          {fTranslate('text_f7e31759b202')}
        </div>
      {/if}
    {/if}
  {/snippet}
</ToolCallBlock>
