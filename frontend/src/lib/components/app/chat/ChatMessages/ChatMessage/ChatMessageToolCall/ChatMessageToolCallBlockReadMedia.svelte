<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { fParseReadMediaMeta } from './parsers/read-media';
  import ToolCallBlock from './ToolCallBlock.svelte';
  import { cAttachmentSavedRegex } from '$lib/constants/agentic.constants';
  import { AttachmentType, MimeTypeAudio } from '$lib/enums';
  import type { DatabaseMessageExtraAudioFile, DatabaseMessageExtraImageFile } from '$lib/types';
  import type { AgenticSection } from '$lib/types';
  import { fCreateBase64DataUrl } from '$lib/utils/data-url';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    onToggle?: () => void;
  }

  let { isStreaming, onToggle, open, section }: Props = $props();

  const cReadMediaMeta = $derived(fParseReadMediaMeta(section));

  // extractBase64Attachments swapped the data URI line for [Attachment saved: name]
  // and moved the bytes to the message extras, so the name is the only link back
  const cMediaAttachment = $derived.by(() => {
    const cExtras = section.toolResultExtras;

    if (!cExtras || cExtras.length === 0) return null;

    const cMatch = section.toolResult?.match(cAttachmentSavedRegex);

    if (!cMatch) return null;

    const cAttachmentName = cMatch[1];

    return (
      cExtras.find(
        (pE): pE is DatabaseMessageExtraImageFile | DatabaseMessageExtraAudioFile =>
          (pE.type === AttachmentType.IMAGE || pE.type === AttachmentType.AUDIO) &&
          pE.name === cAttachmentName
      ) ?? null
    );
  });

  const cAudioMimeType = $derived(cReadMediaMeta?.mimeType ?? MimeTypeAudio.cMp3Mpeg);
</script>

<ToolCallBlock {isStreaming} meta={cReadMediaMeta} {onToggle} {open} {section}>
  {#snippet titleSnippet()}
    <span class="flex min-w-0 flex-wrap items-baseline gap-x-1">
      <span class="shrink-0 text-muted-foreground">{fTranslate('dgs_aa4c7701ac82')}</span>

      <span class="min-w-0 overflow-x-auto font-mono">{cReadMediaMeta?.fileName}</span>
    </span>
  {/snippet}

  {#snippet children(_meta, _ctx)}
    {#if section.toolResult}
      {#if !cMediaAttachment}
        <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
          {fTranslate('dgs_1b5ee78006e8')}
        </div>
      {:else if cMediaAttachment.type === AttachmentType.AUDIO}
        <div class="mt-2">
          <audio class="w-full rounded-lg" controls>
            <source
              src={fCreateBase64DataUrl(cAudioMimeType, cMediaAttachment.base64Data)}
              type={cAudioMimeType}
            />
            {fTranslate('text_a21b58ba4c2f')}
          </audio>
        </div>
      {:else}
        <div class="mt-2">
          <img
            alt={cReadMediaMeta?.fileName ?? 'media'}
            class="max-h-[60vh] max-w-full rounded-lg object-contain shadow-lg"
            loading="lazy"
            src={cMediaAttachment.base64Url}
          />
        </div>
      {/if}

      {#if cReadMediaMeta?.sizeBytes || cReadMediaMeta?.mimeType}
        <div class="mt-2 flex gap-4 text-xs text-muted-foreground">
          {#if cReadMediaMeta?.sizeBytes}
            <span
              >{fTranslate('dgs_c987f668d963')}
              {cReadMediaMeta.sizeBytes}
              {fTranslate('text_277089d91c0b')}</span
            >
          {/if}

          {#if cReadMediaMeta?.mimeType}
            <span>{fTranslate('dgs_a65bf06ff610')} {cReadMediaMeta.mimeType}</span>
          {/if}
        </div>
      {/if}

      {#if cReadMediaMeta?.path}
        <div class="mt-1 font-mono text-xs text-muted-foreground/60">{cReadMediaMeta.path}</div>
      {/if}
    {:else}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {fTranslate('dgs_d226fe2dd9b7')}
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
