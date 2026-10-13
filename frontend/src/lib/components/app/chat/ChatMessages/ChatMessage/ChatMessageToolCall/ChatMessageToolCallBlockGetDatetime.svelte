<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Clock, Loader2 } from '@lucide/svelte';
  import { AgenticSectionType } from '$lib/enums';
  import type { AgenticSection } from '$lib/types';

  interface Props {
    section: AgenticSection;
    isStreaming?: boolean;
  }

  let { isStreaming = false, section }: Props = $props();

  const cIsPending = $derived(section.type === AgenticSectionType.cToolCallPending);
  const cIsStreamingCall = $derived(section.type === AgenticSectionType.cToolCallStreaming);
  const cShowSpinner = $derived(cIsPending || (cIsStreamingCall && isStreaming));

  type GetDatetimeMeta = {
    dateString?: string;
    errorMessage?: string;
  };

  function fParseGetDatetimeMeta(pToolResultString: string | undefined): GetDatetimeMeta {
    if (!pToolResultString) return {};

    try {
      const cParsed: unknown = JSON.parse(pToolResultString);

      if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
        const dObj = cParsed as Record<string, unknown>;

        if (typeof dObj.error === 'string') return { errorMessage: dObj.error };

        if (typeof dObj.result === 'string') return { dateString: dObj.result.trim() };
      }
    } catch {
      // not JSON - nothing to show
    }

    return {};
  }

  const cDateMeta = $derived(fParseGetDatetimeMeta(section.toolResult));
</script>

<div class="text-muted-foreground flex items-center gap-2 py-1.5">
  <Clock class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />

  {#if cShowSpinner}
    <span class="text-foreground/80 text-sm font-medium">{fTranslate('text_06dafda7802b')}</span>

    <Loader2 class="text-muted-foreground/70 h-3 w-3 animate-spin" />
  {:else if cDateMeta.errorMessage}
    <span class="text-foreground/80 text-sm font-medium">{fTranslate('text_06dafda7802b')} </span>

    <span class="text-red-600 text-xs italic dark:text-red-400"
      >-&nbsp;{cDateMeta.errorMessage}</span
    >
  {:else if cDateMeta.dateString}
    <span class="text-foreground/80 text-sm font-medium">{fTranslate('text_61655615b3bd')} </span>

    <span class="font-mono text-foreground/90 text-sm">{cDateMeta.dateString}</span>
  {:else}
    <span class="text-foreground/80 text-sm font-medium">{fTranslate('text_06dafda7802b')}</span>
  {/if}
</div>
