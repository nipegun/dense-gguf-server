<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Loader2 } from '@lucide/svelte';
  import { StreamConnectionState } from '$lib/enums';
  import { cChatStore } from '$lib/stores';

  let vState = $derived(cChatStore.vStreamConnectionState);
</script>

{#if vState === StreamConnectionState.cResuming}
  <div
    aria-live="polite"
    class="pointer-events-auto mx-auto mt-2 mb-2 flex max-w-[48rem] items-center gap-2 rounded-md border border-blue-400/40 bg-blue-50/60 px-3 py-1.5 text-sm text-blue-700 dark:bg-blue-950/40 dark:text-blue-200"
    role="status"
  >
    <Loader2 class="h-3.5 w-3.5 animate-spin" />

    <span>{fTranslate('text_8d5d7fa8d9e8')}</span>
  </div>
{/if}
