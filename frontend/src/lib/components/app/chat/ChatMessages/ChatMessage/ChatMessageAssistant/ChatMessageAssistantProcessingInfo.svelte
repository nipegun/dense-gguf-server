<script lang="ts">
  import { cQueueStore } from '$lib/stores/queue.svelte';
  import { cConversationsStore } from '$lib/stores';
  import { fTranslate } from '$lib/i18n';
  const cQueue = $derived(cQueueStore.mGet(cConversationsStore.vActiveConversation?.id));
  const cQueueText = $derived(
    cQueue
      ? fTranslate(cQueue.ahead_api > 0 ? 'text_5e0c9a7d2b41' : 'text_8f3b6c1e9d07', {
          p0: cQueue.ahead
        })
      : null
  );

  import type { UseProcessingStateReturn } from '$lib/hooks/use-processing-state.svelte';
  import { fade } from 'svelte/transition';

  interface Props {
    modelLoadingText: string | null;
    processingState: UseProcessingStateReturn;
    position: 'top' | 'bottom';
  }

  let { modelLoadingText, position, processingState }: Props = $props();

  const cMarginClass = $derived(position === 'top' ? 'mt-6' : 'mt-4');
</script>

<div in:fade class="{cMarginClass} w-full max-w-3xl">
  <div class="flex flex-col items-start gap-2">
    <span class="shimmer-text text-sm">
      {cQueueText ??
        modelLoadingText ??
        processingState.mGetPromptProgressText() ??
        processingState.mGetProcessingMessage() ??
        'Processing...'}
    </span>
  </div>
</div>
