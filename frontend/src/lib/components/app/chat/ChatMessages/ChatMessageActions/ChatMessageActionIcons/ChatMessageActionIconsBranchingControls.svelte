<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ChevronLeft, ChevronRight } from '@lucide/svelte';
  import { ActionIcon } from '$lib/components/app';
  import { fGetChatMessageActionsContext } from '$lib/contexts';

  interface Props {
    class?: string;
  }

  let { class: className = '' }: Props = $props();

  const cMessageActions = fGetChatMessageActionsContext();

  let vSiblingInfo = $derived(cMessageActions.siblingInfo);

  let vHasPrevious = $derived(vSiblingInfo && vSiblingInfo.currentIndex > 0);
  let vHasNext = $derived(
    vSiblingInfo && vSiblingInfo.currentIndex < vSiblingInfo.totalSiblings - 1
  );
  let vNextSiblingId = $derived(
    vHasNext ? vSiblingInfo!.siblingIds[vSiblingInfo!.currentIndex + 1] : null
  );
  let vPreviousSiblingId = $derived(
    vHasPrevious ? vSiblingInfo!.siblingIds[vSiblingInfo!.currentIndex - 1] : null
  );
</script>

{#if vSiblingInfo && vSiblingInfo.totalSiblings > 1}
  <div
    aria-label="Message version {vSiblingInfo.currentIndex + 1} of {vSiblingInfo.totalSiblings}"
    class="flex items-center gap-1 text-xs text-muted-foreground {className}"
    role="navigation"
  >
    <ActionIcon
      class="h-5 w-5 p-0 {!vHasPrevious ? '!cursor-not-allowed opacity-30' : ''}"
      disabled={!vHasPrevious}
      icon={ChevronLeft}
      onclick={() => cMessageActions.navigateToSibling(vPreviousSiblingId!)}
      tooltip={fTranslate('text_21d886b74bc1')}
    />

    <span class="px-1 font-mono text-xs">
      {vSiblingInfo.currentIndex + 1}/{vSiblingInfo.totalSiblings}
    </span>

    <ActionIcon
      class="h-5 w-5 p-0 {!vHasNext ? 'opacity-30' : ''}"
      disabled={!vHasNext}
      icon={ChevronRight}
      onclick={() => cMessageActions.navigateToSibling(vNextSiblingId!)}
      tooltip={fTranslate('text_0671058c0f46')}
    />
  </div>
{/if}
