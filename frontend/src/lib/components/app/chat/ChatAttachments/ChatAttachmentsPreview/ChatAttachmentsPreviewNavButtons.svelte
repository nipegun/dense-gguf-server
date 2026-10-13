<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ChevronLeft, ChevronRight } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';

  interface Props {
    onPrev: () => void;
    onNext: () => void;
    show: boolean;
  }

  let { onNext, onPrev, show }: Props = $props();
</script>

{#if show}
  <Button
    aria-label={fTranslate('text_a57b08a480b8')}
    class="absolute top-1/2 left-4 z-10 h-8 w-8 -translate-y-1/2 rounded-full bg-background/5 p-0 text-white!"
    onclick={onPrev}
    size="icon"
    variant="secondary"
  >
    <ChevronLeft class="size-4" />
  </Button>

  <Button
    aria-label={fTranslate('text_1ff57a29d7c9')}
    class="absolute top-1/2 right-4 z-10 h-8 w-8 -translate-y-1/2 rounded-full bg-background/5 p-0 text-white!"
    onclick={onNext}
    size="icon"
    variant="secondary"
  >
    <ChevronRight class="size-4" />
  </Button>
{/if}
