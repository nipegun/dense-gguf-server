<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ChatMessageActionCard from './ChatMessageActionCard.svelte';
  import { RotateCw } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';

  interface Props {
    onDecision: (pShouldContinue: boolean) => void;
  }

  let { onDecision }: Props = $props();
</script>

<ChatMessageActionCard icon={RotateCw}>
  {#snippet message()}{fTranslate('text_8e578ac4f04b')}{/snippet}

  {#snippet actions()}
    <Button onclick={() => onDecision(true)} size="sm">{fTranslate('text_31fbef162594')}</Button>

    <Button
      class="text-destructive hover:text-destructive"
      onclick={() => onDecision(false)}
      size="sm"
      variant="destructive"
    >
      {fTranslate('text_cae7d57bc067')}
    </Button>
  {/snippet}
</ChatMessageActionCard>
