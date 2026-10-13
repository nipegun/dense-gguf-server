<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ChevronDown, ChevronRight } from '@lucide/svelte';
  import * as Collapsible from '$lib/components/ui/collapsible';

  interface Props {
    instructions?: string;
    class?: string;
  }

  let { class: className, instructions }: Props = $props();

  let vIsExpanded = $state(false);
</script>

{#if instructions}
  <Collapsible.Root bind:open={vIsExpanded} class={className}>
    <Collapsible.Trigger
      class="flex w-full items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
    >
      {#if vIsExpanded}
        <ChevronDown class="h-3.5 w-3.5" />
      {:else}
        <ChevronRight class="h-3.5 w-3.5" />
      {/if}

      <span>{fTranslate('text_1ec4baa68605')}</span>
    </Collapsible.Trigger>

    <Collapsible.Content class="mt-2">
      <p class="rounded bg-muted/50 p-2 text-xs text-muted-foreground">
        {instructions}
      </p>
    </Collapsible.Content>
  </Collapsible.Root>
{/if}
