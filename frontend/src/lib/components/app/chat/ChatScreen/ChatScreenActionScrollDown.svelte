<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ArrowDown } from '@lucide/svelte';
  import ActionIcon from '$lib/components/app/actions/ActionIcon.svelte';
  import { cIconClassDefault } from '$lib/constants';

  let { onclick }: { onclick: (pE?: MouseEvent) => void } = $props();
</script>

<div class="pointer-events-auto flex justify-center relative h-0">
  <ActionIcon
    ariaLabel="Scroll to bottom"
    class="h-9 w-9 rounded-full bg-muted/60 border border-border/20 shadow-sm text-accent-foreground absolute bottom-4"
    icon={ArrowDown}
    iconSize={cIconClassDefault}
    {onclick}
    size="lg"
    tooltip={fTranslate('text_6a36c1b26f1c')}
  />
</div>
