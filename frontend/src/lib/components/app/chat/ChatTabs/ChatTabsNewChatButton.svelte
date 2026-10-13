<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Plus } from '@lucide/svelte';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cIconClassDefault } from '$lib/constants';

  interface Props {
    onclick?: () => void;
  }

  let { onclick }: Props = $props();
</script>

<Tooltip.fRoot>
  <Tooltip.Trigger>
    {#snippet child({ props: dProps })}
      <button
        {...dProps}
        aria-label={fTranslate('text_db18382a249e')}
        class="backdrop-blur-lg flex h-8 w-8 mr-4 shrink-0 cursor-pointer items-center justify-center rounded-md transition-colors hover:bg-foreground/5"
        {onclick}
      >
        <Plus class="{cIconClassDefault} opacity-40 transition-opacity group-hover:opacity-100" />
      </button>
    {/snippet}
  </Tooltip.Trigger>

  <Tooltip.Content>
    <p>{fTranslate('text_db18382a249e')}</p>
  </Tooltip.Content>
</Tooltip.fRoot>
