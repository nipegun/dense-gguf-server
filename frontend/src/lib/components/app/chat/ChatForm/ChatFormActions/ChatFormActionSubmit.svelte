<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ArrowUp } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Tooltip from '$lib/components/ui/tooltip';

  interface Props {
    canSend?: boolean;
    disabled?: boolean;
    showErrorState?: boolean;
    tooltipLabel?: string;
  }

  let { canSend = false, disabled = false, showErrorState = false, tooltipLabel }: Props = $props();

  let vIsDisabled = $derived(!canSend || disabled);
</script>

{#snippet submitButton(pProps = {})}
  <Button
    class={[
      'md:h-8 md:w-8 h-9 w-9 rounded-full p-0',
      showErrorState &&
        'bg-red-400/10 text-red-400 hover:bg-red-400/20 hover:text-red-400 disabled:opacity-100'
    ]}
    disabled={vIsDisabled}
    type="submit"
    {...pProps}
  >
    <span class="sr-only">{fTranslate('text_f6f4688ff23d')}</span>

    <ArrowUp class="h-12 w-12" />
  </Button>
{/snippet}

{#if tooltipLabel}
  <Tooltip.fRoot>
    <Tooltip.Trigger>
      {@render submitButton()}
    </Tooltip.Trigger>

    <Tooltip.Content>
      <p>{tooltipLabel}</p>
    </Tooltip.Content>
  </Tooltip.fRoot>
{:else}
  {@render submitButton()}
{/if}
