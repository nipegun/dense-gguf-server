<script lang="ts">
  import ModelId from './ModelId.svelte';
  import { ActionIconCopyToClipboard, BadgeInfo } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { ModelIcon } from '$lib/constants';
  import { cModelsStore, cServerStore } from '$lib/stores';

  interface Props {
    class?: string;
    model?: string;
    onclick?: () => void;
    showCopyIcon?: boolean;
    showTooltip?: boolean;
  }

  let {
    class: className = '',
    model: modelProp,
    onclick,
    showCopyIcon = false,
    showTooltip = false
  }: Props = $props();

  let vModel = $derived(modelProp || cModelsStore.singleModelName);
  let vIsModelMode = $derived(cServerStore.isModelMode);
  let vShouldShow = $derived(vModel && (modelProp !== undefined || vIsModelMode));
</script>

{#snippet badgeContent(pTriggerProps?: Record<string, unknown>)}
  <BadgeInfo {...pTriggerProps ?? {}} class={className} {onclick}>
    {#snippet icon()}
      <ModelIcon class="h-3 w-3" />
    {/snippet}

    {#if vModel}
      <ModelId modelId={vModel} />
    {/if}

    {#if showCopyIcon}
      <ActionIconCopyToClipboard ariaLabel="Copy model name" text={vModel || ''} />
    {/if}
  </BadgeInfo>
{/snippet}

{#if vShouldShow}
  {#if showTooltip}
    <Tooltip.fRoot>
      <Tooltip.Trigger>
        <!-- prevent another nested button element -->
        {#snippet child({ props: dProps })}
          {@render badgeContent(dProps)}
        {/snippet}
      </Tooltip.Trigger>

      <Tooltip.Content>
        {onclick ? 'Click for model details' : vModel}
      </Tooltip.Content>
    </Tooltip.fRoot>
  {:else}
    {@render badgeContent()}
  {/if}
{/if}
