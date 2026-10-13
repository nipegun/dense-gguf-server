<script lang="ts">
  import { BadgeInfo } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { fCopyToClipboard } from '$lib/utils';
  import type { Component } from 'svelte';

  interface Props {
    class?: string;
    icon: Component;
    value: string | number;
    tooltipLabel?: string;
  }

  let { class: className = '', icon: IconComponent, tooltipLabel, value }: Props = $props();

  function fHandleClick() {
    void fCopyToClipboard(String(value));
  }
</script>

{#if tooltipLabel}
  <Tooltip.fRoot>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props: dProps })}
        <BadgeInfo {...dProps} class={className} onclick={fHandleClick}>
          {#snippet icon()}
            <IconComponent class="h-3 w-3" />
          {/snippet}

          {value}
        </BadgeInfo>
      {/snippet}
    </Tooltip.Trigger>

    <Tooltip.Content>
      <p>{tooltipLabel}</p>
    </Tooltip.Content>
  </Tooltip.fRoot>
{:else}
  <BadgeInfo class={className} onclick={fHandleClick}>
    {#snippet icon()}
      <IconComponent class="h-3 w-3" />
    {/snippet}

    {value}
  </BadgeInfo>
{/if}
