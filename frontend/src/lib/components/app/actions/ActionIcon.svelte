<script lang="ts">
  import { Button, type ButtonSize, type ButtonVariant } from '$lib/components/ui/button';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { TooltipSide } from '$lib/enums';
  import type { Component } from 'svelte';

  interface Props {
    ariaLabel?: string;
    class?: string;
    disabled?: boolean;
    href?: string;
    icon: Component;
    iconSize?: string;
    onclick?: (pE?: MouseEvent) => void;
    size?: ButtonSize;
    stopPropagationOnClick?: boolean;
    tooltip?: string;
    tooltipAsTitle?: boolean;
    tooltipSide?: TooltipSide;
    variant?: ButtonVariant;
  }

  let {
    ariaLabel,
    class: className = '',
    disabled = false,
    href = '',
    icon,
    iconSize = 'h-3 w-3',
    onclick,
    size = 'sm',
    stopPropagationOnClick = false,
    tooltip,
    tooltipAsTitle = false,
    tooltipSide = TooltipSide.TOP,
    variant = 'ghost'
  }: Props = $props();

  let vInnerWidth = $state(0);
  const cShowTooltip = $derived(!!tooltip && !tooltipAsTitle && vInnerWidth > 768);
</script>

{#snippet button(pProps = {})}
  <Button
    {...pProps}
    aria-label={ariaLabel || tooltip}
    class="h-6 w-6 p-0 {className} flex hover:bg-transparent data-[state=open]:bg-transparent!"
    {disabled}
    {href}
    onclick={(pE: MouseEvent) => {
      if (stopPropagationOnClick) pE.stopPropagation();

      onclick?.(pE);
    }}
    {size}
    title={tooltipAsTitle ? tooltip : undefined}
    {variant}
  >
    {#if icon}
      {@const IconComponent = icon}

      <IconComponent class={iconSize} />
    {/if}
  </Button>
{/snippet}

{#if cShowTooltip}
  <Tooltip.fRoot>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props: dProps })}
        {#if disabled}
          <!-- disabled buttons have pointer-events:none; wrap in a span so the tooltip hover surface stays alive -->
          <span {...dProps}>
            {@render button({})}
          </span>
        {:else}
          {@render button(dProps)}
        {/if}
      {/snippet}
    </Tooltip.Trigger>

    <Tooltip.Content side={tooltipSide}>
      <p>{tooltip}</p>
    </Tooltip.Content>
  </Tooltip.fRoot>
{:else}
  {@render button({ href })}
{/if}

<svelte:window bind:innerWidth={vInnerWidth} />
