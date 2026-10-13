<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ChevronDown from '@lucide/svelte/icons/chevron-down';
  import * as Collapsible from '$lib/components/ui/collapsible/index.js';
  import { fCn } from '$lib/components/ui/utils';
  import { cIconClassDefault } from '$lib/constants';
  import type { Snippet } from 'svelte';
  import type { Component } from 'svelte';

  interface Props {
    open?: boolean;
    class?: string;
    icon?: Component;
    iconClass?: string;
    iconUrl?: string | null;
    title?: string;
    titleSnippet?: Snippet;
    subtitle?: string;
    shimmerTitle?: boolean;
    onToggle?: () => void;
    children: Snippet;
  }

  let {
    children,
    class: className = '',
    icon: IconComponent,
    iconClass = cIconClassDefault,
    iconUrl = null,
    onToggle,
    open = $bindable(false),
    shimmerTitle = false,
    subtitle,
    title = '',
    titleSnippet
  }: Props = $props();

  function fHideBrokenIcon(pEvent: Event) {
    (pEvent.currentTarget as HTMLImageElement).style.display = 'none';
  }
</script>

<Collapsible.Root
  class={fCn('group/collapsible', 'overflow-hidden rounded-md', className)}
  onOpenChange={(pValue) => {
    open = pValue;
    onToggle?.();
  }}
  {open}
  style="background: var(--code-background); border: 1px solid color-mix(in oklch, var(--border) 30%, transparent);"
>
  <Collapsible.Trigger
    class={fCn(
      'flex w-full cursor-pointer items-start justify-between gap-2 text-left',
      'px-3 py-2'
    )}
  >
    <div class="flex min-w-0 items-start gap-2 text-muted-foreground">
      {#if iconUrl}
        <img
          alt=""
          class={fCn('shrink-0 rounded-sm mt-0.5', iconClass)}
          onerror={fHideBrokenIcon}
          src={iconUrl}
        />
      {:else if IconComponent}
        <IconComponent class={fCn('shrink-0 text-muted-foreground/60 mt-0.5', iconClass)} />
      {/if}

      <span
        class={fCn(
          'min-w-0 overflow-x-auto text-sm font-medium',
          shimmerTitle ? 'shimmer-text' : 'text-foreground/80'
        )}
      >
        {#if titleSnippet}
          {@render titleSnippet()}
        {:else}
          {title}
        {/if}
      </span>

      {#if subtitle}
        <span class="text-xs italic text-muted-foreground/70">{subtitle}</span>
      {/if}
    </div>

    <ChevronDown
      class={fCn(
        'size-4 shrink-0 text-muted-foreground/60 transition-all duration-150 ease-out opacity-0 group-hover/collapsible:opacity-100 mt-0.5',
        open && 'rotate-180'
      )}
    />

    <span class="sr-only">{fTranslate('text_fc7a07580f42')}</span>
  </Collapsible.Trigger>

  <Collapsible.Content>
    <!-- See CollapsibleContentBlock: bits-ui keeps closed content mounted, which
		     makes a collapsed tool result re-render on every streamed token. -->
    {#if open}
      <div class="p-3 pt-1">
        {@render children()}
      </div>
    {/if}
  </Collapsible.Content>
</Collapsible.Root>
