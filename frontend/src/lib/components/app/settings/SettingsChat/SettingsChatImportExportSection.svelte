<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Button, type ButtonVariant } from '$lib/components/ui/button';
  import { cIconClassDefault } from '$lib/constants';
  import type { Component } from 'svelte';

  let {
    buttonClass,
    buttonText,
    buttonVariant,
    description,
    IconComponent,
    onclick,
    summary,
    title,
    titleClass,
    wrapperClass
  }: {
    title: string;
    description: string;
    IconComponent: Component;
    buttonText: string;
    onclick: () => void;
    titleClass?: string;
    buttonVariant?: ButtonVariant;
    buttonClass?: string;
    wrapperClass?: string;
    summary?: { show: boolean; verb: string; items: DatabaseConversation[] };
  } = $props();

  let vSectionButtonClass = $derived(buttonClass ?? 'justify-start justify-self-start md:w-auto');
  let vSectionButtonVariant = $derived(buttonVariant ?? 'outline');
</script>

<div class="grid gap-1 {wrapperClass ?? ''}">
  <h4 class="mt-0 mb-2 text-sm font-medium {titleClass ?? ''}">{title}</h4>

  <p class="mb-4 text-sm text-muted-foreground">{description}</p>

  <Button class={vSectionButtonClass} {onclick} variant={vSectionButtonVariant}>
    <IconComponent class="mr-2 {cIconClassDefault}" />

    {buttonText}
  </Button>

  {#if summary && summary.show && summary.items.length > 0}
    <div class="mt-4 grid overflow-x-auto rounded-lg border border-border/50 bg-muted/30 p-4">
      <h5 class="mb-2 text-sm font-medium">
        {summary.verb}
        {summary.items.length}
        {fTranslate('text_8b34dbc2c05e')}{summary.items.length === 1 ? '' : 's'}
      </h5>

      <ul class="space-y-1 text-sm text-muted-foreground">
        {#each summary.items.slice(0, 10) as vConv (vConv.id)}
          <li class="truncate">• {vConv.name || 'Untitled conversation'}</li>
        {/each}

        {#if summary.items.length > 10}
          <li class="italic">
            {fTranslate('text_4044e985a090')}
            {summary.items.length - 10}
            {fTranslate('text_187897ce0afc')}
          </li>
        {/if}
      </ul>
    </div>
  {/if}
</div>
