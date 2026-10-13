<script lang="ts">
  import { KeyboardShortcutInfo } from '$lib/components/app';
  import * as Drawer from '$lib/components/ui/drawer';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import { Separator } from '$lib/components/ui/separator';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cDeviceStore } from '$lib/stores';
  import type { Component } from 'svelte';

  interface ActionItem {
    icon: Component;
    label: string;
    onclick: (pEvent: Event) => void;
    variant?: 'default' | 'destructive';
    disabled?: boolean;
    shortcut?: string[];
    separator?: boolean;
  }

  interface Props {
    triggerIcon: Component;
    triggerTooltip?: string;
    triggerClass?: string;
    actions: ActionItem[];
    align?: 'start' | 'center' | 'end';
    open?: boolean;
  }

  let {
    actions,
    align = 'end',
    open = $bindable(false),
    triggerClass = '',
    triggerIcon,
    triggerTooltip
  }: Props = $props();

  // a phone shows the same actions in a drawer, where a thumb reaches them and every
  // row is a touch target of its own
  let vIsMobile = $derived(cDeviceStore.isMobile);

  function fRunAction(pAction: ActionItem, pEvent: Event) {
    open = false;

    // let the drawer close before an action that moves the focus, such as a confirm dialog
    setTimeout(() => pAction.onclick(pEvent), 0);
  }
</script>

{#if vIsMobile}
  <button
    class="flex h-6 w-6 cursor-pointer items-center justify-center rounded-md p-0 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-accent data-[state=open]:text-accent-foreground {triggerClass}"
    data-slot="dropdown-menu-trigger"
    data-state={open ? 'open' : 'closed'}
    onclick={(pEvent) => {
      pEvent.stopPropagation();
      open = true;
    }}
    type="button"
  >
    {@render iconComponent(triggerIcon, 'h-3 w-3')}

    {#if triggerTooltip}
      <span class="sr-only">{triggerTooltip}</span>
    {/if}
  </button>

  <Drawer.Root bind:open>
    <Drawer.Content>
      <Drawer.Header>
        <Drawer.Title>{triggerTooltip ?? 'Actions'}</Drawer.Title>
      </Drawer.Header>

      <div class="flex flex-col px-2 pb-4">
        {#each actions as vAction, vIndex (vAction.label)}
          {#if vAction.separator && vIndex > 0}
            <Separator class="my-1" />
          {/if}

          <button
            class="flex min-h-11 w-full cursor-pointer items-center gap-3 rounded-md px-3 text-left text-sm transition-colors hover:bg-accent disabled:pointer-events-none disabled:opacity-50 {vAction.variant ===
            'destructive'
              ? 'text-destructive'
              : ''}"
            disabled={vAction.disabled}
            onclick={(pEvent) => fRunAction(vAction, pEvent)}
            type="button"
          >
            {@render iconComponent(vAction.icon, 'h-4 w-4 shrink-0')}

            <span class="flex-1">{vAction.label}</span>

            {#if vAction.shortcut}
              <KeyboardShortcutInfo keys={vAction.shortcut} variant={vAction.variant} />
            {/if}
          </button>
        {/each}
      </div>
    </Drawer.Content>
  </Drawer.Root>
{:else}
  <DropdownMenu.fRoot bind:open>
    <Tooltip.fRoot>
      <Tooltip.Trigger>
        <!-- prevent another nested button element -->
        {#snippet child({ props: dProps })}
          <DropdownMenu.Trigger
            {...dProps}
            class="flex h-6 w-6 cursor-pointer items-center justify-center rounded-md p-0 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-accent data-[state=open]:text-accent-foreground {triggerClass}"
            onclick={(pE) => pE.stopPropagation()}
          >
            {@render iconComponent(triggerIcon, 'h-3 w-3')}

            {#if triggerTooltip}
              <span class="sr-only">{triggerTooltip}</span>
            {/if}
          </DropdownMenu.Trigger>
        {/snippet}
      </Tooltip.Trigger>

      {#if triggerTooltip}
        <Tooltip.Content>
          <p>{triggerTooltip}</p>
        </Tooltip.Content>
      {/if}
    </Tooltip.fRoot>

    <DropdownMenu.Content {align} class="z-[999999] w-48">
      {#each actions as vAction, vIndex (vAction.label)}
        {#if vAction.separator && vIndex > 0}
          <DropdownMenu.Separator />
        {/if}

        <DropdownMenu.Item
          class="flex items-center justify-between hover:[&>kbd]:opacity-100"
          disabled={vAction.disabled}
          onclick={vAction.onclick}
          variant={vAction.variant}
        >
          <div class="flex items-center gap-2">
            {@render iconComponent(
              vAction.icon,
              `h-4 w-4 ${vAction.variant === 'destructive' ? 'text-destructive' : ''}`
            )}
            {vAction.label}
          </div>

          {#if vAction.shortcut}
            <KeyboardShortcutInfo keys={vAction.shortcut} variant={vAction.variant} />
          {/if}
        </DropdownMenu.Item>
      {/each}
    </DropdownMenu.Content>
  </DropdownMenu.fRoot>
{/if}

{#snippet iconComponent(IconComponent: Component, pClassName: string)}
  <IconComponent class={pClassName} />
{/snippet}
