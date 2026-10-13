<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Search } from '@lucide/svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { ActionIcon, KeyboardShortcutInfo, SearchInput } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import {
    cIconClassDefault,
    cIconStripTransitionDelayMultiplier,
    cIconStripTransitionDuration,
    cRoutes,
    aSidebarActionsItems
  } from '$lib/constants';
  import { SidebarAction, TooltipSide } from '$lib/enums';
  import { cConversationsStore, cDeviceStore } from '$lib/stores';
  import type { DesktopIconStripItem } from '$lib/types';
  import type { Component } from 'svelte';
  import { onMount } from 'svelte';
  import { circIn } from 'svelte/easing';
  import { fade } from 'svelte/transition';

  interface Props {
    class: string;
    isExpandedMode: boolean;
    isSearchModeActive: boolean;
    searchQuery: string;
    onSearchDeactivated?: () => void;
    onSearchClick?: () => void;
    onManageModelsClick?: () => void;
    onMcpClick?: () => void;
    onNewChat?: () => void;
    onSettingsClick?: () => void;
  }

  let {
    class: className,
    isExpandedMode = false,
    isSearchModeActive = $bindable(false),
    onManageModelsClick,
    onMcpClick,
    onNewChat,
    onSearchClick,
    onSearchDeactivated,
    onSettingsClick,
    searchQuery = $bindable('')
  }: Props = $props();

  // the models manager stays desktop-only: a phone picks its models in the selector
  const actions = $derived(
    aSidebarActionsItems.filter(
      (pItem) => pItem.action !== SidebarAction.cManageModels || !cDeviceStore.isMobile
    )
  );
  let vInitialized = $state(false);
  let vShowIcons = $state(false);
  let vSearchInputRef = $state<HTMLInputElement | null>(null);

  const cIsOnMobile = $derived(cDeviceStore.isMobile);

  $effect(() => {
    if (isSearchModeActive && vSearchInputRef) {
      vSearchInputRef.focus();
    }
  });

  onMount(() => {
    vShowIcons = true;

    setTimeout(() => {
      vInitialized = true;
    }, cIconStripTransitionDelayMultiplier * actions.length);
  });

  /** Click handler of one icon strip item: a sidebar action, a route, or the search. */
  function fItemClickHandler(
    pItem: DesktopIconStripItem,
    pIsSearchOnMobile: boolean
  ): (() => void) | undefined {
    switch (pItem.action) {
      case SidebarAction.cNewChat:
        return () => {
          onNewChat?.();
          void cConversationsStore.mOpenNewChat();
        };
      case SidebarAction.cManageModels:
        return () => onManageModelsClick?.();
      case SidebarAction.cMcp:
        return () => onMcpClick?.();
      case SidebarAction.cSettings:
        return () => onSettingsClick?.();
      default:
        break;
    }

    if (pItem.route) {
      return () => {
        onNewChat?.();
        goto(pItem.route!);
      };
    }

    return pIsSearchOnMobile ? undefined : onSearchClick;
  }

  function fHandleSearchModeDeactivate() {
    isSearchModeActive = false;
    searchQuery = '';
    onSearchDeactivated?.();
  }

  function fIsItemActive(pItem: {
    activeRouteId?: string;
    activeRoutePrefix?: string;
    activeUrlIncludes?: string;
  }): boolean {
    if (pItem.activeRouteId) {
      return page.route.id === pItem.activeRouteId;
    }

    if (pItem.activeRoutePrefix) {
      return !!page.route.id?.startsWith(pItem.activeRoutePrefix);
    }

    if (pItem.activeUrlIncludes) {
      return page.url?.hash?.includes(pItem.activeUrlIncludes) ?? false;
    }

    return false;
  }
</script>

{#snippet itemIcon(IconComponent: Component)}
  <IconComponent class={cIconClassDefault} />
{/snippet}

{#if isSearchModeActive}
  <div class="px-4 my-2">
    <SearchInput
      bind:ref={vSearchInputRef}
      bind:value={searchQuery}
      onClose={fHandleSearchModeDeactivate}
      onKeyDown={(pE) => pE.key === 'Escape' && fHandleSearchModeDeactivate()}
      placeholder={fTranslate('text_e90dd0173105')}
    />
  </div>
{:else if isExpandedMode || cIsOnMobile}
  <div
    class="{className} flex flex-col gap-5 md:gap-1 mt-2 md:mt-0 {!isExpandedMode && cIsOnMobile
      ? 'hidden pointer-events-none'
      : ''}"
  >
    {#each actions as vItem, vI (vItem.tooltip)}
      {@const cIsActive = fIsItemActive(vItem)}
      {@const cIsSearchOnMobile = vItem.icon === Search && cDeviceStore.isMobile}
      {@const cItemHref = cIsSearchOnMobile ? cRoutes.SEARCH : vItem.route}
      {@const cItemOnClick = fItemClickHandler(vItem, cIsSearchOnMobile)}
      {@const cItemTransition = {
        delay: !vInitialized ? vI * cIconStripTransitionDelayMultiplier : 0,
        duration: cIconStripTransitionDuration,
        easing: circIn
      }}

      {#if vShowIcons}
        <div transition:fade={cItemTransition}>
          <Button
            class="w-full min-w-9 justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {cIsActive
              ? 'bg-accent text-accent-foreground'
              : ''}"
            href={cItemHref}
            onclick={cItemOnClick}
            size="default"
            variant="ghost"
          >
            <span class="flex min-w-0 items-center px-0.5 gap-2">
              {@render itemIcon(vItem.icon)}

              {#if vShowIcons}
                <span in:fade={cItemTransition} out:fade={cItemTransition} class="min-w-0 truncate"
                  >{vItem.tooltip}</span
                >
              {/if}
            </span>

            {#if vItem.keys}
              <KeyboardShortcutInfo keys={vItem.keys} />
            {/if}
          </Button>
        </div>
      {/if}
    {/each}
  </div>
{:else}
  <div class="{className} flex-col gap-1 hidden md:flex">
    {#each actions as vItem, vI (vItem.tooltip)}
      {@const cIsActive = fIsItemActive(vItem)}
      {@const cIsSearchOnMobile = vItem.icon === Search && cDeviceStore.isMobile}
      {@const cItemOnClick = fItemClickHandler(vItem, cIsSearchOnMobile)}
      {@const cItemTransition = {
        delay: !vInitialized ? vI * cIconStripTransitionDelayMultiplier : 0,
        duration: cIconStripTransitionDuration,
        easing: circIn
      }}

      {#if vShowIcons}
        <div transition:fade={cItemTransition}>
          <ActionIcon
            class="h-9 w-9 rounded-full hover:bg-accent! {cIsActive
              ? 'bg-accent text-accent-foreground'
              : ''}"
            icon={vItem.icon}
            iconSize={cIconClassDefault}
            onclick={cItemOnClick}
            size="lg"
            tooltip={vItem.tooltip}
            tooltipSide={TooltipSide.RIGHT}
          />
        </div>
      {/if}
    {/each}
  </div>
{/if}
