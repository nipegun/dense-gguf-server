<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ChatTabsItem from './ChatTabsItem.svelte';
  import ChatTabsNewChatButton from './ChatTabsNewChatButton.svelte';
  import { page } from '$app/state';
  import { ScrollCarousel } from '$lib/components/app';
  import {
    cChatTabsMaxWidth,
    cNewChatLabel,
    cNewChatTabId,
    cUiDataAttrs,
    cUnnamedChatLabel
  } from '$lib/constants';
  import { fUseScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
  import { cChatStore, cConversationsStore, cTabsStore, cUiStore } from '$lib/stores';
  import { tick } from 'svelte';

  const cCarousel = fUseScrollCarousel();

  let vActiveId = $derived(page.params.id ?? cNewChatTabId);

  let aTabs = $derived(
    cTabsStore.aOpenTabs.map((pId) => ({
      id: pId,
      isNewChat: pId === cNewChatTabId,
      name:
        pId === cNewChatTabId
          ? cNewChatLabel
          : (cConversationsStore.aConversations.find((pC) => pC.id === pId)?.name ??
            cUnnamedChatLabel)
    }))
  );

  // hide the New chat button while a new-chat tab is already open
  let vShowNewChatButton = $derived(!cTabsStore.aOpenTabs.includes(cNewChatTabId));

  let sLoadingIds = $derived(new Set(cChatStore.mGetAllLoadingChats()));

  function fHandleClose(pId: string) {
    void cTabsStore.mClose(pId, vActiveId);
  }

  function fHandleStop(pId: string, pEvent: MouseEvent) {
    pEvent.stopPropagation();
    void cChatStore.mStopGenerationForChat(pId);
  }

  function fHandleAuxClick(pId: string, pEvent: MouseEvent) {
    // middle-click closes, like browser tabs
    if (pEvent.button === 1) {
      pEvent.preventDefault();
      fHandleClose(pId);
    }
  }

  let sPreviousTabIds = new Set<string>();
  let vPreviousActiveId: string | null = null;

  $effect(() => {
    const sCurrentIds = new Set(aTabs.map((pT) => pT.id));
    const cHasAddedTab = aTabs.some((pT) => !sPreviousTabIds.has(pT.id));

    sPreviousTabIds = sCurrentIds;

    const cActiveChanged = vActiveId !== vPreviousActiveId;

    vPreviousActiveId = vActiveId;

    // scroll when the active tab changes (a click) or when a new tab is added
    if (!cHasAddedTab && !cActiveChanged) return;

    // wait for the new tab to be laid out before scrolling to it
    void tick().then(() => {
      const cEl = cCarousel.scrollContainer?.querySelector<HTMLElement>(
        `[${cUiDataAttrs.ACTIVE_TAB}]`
      );

      if (cEl) {
        cCarousel.scrollToCenter(cEl);
      }
    });
  });
</script>

<nav
  aria-label={fTranslate('dgs_f93298c2be57')}
  class="group sticky pl-1 top-0 z-10 hidden md:block chat-tabs-fade transition-[padding] duration-200 ease-in-out pt-3.25 {cUiStore.vIsSidebarExpanded
    ? cChatTabsMaxWidth.EXPANDED_SIDEBAR
    : cChatTabsMaxWidth.COLLAPSED_SIDEBAR}"
>
  <div class="relative">
    <ScrollCarousel
      carousel={cCarousel}
      class="h-10"
      containerClass="flex h-10 min-w-0 items-center"
      innerClass="items-center gap-1.25"
    >
      {#each aTabs as vTab (vTab.id)}
        <ChatTabsItem
          isActive={vTab.id === vActiveId}
          isLoading={sLoadingIds.has(vTab.id)}
          onActivate={(pId) => cTabsStore.mActivate(pId)}
          onAuxClick={fHandleAuxClick}
          onClose={fHandleClose}
          onStop={fHandleStop}
          tab={vTab}
        />
      {/each}

      {#if vShowNewChatButton}
        <ChatTabsNewChatButton onclick={() => void cConversationsStore.mOpenNewChat()} />
      {/if}
    </ScrollCarousel>

    <div
      class="pointer-events-none absolute inset-y-0 left-0 z-[5] w-8 bg-gradient-to-r from-background to-transparent transition-opacity {cCarousel.canScrollLeft
        ? 'opacity-100'
        : 'opacity-0'}"
    ></div>

    <div
      class="pointer-events-none absolute inset-y-0 right-0 z-[5] w-8 bg-gradient-to-l from-background to-transparent transition-opacity {cCarousel.canScrollRight
        ? 'opacity-100'
        : 'opacity-0'}"
    ></div>
  </div>
</nav>

<style>
  .chat-tabs-fade {
    background: linear-gradient(
      to bottom,
      color-mix(in srgb, var(--background) 100%, transparent) 25%,
      color-mix(in srgb, var(--background) 80%, transparent) 50%,
      color-mix(in srgb, var(--background) 40%, transparent) 75%,
      transparent 100%
    );
  }
</style>
