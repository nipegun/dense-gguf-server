<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { PanelLeftClose, PanelLeftOpen, X } from '@lucide/svelte';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import {
    ActionIcon,
    DialogConversationRename,
    DialogManageModels,
    DialogMcpServers,
    DialogModelInformation,
    DialogSettingsChat,
    Logo,
    SidebarNavigationActions,
    SidebarNavigationConversationList
  } from '$lib/components/app';
  import { cRoutes } from '$lib/constants';
  import { TooltipSide } from '$lib/enums';
  import { fUseKeyboardShortcuts } from '$lib/hooks/use-keyboard-shortcuts.svelte';
  import { fUseMarqueeSelection } from '$lib/hooks/use-marquee-selection.svelte';
  import { RouterService } from '$lib/services/router.service';
  import {
    cChatStore,
    cConversationsStore,
    cDeviceStore,
    cSettingsStore,
    cUiStore
  } from '$lib/stores';
  import { fBuildConversationTree } from '$lib/utils';
  import { circIn } from 'svelte/easing';
  import { SvelteSet } from 'svelte/reactivity';
  import { fade } from 'svelte/transition';

  interface Props {
    onSearchClick?: () => void;
  }

  let { onSearchClick = () => {} }: Props = $props();

  const { handleKeydown: fHandleKeydown } = fUseKeyboardShortcuts({
    activateSearchMode: () => onSearchClick(),
    toggleSidebar: () => fToggleExpandedMode()
  });

  let vHoveredTooltip = $state<string | null>(null);
  let vLogoHovered = $state(false);

  const cIsStripExpanded = $derived(cUiStore.vIsSidebarExpanded || vHoveredTooltip !== null);
  const cIsOnMobile = $derived(cDeviceStore.isMobile);
  const cAlwaysShowOnDesktop = $derived(
    cSettingsStore.vConfig.alwaysShowSidebarOnDesktop as boolean
  );

  $effect(() => {
    if (cAlwaysShowOnDesktop && !cIsOnMobile) {
      cUiStore.vIsSidebarExpanded = true;
    }
  });

  function fToggleExpandedMode() {
    cUiStore.vIsSidebarExpanded = !cUiStore.vIsSidebarExpanded;

    if (!cUiStore.vIsSidebarExpanded) {
      vHoveredTooltip = null;
    }
  }

  $effect(() => {
    if (!cUiStore.vIsSidebarExpanded) {
      vIsSearchModeActive = false;
      vSearchQuery = '';

      if (vIsSelectionMode) fExitSelectionMode();

      fCancelMobileCollapse();
    }
  });

  $effect(() => {
    if (cDeviceStore.isMobile && page.url.hash.includes(cRoutes.SEARCH)) {
      cUiStore.vIsSidebarExpanded = false;
    }
  });

  let vCurrentChatId = $derived(page.params.id);
  let vIsSearchModeActive = $state(false);
  let vSearchQuery = $state('');

  let aFilteredConversations = $derived.by(() => {
    if (vIsSearchModeActive) {
      if (vSearchQuery.trim().length > 0) {
        return cConversationsStore.aConversations.filter((pConversation: { name: string }) =>
          pConversation.name.toLowerCase().includes(vSearchQuery.toLowerCase())
        );
      }

      return [];
    }

    return cConversationsStore.aConversations;
  });

  let vIsSelectionMode = $state(false);
  let vSelectedIds = new SvelteSet<string>();

  let vRenameDialogOpen = $state(false);
  let vSettingsDialogOpen = $state(false);
  let vMcpDialogOpen = $state(false);
  let vRenameTargetConversationId = $state<string | null>(null);
  let vRenameDraft = $state('');
  let vRenameOriginalTitle = $state('');

  const aRenderedOrderIds = $derived(
    fBuildConversationTree(aFilteredConversations).map((pT) => pT.conversation.id)
  );

  const cAllSelectedArePinned = $derived.by(() => {
    if (vSelectedIds.size === 0) return false;

    const aConvs = cConversationsStore.aConversations;

    for (const cId of vSelectedIds) {
      const c = aConvs.find((pConv) => pConv.id === cId);

      if (c && !c.pinned) return false;
    }

    return true;
  });

  const cPinStateIsMixed = $derived.by(() => {
    if (vSelectedIds.size === 0) return false;

    const aConvs = cConversationsStore.aConversations;

    let vAnyPinned = false;
    let vAnyUnpinned = false;

    for (const cId of vSelectedIds) {
      const c = aConvs.find((pConv) => pConv.id === cId);

      if (!c) continue;

      if (c.pinned) vAnyPinned = true;
      else vAnyUnpinned = true;

      if (vAnyPinned && vAnyUnpinned) return true;
    }

    return false;
  });

  const cVisibleSelectionStats = $derived.by(() => {
    const aVisibleIds = aFilteredConversations.map((pC) => pC.id);

    let vSelectedVisible = 0;

    for (const cId of aVisibleIds) {
      if (vSelectedIds.has(cId)) vSelectedVisible++;
    }

    return {
      selectedVisibleCount: vSelectedVisible,
      visibleCount: aVisibleIds.length
    };
  });

  function fEnterSelectionMode(pId?: string) {
    vIsSelectionMode = true;

    if (pId !== undefined) {
      vSelectedIds.add(pId);
    }
  }

  function fExitSelectionMode() {
    vIsSelectionMode = false;
    vSelectedIds.clear();
  }

  function fToggleSelected(pId: string) {
    if (vSelectedIds.has(pId)) {
      vSelectedIds.delete(pId);
    } else {
      vSelectedIds.add(pId);
    }
  }

  function fToggleSelectAllVisible() {
    const aVisibleIds = aFilteredConversations.map((pC) => pC.id);
    const cAllSelected =
      aVisibleIds.length > 0 && aVisibleIds.every((pId) => vSelectedIds.has(pId));

    if (cAllSelected) {
      for (const cId of aVisibleIds) vSelectedIds.delete(cId);
    } else {
      for (const cId of aVisibleIds) vSelectedIds.add(cId);
    }
  }

  async function fHandleBulkDelete() {
    const aIds = Array.from(vSelectedIds);

    if (aIds.length === 0) return;

    await cConversationsStore.mBulkDeleteConversations(aIds);
    fExitSelectionMode();
  }

  async function fHandleBulkPinToggle() {
    const aIds = Array.from(vSelectedIds);

    if (aIds.length === 0) return;

    await cConversationsStore.mBulkToggleConversationPin(aIds);
  }

  async function fHandleBulkExport() {
    const aIds = Array.from(vSelectedIds);

    if (aIds.length === 0) return;

    await cConversationsStore.mBulkExportConversations(aIds);
  }

  const cMarquee = fUseMarqueeSelection({
    enabled: () => vIsSelectionMode,
    orderedIds: () => aRenderedOrderIds,
    selectedIds: () => vSelectedIds
  });

  function fHandleRowMouseDown(pId: string, pEvent: MouseEvent) {
    if (!vIsSelectionMode) return;

    cMarquee.rowMouseDown(pId, pEvent);
  }

  function fHandleSelectionClick(pId: string, pOptions: { shiftKey: boolean }): void {
    if (!vIsSelectionMode) return;

    cMarquee.rowClick(pId, pOptions.shiftKey);
  }

  async function fSelectConversation(pId: string) {
    if (cDeviceStore.isMobile) {
      fScheduleMobileCollapse();
    }

    await goto(RouterService.mChat(pId));
  }

  async function fHandleEditConversation(pId: string) {
    const cConversation = cConversationsStore.aConversations.find((pConv) => pConv.id === pId);

    if (!cConversation) return;

    vRenameTargetConversationId = pId;
    vRenameOriginalTitle = cConversation.name;
    vRenameDraft = cConversation.name;
    vRenameDialogOpen = true;
  }

  async function fHandleRenameConfirm() {
    const cId = vRenameTargetConversationId;

    if (!cId) return;

    const cNextName = vRenameDraft.trim();

    if (!cNextName || cNextName === vRenameOriginalTitle.trim()) return;

    await cConversationsStore.mUpdateConversationName(cId, cNextName);

    vRenameDialogOpen = false;
    vRenameTargetConversationId = null;
  }

  function fHandleRenameCancel() {
    vRenameDialogOpen = false;
    vRenameTargetConversationId = null;
    vRenameDraft = '';
    vRenameOriginalTitle = '';
  }

  async function fHandleDeleteConversation(pId: string) {
    const cConversation = cConversationsStore.aConversations.find((pConv) => pConv.id === pId);

    if (!cConversation) return;

    const cConfirmed = window.confirm(
      `Delete "${cConversation.name}"? This action cannot be undone.`
    );

    if (!cConfirmed) return;

    await cConversationsStore.mDeleteConversation(pId, { deleteWithForks: false });
  }

  function fHandleStopGeneration(pId: string) {
    cChatStore.mStopGenerationForChat(pId);
  }

  let vInnerWidth = $state(0);
  let vPendingCollapse = $state<ReturnType<typeof setTimeout> | null>(null);

  function fScheduleMobileCollapse() {
    if (vPendingCollapse) {
      clearTimeout(vPendingCollapse);
    }

    vPendingCollapse = setTimeout(() => {
      cUiStore.vIsSidebarExpanded = false;
      vPendingCollapse = null;
    }, 100);
  }

  function fCancelMobileCollapse() {
    if (vPendingCollapse) {
      clearTimeout(vPendingCollapse);
      vPendingCollapse = null;
    }
  }

  /** A dialog owns the screen on a phone, so the expanded list steps aside for it. */
  function fOpenDialog(pOpen: () => void): void {
    if (cDeviceStore.isMobile) {
      fScheduleMobileCollapse();
    }

    pOpen();
  }
</script>

<svelte:window bind:innerWidth={vInnerWidth} onkeydown={fHandleKeydown} />

{#if vInnerWidth > 768 || !page.url.hash.includes(cRoutes.SEARCH)}
  <aside
    class={[
      'fixed md:sticky top-2 left-2 md:left-0 md:ml-2 md:mt-2 pt-2 z-10 w-[calc(100dvw-1rem)]',
      'md:h-[calc(100dvh-var(--status-bar-height,0px)-1.125rem)]',
      cUiStore.vIsSidebarExpanded &&
        (cDeviceStore.vIsStandalone
          ? 'h-[calc(100dvh-var(--status-bar-height,0px)-2rem)]'
          : cDeviceStore.vIsIOSDevice
            ? 'h-[calc(100dvh-var(--status-bar-height,0px)-0.5rem)]'
            : 'h-[calc(100dvh-var(--status-bar-height,0px)-1rem)]'),
      'rounded-3xl md:rounded-2xl',
      'flex flex-col justify-between',
      'md:transition-[width,padding] duration-200 ease-out',
      cIsStripExpanded && 'md:w-72 md:bg-muted/60 md:backdrop-blur-xl shadow-md',
      !cIsStripExpanded && 'md:w-12',
      cUiStore.vIsSidebarExpanded && 'is-expanded'
    ]}
  >
    <div class="px-2 flex items-center justify-between">
      <div
        class="relative"
        onmouseenter={() => (vLogoHovered = true)}
        onmouseleave={() => (vLogoHovered = false)}
        role="button"
        tabindex="0"
      >
        <ActionIcon
          ariaLabel={cUiStore.vIsSidebarExpanded ? 'Go to start' : 'Expand navigation'}
          class="{cUiStore.vIsSidebarExpanded
            ? 'bg-muted! md:bg-foreground/5!'
            : 'bg-transparent!'} md:h-9 md:w-9 h-10 w-10 rounded-full md:hover:bg-foreground/10! pointer-events-auto"
          href={cUiStore.vIsSidebarExpanded ? cRoutes.START : undefined}
          icon={!cUiStore.vIsSidebarExpanded && vLogoHovered && vInnerWidth > 768
            ? PanelLeftOpen
            : Logo}
          iconSize="h-4.5 w-4.5 md:h-4 md:w-4"
          onclick={cUiStore.vIsSidebarExpanded ? undefined : fToggleExpandedMode}
          size="lg"
          tooltip={cUiStore.vIsSidebarExpanded ? undefined : 'Open Sidebar'}
          tooltipSide={TooltipSide.RIGHT}
        />
      </div>

      {#if cIsOnMobile || (cUiStore.vIsSidebarExpanded && !cAlwaysShowOnDesktop)}
        <div
          in:fade={{ delay: 50, duration: 150, easing: circIn }}
          out:fade={{ duration: 100 }}
          class="flex items-center transition-all duration-150 ease-out {cDeviceStore.isMobile &&
          !cUiStore.vIsSidebarExpanded
            ? 'opacity-0 h-0!'
            : ''}"
        >
          <ActionIcon
            ariaLabel="Collapse navigation"
            class="backdrop-blur-none md:h-9 md:w-9 h-10 w-10 rounded-full mr-1 max-md:-mr-1 max-md:-mt-1 hover:bg-accent!"
            icon={cDeviceStore.isMobile ? X : PanelLeftClose}
            iconSize="h-4 w-4"
            onclick={fToggleExpandedMode}
            size="lg"
            tooltip={fTranslate('text_8896c14d09e7')}
            tooltipSide={TooltipSide.LEFT}
          />
        </div>
      {/if}
    </div>

    <div
      in:fade={{ duration: 200 }}
      out:fade={{ duration: 200 }}
      class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-1 {cDeviceStore.isMobile
        ? 'transition-[opacity,height] duration-200 ease-out'
        : ''} {cDeviceStore.isMobile && !cUiStore.vIsSidebarExpanded ? 'opacity-0 !h-0' : ''}"
    >
      <SidebarNavigationActions
        bind:isSearchModeActive={vIsSearchModeActive}
        bind:searchQuery={vSearchQuery}
        class="px-2"
        isExpandedMode={vInnerWidth > 768 ? cUiStore.vIsSidebarExpanded : true}
        onManageModelsClick={() => fOpenDialog(() => cUiStore.mOpenModelsManager())}
        onMcpClick={() => fOpenDialog(() => (vMcpDialogOpen = true))}
        onNewChat={() => {
          if (cDeviceStore.isMobile) {
            fScheduleMobileCollapse();
          }
        }}
        onSearchClick={() => {
          cUiStore.vIsSidebarExpanded = true;
          vIsSearchModeActive = true;
        }}
        onSearchDeactivated={() => {
          vIsSearchModeActive = false;
          vSearchQuery = '';
        }}
        onSettingsClick={() => fOpenDialog(() => (vSettingsDialogOpen = true))}
      />

      {#if cUiStore.vIsSidebarExpanded || cIsOnMobile}
        <div class="flex min-h-0 flex-1 flex-col overflow-y-auto">
          <SidebarNavigationConversationList
            allSelectedArePinned={cAllSelectedArePinned}
            allVisibleSelected={cVisibleSelectionStats.visibleCount > 0 &&
              cVisibleSelectionStats.selectedVisibleCount === cVisibleSelectionStats.visibleCount}
            class="px-2"
            currentChatId={vCurrentChatId}
            filteredConversations={aFilteredConversations}
            isSearchModeActive={vIsSearchModeActive}
            isSelectionMode={vIsSelectionMode}
            onBulkDelete={fHandleBulkDelete}
            onBulkExport={fHandleBulkExport}
            onBulkPinToggle={fHandleBulkPinToggle}
            onCloseSelection={fExitSelectionMode}
            onDelete={fHandleDeleteConversation}
            onEdit={fHandleEditConversation}
            onEnterSelectionMode={fEnterSelectionMode}
            onRowMouseDown={fHandleRowMouseDown}
            onSelect={fSelectConversation}
            onSelectAllToggle={fToggleSelectAllVisible}
            onSelectionClick={fHandleSelectionClick}
            onStop={fHandleStopGeneration}
            onToggleSelect={fToggleSelected}
            pinStateIsMixed={cPinStateIsMixed}
            searchQuery={vSearchQuery}
            selectedIds={vSelectedIds}
            someVisibleSelected={cVisibleSelectionStats.selectedVisibleCount > 0 &&
              cVisibleSelectionStats.selectedVisibleCount < cVisibleSelectionStats.visibleCount}
            visibleCount={cVisibleSelectionStats.visibleCount}
          />
        </div>
      {/if}
    </div>
  </aside>
{/if}

<DialogConversationRename
  bind:open={vRenameDialogOpen}
  bind:value={vRenameDraft}
  currentTitle={vRenameOriginalTitle}
  onCancel={fHandleRenameCancel}
  onConfirm={fHandleRenameConfirm}
/>

<DialogSettingsChat bind:open={vSettingsDialogOpen} />

<DialogMcpServers bind:open={vMcpDialogOpen} />

<DialogManageModels bind:open={cUiStore.vManageModelsOpen} />

<DialogModelInformation />

<style>
  aside {
    @media (max-width: 768px) {
      --size: 1.125rem;
    }
  }

  @media (max-width: 768px) {
    aside {
      &:not(.is-expanded) {
        pointer-events: none;
      }
    }

    aside.is-expanded::before {
      content: '';
      position: fixed;
      top: -0.5rem;
      bottom: -0.25rem;
      left: -0.5rem;
      right: -0.5rem;
      z-index: -1;
      background: var(--background);
      backdrop-filter: blur(1rem);
      pointer-events: none;
    }
  }
</style>
