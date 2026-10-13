<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { browser } from '$app/environment';
  import { goto } from '$app/navigation';
  import { page } from '$app/state';
  import { SearchInput, SidebarNavigationSearchResults } from '$lib/components/app';
  import { cRoutes } from '$lib/constants';
  import { RouterService } from '$lib/services/router.service';
  import { cChatStore, cConversationsStore, cDeviceStore } from '$lib/stores';

  let vSearchQuery = $state('');
  let vSearchInputRef = $state<HTMLInputElement | null>(null);

  let vCurrentChatId = $derived(page.params.id);

  let aFilteredConversations = $derived.by(() => {
    const cQuery = vSearchQuery.trim().toLowerCase();

    if (cQuery.length === 0) return [];

    return cConversationsStore.aConversations.filter((pC) =>
      pC.name.toLowerCase().includes(cQuery)
    );
  });

  // Search page is intended for mobile; on desktop the sidebar already exposes
  // in-place search, so bounce back to a new-chat tab without a history entry.
  $effect(() => {
    if (browser && !cDeviceStore.isMobile) {
      goto(cRoutes.START, { replaceState: true });
    }
  });

  async function fSelectConversation(pId: string) {
    await goto(RouterService.mChat(pId));
  }

  async function fHandleEditConversation(pId: string) {
    const cConversation = cConversationsStore.aConversations.find((pC) => pC.id === pId);

    if (!cConversation) return;

    const cNewName = window.prompt('Rename conversation', cConversation.name);

    if (cNewName && cNewName.trim()) {
      await cConversationsStore.mUpdateConversationName(pId, cNewName.trim());
    }
  }

  async function fHandleDeleteConversation(pId: string) {
    const cConversation = cConversationsStore.aConversations.find((pC) => pC.id === pId);

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

  function fHandleBack() {
    if (history.length > 1) {
      history.back();
    } else {
      cConversationsStore.mOpenNewChat();
    }
  }
</script>

<svelte:head>
  <title>{fTranslate('dgs_74cecb1f9b3d')}</title>
</svelte:head>

<div class="fixed top-0 z-10 left-0 right-0 p-2">
  <SearchInput
    bind:ref={vSearchInputRef}
    bind:value={vSearchQuery}
    autofocus
    onClose={fHandleBack}
    placeholder={fTranslate('text_e90dd0173105')}
  />
</div>

<div class="p-2 pt-16">
  <SidebarNavigationSearchResults
    currentChatId={vCurrentChatId}
    filteredConversations={aFilteredConversations}
    onDelete={fHandleDeleteConversation}
    onEdit={fHandleEditConversation}
    onSelect={fSelectConversation}
    onStop={fHandleStopGeneration}
    searchQuery={vSearchQuery}
  />
</div>
