<script lang="ts">
  import { goto, replaceState } from '$app/navigation';
  import { afterNavigate } from '$app/navigation';
  import { page } from '$app/state';
  import { DialogModelNotAvailable } from '$lib/components/app';
  import { cAppName, cRoutes, cUrlParams } from '$lib/constants';
  import { cChatStore, cConversationsStore, cModelsStore } from '$lib/stores';

  let vChatId = $derived(page.params.id);
  let vCurrentChatId: string | undefined = undefined;

  // URL parameters for prompt and model selection
  let vQParam = $derived(page.url.searchParams.get(cUrlParams.QUERY));
  let vModelParam = $derived(page.url.searchParams.get(cUrlParams.MODEL));

  // Dialog state for model not available error
  let vShowModelNotAvailable = $state(false);
  let vRequestedModelName = $state('');
  let aAvailableModelNames = $derived(cModelsStore.aModels.map((pM) => pM.model));

  // Track if URL params have been processed for this chat
  let vUrlParamsProcessed = $state(false);

  /**
   * Clear URL params after message is sent to prevent re-sending on refresh
   */
  function fClearUrlParams() {
    const url = new URL(page.url);

    url.searchParams.delete(cUrlParams.QUERY);
    url.searchParams.delete(cUrlParams.MODEL);
    replaceState(url.toString(), {});
  }

  async function fHandleUrlParams() {
    // Ensure models are loaded first
    await cModelsStore.mFetch();

    // Handle model parameter - select model if provided
    if (vModelParam) {
      const model = cModelsStore.mFindModelByName(vModelParam);

      if (model) {
        try {
          await cModelsStore.mSelectModelById(model.id);
        } catch (error) {
          console.error('Failed to select model:', error);
          vRequestedModelName = vModelParam;
          vShowModelNotAvailable = true;

          return;
        }
      } else {
        // Model not found - show error dialog
        vRequestedModelName = vModelParam;
        vShowModelNotAvailable = true;

        return;
      }
    }

    // Handle ?q= parameter - send message in current conversation
    if (vQParam !== null) {
      await cChatStore.mSendMessage(vQParam);
      // Clear URL params after message is sent
      fClearUrlParams();
    } else if (vModelParam) {
      // Clear params even if no message was sent (just model selection)
      fClearUrlParams();
    }

    vUrlParamsProcessed = true;
  }

  afterNavigate(() => {
    setTimeout(() => {
      void cModelsStore.mSelectModelFromLastAssistantResponse();
    }, 100);
  });

  $effect(() => {
    if (vChatId && vChatId !== vCurrentChatId) {
      vCurrentChatId = vChatId;
      vUrlParamsProcessed = false; // Reset for new chat

      // Skip loading if this conversation is already active (e.g., just created)
      if (cConversationsStore.vActiveConversation?.id === vChatId) {
        void cChatStore.mDiscoverActiveStream(vChatId);

        if ((vQParam !== null || vModelParam !== null) && !vUrlParamsProcessed) {
          fHandleUrlParams();
        }

        return;
      }

      (async () => {
        const success = await cConversationsStore.mLoadConversation(vChatId);

        if (!success) {
          await goto(cRoutes.START);

          return;
        }

        cChatStore.mSyncLoadingStateForChat(vChatId);
        // server probe (with localStorage fallback) and attach
        await cChatStore.mDiscoverActiveStream(vChatId);

        if ((vQParam !== null || vModelParam !== null) && !vUrlParamsProcessed) {
          await fHandleUrlParams();
        }
      })();
    }
  });

  $effect(() => {
    if (typeof window === 'undefined' || typeof document === 'undefined') return;

    // when the tab comes back to the foreground, re-run discovery to catch any race
    // where the initial mount probe missed an active session
    const fOnVisibility = () => {
      if (document.visibilityState !== 'visible') return;

      if (!vChatId) return;

      void cChatStore.mDiscoverActiveStream(vChatId);
    };

    document.addEventListener('visibilitychange', fOnVisibility);

    return () => document.removeEventListener('visibilitychange', fOnVisibility);
  });
</script>

<svelte:head>
  <title>{cConversationsStore.vActiveConversation?.name || 'Chat'} - {cAppName}</title>
</svelte:head>

<DialogModelNotAvailable
  bind:open={vShowModelNotAvailable}
  availableModels={aAvailableModelNames}
  modelName={vRequestedModelName}
/>
