<script lang="ts">
  import { replaceState } from '$app/navigation';
  import { page } from '$app/state';
  import { DialogModelNotAvailable } from '$lib/components/app';
  import { cAppName, cUrlParams } from '$lib/constants';
  import { cConversationsStore, cModelsStore, cServerStore } from '$lib/stores';
  import { onMount } from 'svelte';

  let vQParam = $derived(page.url.searchParams.get(cUrlParams.QUERY));
  let vModelParam = $derived(page.url.searchParams.get(cUrlParams.MODEL));
  let vLoadParam = $derived(page.url.searchParams.get(cUrlParams.LOAD));

  let vShowModelNotAvailable = $state(false);
  let vRequestedModelName = $state('');
  let aAvailableModelNames = $derived(cModelsStore.aModels.map((pM) => pM.model));

  // Clear params after handling the deep link so a refresh does not replay them
  function fClearUrlParams() {
    const url = new URL(page.url);

    url.searchParams.delete(cUrlParams.QUERY);
    url.searchParams.delete(cUrlParams.MODEL);
    url.searchParams.delete(cUrlParams.LOAD);

    replaceState(url.toString(), {});
  }

  async function fHandleUrlParams() {
    await cModelsStore.mFetch();

    if (vModelParam) {
      const model = cModelsStore.mFindModelByName(vModelParam);

      if (model) {
        try {
          await cModelsStore.mSelectModelById(model.id);

          // with ?load=true in router mode, start loading right away so the
          // model is ready sooner; not awaited so the UI stays usable
          if (
            vLoadParam === 'true' &&
            cServerStore.isRouterMode &&
            !cModelsStore.mIsModelLoaded(model.id)
          ) {
            cModelsStore.status
              .load(model.id)
              .catch((pError) => console.error('Failed to load model:', pError));
          }
        } catch (error) {
          console.error('Failed to select model:', error);
          vRequestedModelName = vModelParam;
          vShowModelNotAvailable = true;

          return;
        }
      } else {
        vRequestedModelName = vModelParam;
        vShowModelNotAvailable = true;

        return;
      }
    }

    // ?q= creates the conversation, the chat route sends the prompt once the
    // conversation id is in the URL
    if (vQParam !== null) {
      await cConversationsStore.mCreateConversation();
      fClearUrlParams();
    } else if (vModelParam) {
      fClearUrlParams();
    }
  }

  onMount(async () => {
    if (!cConversationsStore.vIsInitialized) {
      await cConversationsStore.mInitialize();
    }

    cConversationsStore.mClearActiveConversation();

    await cModelsStore.mFetch();

    if (vQParam !== null || vModelParam !== null) {
      await fHandleUrlParams();
    }

    await cModelsStore.mEnsureFirstModelSelected();
  });
</script>

<svelte:head>
  <title>{cAppName}</title>
</svelte:head>

<DialogModelNotAvailable
  bind:open={vShowModelNotAvailable}
  availableModels={aAvailableModelNames}
  modelName={vRequestedModelName}
/>
