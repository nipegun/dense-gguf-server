<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelsManagerModelConfigurationInformation from '$lib/components/app/models/ModelsManager/ModelsManagerModelConfiguration/ModelsManagerModelConfigurationInformation.svelte';
  import * as Dialog from '$lib/components/ui/dialog';
  import { ModelIcon } from '$lib/constants';
  import { HuggingFaceService } from '$lib/services';
  import { cModelsStore, cUiStore } from '$lib/stores';
  import type { HfModelDetailInfo } from '$lib/types/huggingface';
  import { fRepoOf } from '$lib/utils';

  // the props cache is a plain Map, so the read has to name its version to stay reactive
  let vServerProps = $derived.by(() => {
    void cModelsStore.props.vCacheVersion;

    return cUiStore.vModelInformation
      ? cModelsStore.props.mGetModelProps(cUiStore.vModelInformation.model)
      : null;
  });

  // The server reports the full metadata only once a model is loaded. With the
  // Hub enabled, it fills those gaps instead.
  let vHubDetails = $state<HfModelDetailInfo | null>(null);

  $effect(() => {
    const cOption = cUiStore.vModelInformation;

    if (!cOption) return;

    const cRepo = fRepoOf(cOption.model);

    let vCancelled = false;

    vHubDetails = null;

    if (!HuggingFaceService.mIsEnabled() || !cRepo.includes('/')) {
      return;
    }

    void HuggingFaceService.mGetDetails(cRepo)
      .then((pDetails) => {
        if (!vCancelled) vHubDetails = pDetails;
      })
      .catch(() => {});

    return () => {
      vCancelled = true;
    };
  });
</script>

<Dialog.fRoot
  onOpenChange={(pOpen) => {
    if (!pOpen) cUiStore.vModelInformation = null;
  }}
  open={cUiStore.vModelInformation !== null}
>
  <Dialog.Content
    class="z-[70] flex max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! max-md:border-0! md:h-[calc(100vh-4rem)]! md:w-[calc(100vw-4rem)]! md:max-w-md! flex-col"
    onOpenAutoFocus={(pEvent) => pEvent.preventDefault()}
    overlayClass="z-[70]"
  >
    <Dialog.Header class="flex flex-row items-center pr-8">
      <Dialog.Title class="flex min-w-0 items-center gap-2 text-base">
        <ModelIcon class="h-4 w-4 shrink-0 text-muted-foreground" />

        {#if cUiStore.vModelInformation}
          <span class="min-w-0 truncate">{cUiStore.vModelInformation.model}</span>
        {/if}
      </Dialog.Title>

      <Dialog.Description class="sr-only">
        {fTranslate('dgs_c6896e45b1f6')}
      </Dialog.Description>
    </Dialog.Header>

    <div class="min-h-0 flex-1 overflow-y-auto">
      {#if cUiStore.vModelInformation}
        <ModelsManagerModelConfigurationInformation
          hub={vHubDetails}
          option={cUiStore.vModelInformation}
          serverProps={vServerProps}
        />
      {/if}
    </div>
  </Dialog.Content>
</Dialog.fRoot>
