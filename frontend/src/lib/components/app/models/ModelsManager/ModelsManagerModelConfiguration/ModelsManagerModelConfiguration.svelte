<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelsManagerModelConfigurationActions from './ModelsManagerModelConfigurationActions.svelte';
  import ModelsManagerModelConfigurationHeader from './ModelsManagerModelConfigurationHeader.svelte';
  import ModelsManagerModelConfigurationInformation from './ModelsManagerModelConfigurationInformation.svelte';
  import { Trash2 } from '@lucide/svelte';
  import { DialogConfirmDownload } from '$lib/components/app/dialogs';
  import { Button } from '$lib/components/ui/button';
  import { ModelDownloadConfirmAction, ServerModelStatus } from '$lib/enums';
  import { HuggingFaceService } from '$lib/services';
  import { cDeviceStore, cModelsStore } from '$lib/stores';
  import type { HfModelDetailInfo } from '$lib/types/huggingface';
  import type { ModelOption } from '$lib/types/models';
  import { fRepoOf } from '$lib/utils';

  interface Props {
    onClose: () => void;
    onToggleLoad: () => void;
    onUseInChat: () => void;
    onUseInNewChat: () => void;
    option: ModelOption;
  }

  let { onClose, onToggleLoad, onUseInChat, onUseInNewChat, option }: Props = $props();

  // the same removal the table row offers, confirmed by the same dialog
  let vDeleteOpen = $state(false);

  // the props cache is a plain Map, so the read has to name its version to stay reactive
  let vServerProps = $derived.by(() => {
    void cModelsStore.props.vCacheVersion;

    return cModelsStore.props.mGetModelProps(option.model);
  });
  let status = $derived(cModelsStore.mGetModelStatus(option.model));
  let vIsLoaded = $derived(cModelsStore.mIsModelRunning(option.model));

  // The server reports the full metadata only once a model is loaded, which loads it.
  // With the Hub enabled, it fills those gaps instead.
  let vHubDetails = $state<HfModelDetailInfo | null>(null);

  $effect(() => {
    const cRepo = fRepoOf(option.model);

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

<div class="flex h-full min-h-0 flex-col">
  <ModelsManagerModelConfigurationHeader
    isLoaded={vIsLoaded}
    {onClose}
    {onToggleLoad}
    {onUseInChat}
    {onUseInNewChat}
    {option}
    {status}
  />

  <div class="min-h-0 flex-1 overflow-y-auto py-4 pl-4 max-md:pr-4">
    <ModelsManagerModelConfigurationInformation
      hub={vHubDetails}
      {option}
      serverProps={vServerProps}
    />

    <div class="mt-4 border-t border-border/30 pt-3 pb-2">
      <Button
        class="w-full justify-start gap-2 text-destructive hover:bg-destructive/10 hover:text-destructive dark:hover:bg-destructive/20"
        onclick={() => (vDeleteOpen = true)}
        size="sm"
        variant="ghost"
      >
        <Trash2 class="h-4 w-4" />
        {fTranslate('dgs_ba8f231bbc12')}
      </Button>
    </div>
  </div>

  <!-- a phone keeps the actions at the bottom of its screen, in reach of the thumb -->
  {#if cDeviceStore.isMobile}
    <div class="shrink-0 border-t border-border/40 px-4 py-3">
      <ModelsManagerModelConfigurationActions
        isLoaded={vIsLoaded}
        isLoading={status === ServerModelStatus.LOADING}
        {onToggleLoad}
        {onUseInChat}
        {onUseInNewChat}
      />
    </div>
  {/if}

  <DialogConfirmDownload
    action={ModelDownloadConfirmAction.DELETE}
    onClose={() => (vDeleteOpen = false)}
    open={vDeleteOpen}
    repoWithTag={option.model}
  />
</div>
