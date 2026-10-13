<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelAvatar from '../ModelAvatar.svelte';
  import ModelCapabilities from '../ModelCapabilities.svelte';
  import ModelContext from '../ModelContext.svelte';
  import ModelDownloadProgressBar from '../ModelDownloadProgressBar.svelte';
  import ModelId from '../ModelId.svelte';
  import ModelsManagerStatusCell from './ModelsManagerStatusCell.svelte';
  import { fModelRowActions } from './row-actions';
  import { fConfiguredContext, fDownloadProgressFor } from './utils';
  import { Heart, MoreHorizontal } from '@lucide/svelte';
  import { DropdownMenuActions } from '$lib/components/app';
  import { cModelRowGridClass, cModelRowTrailingCellClass } from '$lib/constants';
  import { ModelRowDownloadState } from '$lib/enums';
  import { cDeviceStore, cModelsStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';

  interface Props {
    isFavorite: (pOption: ModelOption) => boolean;
    option: ModelOption;
    onDelete: (pOption: ModelOption) => void;
    onSelect: (pOption: ModelOption) => void;
    selected: boolean;
    /** Left padding in px, from the nesting depth. */
    indent?: number;
  }

  let { indent = 0, isFavorite, onDelete, onSelect, option, selected }: Props = $props();

  let vFavorite = $derived(isFavorite(option));
  let vIsHidden = $derived(cModelsStore.mIsHidden(option.id));
  // live while the download runs, frozen while it is paused
  let vDownloadProgress = $derived(fDownloadProgressFor(option.model));
  // a tracked download takes over the status column while it runs
  let vDownload = $derived(
    cModelsStore.status.mIsDownloadPaused(option.model)
      ? ModelRowDownloadState.cPaused
      : cModelsStore.status.mIsDownloadInProgress(option.model)
        ? ModelRowDownloadState.cDownloading
        : null
  );
</script>

<div
  class={[
    cModelRowGridClass,
    'group relative rounded-md px-2 py-3 transition max-md:px-3 max-md:py-4',
    vIsHidden && 'opacity-60',
    selected ? 'bg-accent text-accent-foreground' : 'hover:bg-muted/40'
  ]}
>
  <!-- the row holds a load control and an actions menu, so only the model itself is
	     the button: a button nested in a role="button" row is invalid -->
  <button
    aria-pressed={selected}
    class="flex min-w-0 cursor-pointer items-center gap-3 text-left max-md:gap-2 max-md:pr-9"
    onclick={() => onSelect(option)}
    style="padding-left: {indent}px"
    type="button"
  >
    <ModelAvatar {option} size="size-7 md:size-9" />

    <span class="flex min-w-0 items-center gap-1.25">
      <ModelId
        aliases={option.aliases}
        class="min-w-0 flex-1 max-md:text-sm"
        draftSidecars={option.draftSidecars}
        hideCapabilities
        hideModalities
        modalities={option.modalities}
        modelId={option.model}
        tags={option.tags}
        title={option.model}
      />

      <!-- a phone has no width for the modality icons, the id needs it more -->
      <ModelCapabilities hideModalities={cDeviceStore.isMobile} {option} />

      {#if vFavorite}
        <!-- the heart is decorative, the button's text carries the state -->
        <Heart aria-hidden="true" class="h-3.5 w-3.5 shrink-0 text-rose-500" />

        <span class="sr-only">{fTranslate('dgs_25b08effdd83')}</span>
      {/if}
    </span>
  </button>

  <ModelContext
    class="justify-self-end max-md:hidden"
    configured={fConfiguredContext(option)}
    {option}
  />

  <ModelsManagerStatusCell class="max-md:hidden" download={vDownload} {option} />

  {#if vDownload}
    <ModelDownloadProgressBar
      downloadedBytes={vDownloadProgress?.downloadedBytes ?? 0}
      overlay
      totalBytes={vDownloadProgress?.totalBytes ?? 0}
    />
  {/if}

  <div class="flex items-center justify-center justify-self-center {cModelRowTrailingCellClass}">
    <DropdownMenuActions
      actions={fModelRowActions(option, vFavorite, vIsHidden, onDelete, vDownload)}
      align="end"
      triggerIcon={MoreHorizontal}
      triggerTooltip="Model actions"
    />
  </div>
</div>
