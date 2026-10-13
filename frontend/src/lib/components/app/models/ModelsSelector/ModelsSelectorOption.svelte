<script lang="ts">
  import ModelLoadHighlight from '../ModelLoadHighlight.svelte';
  import {
    ModelAvatar,
    ModelCapabilities,
    ModelId,
    ModelLoadControl,
    ModelRowActions
  } from '$lib/components/app';
  import { cSettingsKeys } from '$lib/constants';
  import { ServerModelStatus } from '$lib/enums';
  import { cModelsStore, cSettingsStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';
  import { fModelLoadFraction, fModelLoadProgressText } from '$lib/utils';

  interface Props {
    option: ModelOption;
    isSelected: boolean;
    isHighlighted: boolean;
    isFav: boolean;
    hideOrgName?: boolean;
    onSelect: (pModelId: string) => void;
    onMouseEnter: () => void;
    onKeyDown: (pE: KeyboardEvent) => void;
  }

  let {
    hideOrgName = false,
    isFav,
    isHighlighted,
    isSelected,
    onKeyDown,
    onMouseEnter,
    onSelect,
    option
  }: Props = $props();

  let vServerStatus = $derived(cModelsStore.mGetModelStatus(option.model));
  let vIsOperationInProgress = $derived(cModelsStore.status.mIsOperationInProgress(option.model));
  let vIsFailed = $derived(vServerStatus === ServerModelStatus.FAILED);
  let vIsSleeping = $derived(vServerStatus === ServerModelStatus.cSleeping);
  let vIsLoaded = $derived(cModelsStore.mIsModelRunning(option.model));
  let isLoading = $derived(vServerStatus === ServerModelStatus.LOADING || vIsOperationInProgress);

  let vLoadProgress = $derived(
    isLoading ? cModelsStore.status.mGetLoadProgress(option.model) : null
  );
  let vLoadPercent = $derived(Math.round(fModelLoadFraction(vLoadProgress) * 100));
  let vLoadTitle = $derived(fModelLoadProgressText(vLoadProgress));
  let vModalities = $derived(option.modalities);
  let vShowCapabilities = $derived(
    cSettingsStore.vConfig[cSettingsKeys.SHOW_MODEL_CAPABILITIES_IN_SELECTOR] ?? false
  );
</script>

<div
  aria-selected={isSelected || isHighlighted}
  class={[
    'group relative flex w-full items-center gap-2 rounded-sm p-2 text-left text-sm transition focus:outline-none',
    'cursor-pointer',
    // skip layout and paint for rows scrolled out of the long lists
    '[content-visibility:auto] [contain-intrinsic-size:auto_2.25rem] max-md:[contain-intrinsic-size:auto_3rem]',
    // a phone gives the row and its tappable area room to breathe
    'max-md:gap-2.5 max-md:px-2.5 max-md:py-2.5',
    isSelected && !isHighlighted && 'bg-accent/50',
    isHighlighted && 'bg-accent',
    (isSelected || isHighlighted) && 'text-accent-foreground',
    'hover:bg-accent',
    'focus:bg-accent',
    vIsLoaded ? 'text-popover-foreground' : 'text-muted-foreground'
  ]}
  onclick={() => onSelect(option.id)}
  onkeydown={onKeyDown}
  onmouseenter={onMouseEnter}
  role="option"
  tabindex="0"
  title={vLoadTitle}
>
  <ModelAvatar {option} size="size-5 max-md:size-7" />

  <ModelId
    aliases={option.aliases}
    class="min-w-0 flex-1"
    draftSidecars={option.draftSidecars}
    hideCapabilities
    hideModalities
    {hideOrgName}
    modalities={vModalities}
    modelId={option.model}
    tags={option.tags}
    title={option.model}
  />

  {#if vShowCapabilities}
    <ModelCapabilities {option} />
  {/if}

  <div class="flex shrink-0 items-center gap-1 max-md:gap-2.5">
    <ModelRowActions {isFav} {option} />

    <ModelLoadControl
      isFailed={vIsFailed}
      isLoaded={vIsLoaded}
      {isLoading}
      isSleeping={vIsSleeping}
      {option}
    />
  </div>

  {#if isLoading}
    <ModelLoadHighlight percent={vLoadPercent} />
  {/if}
</div>
