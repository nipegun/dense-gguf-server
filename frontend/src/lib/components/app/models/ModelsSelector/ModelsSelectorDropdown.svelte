<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelLoadHighlight from '../ModelLoadHighlight.svelte';
  import { ChevronDown, Loader2 } from '@lucide/svelte';
  import {
    DropdownMenuSearchable,
    ModelId,
    ModelsSelectorList,
    ModelsSelectorOption,
    ModelsSelectorTriggerIcon
  } from '$lib/components/app';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { ModelIcon } from '$lib/constants';
  import { KeyboardKey, ServerModelStatus } from '$lib/enums';
  import type { UseModelsSelectorReturn } from '$lib/hooks/use-models-selector.svelte';
  import { ModelsService } from '$lib/services/models.service';
  import { cModelsStore, cUiStore } from '$lib/stores';
  import type { ModelOption, ModelSidecarBadge } from '$lib/types/models';
  import type { ModelItem } from '$lib/utils';
  import { fModelLoadFraction, fRepoOf } from '$lib/utils';

  interface Props {
    /** The selector state this shell renders; the owner derives it once. */
    ms: UseModelsSelectorReturn;
    currentModel?: string | null;
    disabled?: boolean;
    /** Model id the list highlights from the keyboard; the owner tracks it. */
    highlightedId?: string | null;
    /** Bind the menu's open state, so the owner sees the close too. */
    open?: boolean;
    showOrgName?: boolean;
    onHighlight?: (pId: string | null) => void;
    onManageModels?: () => void;
    onModelKeyAction?: (pModelId: string, pUnload: boolean) => void;
    onSearchKeyDown?: (pEvent: KeyboardEvent) => void;
  }

  let {
    currentModel = null,
    disabled = false,
    highlightedId = null,
    ms,
    onHighlight,
    onManageModels,
    onModelKeyAction,
    onSearchKeyDown,
    open = $bindable(false),
    showOrgName = true
  }: Props = $props();

  const cSelectedOption = $derived(ms.mGetDisplayOption());
  const cTriggerModel = $derived(cSelectedOption?.model ?? null);

  let vTriggerStatus = $derived(
    cTriggerModel
      ? cModelsStore.aRouterModels.find((pM) => pM.id === cTriggerModel)?.status?.value
      : undefined
  );
  let vTriggerLoading = $derived(
    !!cTriggerModel &&
      (vTriggerStatus === ServerModelStatus.LOADING ||
        cModelsStore.status.mIsOperationInProgress(cTriggerModel))
  );
  let vTriggerLoadPercent = $derived(
    cTriggerModel && vTriggerLoading
      ? Math.round(fModelLoadFraction(cModelsStore.status.mGetLoadProgress(cTriggerModel)) * 100)
      : 0
  );

  /** Draft sidecar as it reads in the trigger tooltip, with its own quant. */
  function fDraftSidecarLabel(pBaseModel: string, pBadge: ModelSidecarBadge): string {
    const cBaseRepo = fRepoOf(pBaseModel);

    // a sidecar of the model's own repo reads as a bare tag, a foreign one keeps its id
    if (pBadge.repo === cBaseRepo) {
      return `${pBadge.kind.toUpperCase()}${pBadge.quant ? `:${pBadge.quant}` : ''}`;
    }

    return ModelsService.mBuildDownloadTag(pBadge.repo, pBadge.quant, pBadge.kind);
  }

  /** Raw id of the selected model, plus every draft sidecar it pulls. */
  function fTriggerTooltipLabel(pOption: ModelOption): string {
    const aDrafts = (pOption.draftSidecars ?? []).map((pBadge) =>
      fDraftSidecarLabel(pOption.model, pBadge)
    );

    return [pOption.model, ...aDrafts].join(' + ');
  }

  function fHandleManageModels() {
    open = false;

    // let the menu finish closing before the dialog takes focus
    setTimeout(() => (onManageModels ? onManageModels() : cUiStore.mOpenModelsManager()), 0);
  }
</script>

{#snippet selectorTriggerInner()}
  <ModelsSelectorTriggerIcon class="mr-0.375 size-3.5 shrink-0" option={cSelectedOption} />

  <span class="flex min-w-0 items-center gap-0.5">
    {#if cSelectedOption}
      <ModelId
        class="min-w-0 overflow-hidden"
        hideOrgName={!showOrgName}
        hideQuantization
        modelId={cSelectedOption.model}
      />
    {:else}
      <span class="min-w-0 font-medium">{fTranslate('text_1d446005add3')}</span>
    {/if}
  </span>

  {#if ms.updating || ms.isLoadingModel || vTriggerLoading}
    <Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
  {:else}
    <ChevronDown class="h-3 w-3.5 shrink-0" />
  {/if}

  {#if vTriggerLoading}
    <ModelLoadHighlight percent={vTriggerLoadPercent} />
  {/if}
{/snippet}

{#snippet modelOption(pItem: ModelItem, pHideOrgName: boolean)}
  {@const { option: vOption } = pItem}
  {@const cIsSelected = currentModel === vOption.model || ms.activeId === vOption.id}
  {@const cIsHighlighted = vOption.id === highlightedId}
  {@const cIsFav = ms.mIsFavorite(vOption.model)}

  <ModelsSelectorOption
    hideOrgName={pHideOrgName}
    isFav={cIsFav}
    isHighlighted={cIsHighlighted}
    isSelected={cIsSelected}
    onKeyDown={(pEvent) => {
      if (pEvent.key === KeyboardKey.cEnter || pEvent.key === KeyboardKey.cSpace) {
        pEvent.preventDefault();
        onModelKeyAction?.(vOption.id, pEvent.altKey);
      }
    }}
    onMouseEnter={() => onHighlight?.(vOption.id)}
    onSelect={ms.mHandleSelect}
    option={vOption}
  />
{/snippet}

<DropdownMenu.fRoot bind:open onOpenChange={ms.mHandleOpenChange}>
  <Tooltip.fRoot>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props: dProps })}
        <DropdownMenu.Trigger
          {...dProps}
          class={[
            `relative inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 max-md:h-8 max-md:px-2.25 max-md:py-1.25 max-md:text-[13px] focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
            !ms.isCurrentModelInCache
              ? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
              : 'text-foreground',
            open && 'text-foreground',
            'max-w-[min(calc(100vw-4rem) md:max-w-[min(calc(100cqw-9rem),25rem)]'
          ]}
          disabled={disabled || ms.updating}
        >
          {@render selectorTriggerInner()}
        </DropdownMenu.Trigger>
      {/snippet}
    </Tooltip.Trigger>

    {#if cSelectedOption}
      <Tooltip.Content>
        <p class="font-mono">{fTriggerTooltipLabel(cSelectedOption)}</p>
      </Tooltip.Content>
    {/if}
  </Tooltip.fRoot>

  <DropdownMenu.Content
    align="end"
    class="w-full md:min-w-80 md:w-md max-w-[calc(100vw-2rem)] p-0! max-h-[min(40rem,calc(var(--bits-dropdown-menu-content-available-height)-1rem))]"
    onOpenAutoFocus={(pEvent) => pEvent.preventDefault()}
  >
    <DropdownMenuSearchable
      emptyMessage={ms.emptyMessage}
      isEmpty={ms.isEmpty && ms.isCurrentModelInCache}
      onSearchChange={(pV) => ms.mSetSearchTerm(pV)}
      {onSearchKeyDown}
      placeholder={fTranslate('text_37b90680b842')}
      searchClass="bg-transparent"
      searchValue={ms.searchTerm}
    >
      <!-- Option list; the search header sticks to the top and the actions
		     footer to the bottom of the content scrollport. -->
      {@render pickerContent('px-1.5')}

      {#snippet footer()}
        <DropdownMenu.Group class="px-2">
          <DropdownMenu.Item class="gap-2" onSelect={fHandleManageModels}>
            <ModelIcon class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

            {fTranslate('dgs_4318541457e1')}
          </DropdownMenu.Item>
        </DropdownMenu.Group>
      {/snippet}
    </DropdownMenuSearchable>
  </DropdownMenu.Content>
</DropdownMenu.fRoot>

{#snippet pickerContent(pListClass: string)}
  <div class={['models-list', pListClass]}>
    {#if !ms.isCurrentModelInCache && currentModel}
      <!-- Show unavailable model as first option (disabled) -->
      <button
        aria-disabled="true"
        aria-selected="true"
        class="flex w-full cursor-not-allowed items-center bg-red-400/10 p-2 text-left text-sm text-red-400"
        disabled
        role="option"
        type="button"
      >
        <ModelId
          class="flex-1"
          hideOrgName={!showOrgName}
          hideQuantization
          modelId={currentModel}
        />

        <span class="ml-2 text-xs whitespace-nowrap opacity-70"
          >{fTranslate('text_1c96e2456748')}</span
        >
      </button>
    {/if}

    {#if ms.isEmpty}
      <p class="px-4 py-3 text-sm text-muted-foreground">{ms.emptyMessage}</p>
    {/if}

    <ModelsSelectorList
      activeId={ms.activeId}
      {currentModel}
      favorites={ms.favoriteItems}
      groups={ms.groupedFilteredOptions}
      loaded={ms.loadedItems}
      onSelect={ms.mHandleSelect}
      renderOption={modelOption}
      sectionHeaderClass="[&:not(:first-child)]:mt-1 mb-1 px-2 py-2 text-[13px] font-semibold text-foreground/80 select-none"
      {showOrgName}
    />
  </div>
{/snippet}
