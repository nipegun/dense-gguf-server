<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelLoadHighlight from '../ModelLoadHighlight.svelte';
  import { ChevronDown, Loader2 } from '@lucide/svelte';
  import {
    ModelId,
    ModelsSelectorDrawer,
    ModelsSelectorDropdown,
    ModelsSelectorList,
    ModelsSelectorOption,
    ModelsSelectorTriggerIcon
  } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cDropdownMenuContentSearchSelector, ModelIcon, cSettingsKeys } from '$lib/constants';
  import { KeyboardKey, ServerModelStatus } from '$lib/enums';
  import { fUseChatFormModel } from '$lib/hooks/use-chat-form-model.svelte';
  import { fUseModelsSelector } from '$lib/hooks/use-models-selector.svelte';
  import { cDeviceStore, cModelsStore, cServerStore, cSettingsStore } from '$lib/stores';
  import { type ModelItem, fModelLoadFraction } from '$lib/utils';

  interface Props {
    /** Model to show, when the caller owns it, e.g. the model of one message. */
    currentModel?: string | null;
    disabled?: boolean;
    forceForegroundText?: boolean;
    onModelChange?: (pModelId: string, pModelName: string) => Promise<boolean> | boolean | void;
    useGlobalSelection?: boolean;
  }

  let {
    currentModel = null,
    disabled = false,
    forceForegroundText = false,
    onModelChange,
    useGlobalSelection = false
  }: Props = $props();

  // one setting for every model id in the selector: the trigger and the rows
  let vShowOrgName = $derived(cSettingsStore.vConfig[cSettingsKeys.SHOW_MODEL_ORG_NAME] ?? true);

  // a phone opens the picker in a drawer, a desktop keeps the anchored dropdown
  let vIsMobile = $derived(cDeviceStore.isMobile);
  let vIsOffline = $derived(!!cServerStore.error);

  let vIsOpen = $state(false);
  let vHighlightedId = $state<string | null>(null);

  const cFormModel = fUseChatFormModel();

  // the store's pick wins while it differs from the conversation's model: the
  // trigger follows the model the user just picked, whose load state and
  // progress it reports, not the one the last answer came from
  let vSelectorModel = $derived.by(() => {
    const cStoreModel = cModelsStore.vSelectedModelName;

    if (cStoreModel && cStoreModel !== cFormModel.conversationModel) {
      return cStoreModel;
    }

    return cFormModel.conversationModel;
  });

  // the caller's model comes first, then the pick, then the conversation's
  let vDisplayModel = $derived(currentModel ?? vSelectorModel);

  const cMs = fUseModelsSelector({
    currentModel: () => vDisplayModel,
    onModelChange: () => onModelChange,
    onOpenChange: (open) => {
      vIsOpen = open;
      vHighlightedId = null;
    },
    useGlobalSelection: () => useGlobalSelection
  });

  const cSelectedOption = $derived(cMs.mGetDisplayOption());
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

  $effect(() => {
    void cMs.searchTerm;
    vHighlightedId = null;
  });

  // Keyboard navigation follows the on-screen row order, not the flat option list order.
  let aVisualOrder = $derived.by(() => {
    const aOrder: string[] = [];

    for (const cItem of cMs.favoriteItems) aOrder.push(cItem.option.id);
    for (const cGroup of cMs.groupedFilteredOptions.available) {
      for (const cItem of cGroup.items) aOrder.push(cItem.option.id);
    }

    return aOrder;
  });

  let vHighlightedIndex = $derived(vHighlightedId ? aVisualOrder.indexOf(vHighlightedId) : -1);

  function fMoveHighlight(pDirection: 1 | -1) {
    const cLen = aVisualOrder.length;

    if (cLen === 0) {
      vHighlightedId = null;

      return;
    }

    let vIndex = vHighlightedIndex;

    if (vIndex === -1) {
      vIndex = pDirection === 1 ? 0 : cLen - 1;
    } else {
      vIndex = (vIndex + pDirection + cLen) % cLen;
    }

    vHighlightedId = aVisualOrder[vIndex];
  }

  // Alt+Enter only unloads and keeps the picker open.
  async function fHandleModelKeyAction(pModelId: string, pUnload: boolean) {
    if (!pUnload) {
      void cMs.mHandleSelect(pModelId);

      return;
    }

    const cStatus = cModelsStore.mGetModelStatus(pModelId);

    if (cStatus === ServerModelStatus.LOADING) return;

    await cModelsStore.status.mUnload(pModelId);
  }

  function fHandleSearchKeyDown(pEvent: KeyboardEvent) {
    if (pEvent.isComposing) return;

    if (pEvent.key === KeyboardKey.cArrowDown) {
      pEvent.preventDefault();
      fMoveHighlight(1);
    } else if (pEvent.key === KeyboardKey.cArrowUp) {
      pEvent.preventDefault();
      fMoveHighlight(-1);
    } else if (pEvent.key === KeyboardKey.cEnter) {
      pEvent.preventDefault();

      if (vHighlightedId) {
        void fHandleModelKeyAction(vHighlightedId, pEvent.altKey);
      } else if (aVisualOrder.length > 0) {
        vHighlightedId = aVisualOrder[0];
      }
    }
  }

  // bits-ui auto-focuses the opened content, which can yank the page scroll: the
  // content prevents that and this focuses the search input instead
  $effect(() => {
    if (!vIsOpen || vIsMobile) return;

    let vFrames = 0;
    let vHandle = requestAnimationFrame(function focusSearch() {
      const input = document.querySelector<HTMLElement>(cDropdownMenuContentSearchSelector);

      if (input) {
        input.focus({ preventScroll: true });

        return;
      }

      if (vFrames++ < 20) vHandle = requestAnimationFrame(focusSearch);
    });

    return () => cancelAnimationFrame(vHandle);
  });

  export function open() {
    cMs.mHandleOpenChange(true);
  }
</script>

{#snippet selectorTriggerInner()}
  <ModelsSelectorTriggerIcon
    class="mr-0.375 size-3.5 max-md:size-4 shrink-0"
    option={cSelectedOption}
  />

  <span class="flex min-w-0 items-center gap-0.5">
    {#if cSelectedOption}
      <ModelId
        class="min-w-0 overflow-hidden"
        hideOrgName={!vShowOrgName}
        hideQuantization
        modelId={cSelectedOption.model}
      />
    {:else}
      <span class="min-w-0 font-medium">{fTranslate('text_1d446005add3')}</span>
    {/if}
  </span>

  {#if cMs.updating || cMs.isLoadingModel || vTriggerLoading}
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
  {@const cIsSelected = cMs.activeId === vOption.id}
  {@const cIsHighlighted = vOption.id === vHighlightedId}
  {@const cIsFav = cMs.mIsFavorite(vOption.model)}

  <ModelsSelectorOption
    hideOrgName={pHideOrgName}
    isFav={cIsFav}
    isHighlighted={cIsHighlighted}
    isSelected={cIsSelected}
    onKeyDown={(pEvent) => {
      if (pEvent.key === KeyboardKey.cEnter || pEvent.key === KeyboardKey.cSpace) {
        pEvent.preventDefault();
        void fHandleModelKeyAction(vOption.id, pEvent.altKey);
      }
    }}
    onMouseEnter={() => (vHighlightedId = vOption.id)}
    onSelect={cMs.mHandleSelect}
    option={vOption}
  />
{/snippet}

{#snippet pickerList(pListClass: string)}
  <div class={['models-list', pListClass]}>
    {#if !cMs.isCurrentModelInCache && vDisplayModel}
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
          hideOrgName={!vShowOrgName}
          hideQuantization
          modelId={vDisplayModel}
        />

        <span class="ml-2 text-xs whitespace-nowrap opacity-70"
          >{fTranslate('text_1c96e2456748')}</span
        >
      </button>
    {/if}

    {#if cMs.isEmpty}
      <p class="px-4 py-3 text-sm text-muted-foreground">{cMs.emptyMessage}</p>
    {/if}

    <ModelsSelectorList
      activeId={cMs.activeId}
      currentModel={vDisplayModel}
      favorites={cMs.favoriteItems}
      groups={cMs.groupedFilteredOptions}
      loaded={cMs.loadedItems}
      onSelect={cMs.mHandleSelect}
      renderOption={modelOption}
      sectionHeaderClass="[&:not(:first-child)]:mt-2 mb-1 px-2 py-2.5 text-sm font-semibold text-foreground/80 select-none"
      showOrgName={vShowOrgName}
    />
  </div>
{/snippet}

{#if cMs.loading && cMs.options.length === 0 && cMs.isMultiModel}
  <div class="flex items-center gap-2 text-xs text-muted-foreground">
    <Loader2 class="h-3.5 w-3.5 animate-spin" />

    {fTranslate('dgs_802435241f0a')}
  </div>
{:else if cMs.options.length === 0 && cMs.isMultiModel}
  {#if vDisplayModel}
    <span
      class="inline-flex items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs text-muted-foreground"
      style="max-width: min(calc(100cqw - 10rem), 48rem)"
    >
      <ModelIcon class="h-3.5 w-3.5 shrink-0" />
    </span>
  {:else}
    <span class="text-xs text-muted-foreground">{fTranslate('dgs_36604ff47cf7')}</span>
  {/if}
{:else if cMs.isMultiModel}
  {#if vIsMobile}
    <button
      class="relative inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 max-md:gap-1.5 max-md:h-7 max-md:px-2.25 max-md:py-1 max-md:text-[13px] focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground"
      disabled={disabled || cMs.updating}
      onclick={() => cMs.mHandleOpenChange(true)}
      style="max-width: min(calc(100vw-4rem), 32rem)"
      type="button"
    >
      {@render selectorTriggerInner()}
    </button>

    <ModelsSelectorDrawer
      bind:open={vIsOpen}
      emptyMessage={cMs.emptyMessage}
      isEmpty={cMs.isEmpty && cMs.isCurrentModelInCache}
      onOpenChange={cMs.mHandleOpenChange}
      onSearchChange={(pV) => cMs.mSetSearchTerm(pV)}
      onSearchKeyDown={fHandleSearchKeyDown}
      searchTerm={cMs.searchTerm}
    >
      {@render pickerList('px-2.5')}
    </ModelsSelectorDrawer>
  {:else}
    <ModelsSelectorDropdown
      bind:open={vIsOpen}
      currentModel={vDisplayModel}
      disabled={disabled || vIsOffline}
      highlightedId={vHighlightedId}
      ms={cMs}
      onHighlight={(pId) => (vHighlightedId = pId)}
      onModelKeyAction={(pId, pUnload) => void fHandleModelKeyAction(pId, pUnload)}
      onSearchKeyDown={fHandleSearchKeyDown}
      showOrgName={vShowOrgName}
    />
  {/if}
{:else if vIsMobile}
  <!-- a phone has no manager: the single model still gets the drawer, so its
	     load control, actions and information stay reachable -->
  <button
    class="relative inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 max-md:gap-1.5 max-md:h-7 max-md:px-2.25 max-md:py-1 max-md:text-[13px] focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground"
    disabled={disabled || cMs.updating}
    onclick={() => cMs.mHandleOpenChange(true)}
    style="max-width: min(calc(100vw-4rem), 32rem)"
    type="button"
  >
    {@render selectorTriggerInner()}
  </button>

  <ModelsSelectorDrawer
    bind:open={vIsOpen}
    emptyMessage={cMs.emptyMessage}
    isEmpty={cMs.isEmpty && cMs.isCurrentModelInCache}
    onOpenChange={cMs.mHandleOpenChange}
    onSearchChange={(pV) => cMs.mSetSearchTerm(pV)}
    onSearchKeyDown={fHandleSearchKeyDown}
    searchTerm={cMs.searchTerm}
  >
    {@render pickerList('px-2.5')}
  </ModelsSelectorDrawer>
{:else}
  <!-- a lone dgs.cpp server has nothing to choose from: the trigger opens the
	     manager instead, so the button is its own affordance -->
  <Tooltip.fRoot>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props: dProps })}
        <button
          {...dProps}
          class={[
            `inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-background px-1.5 py-1 text-xs shadow-sm transition hover:bg-muted-foreground/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60 dark:bg-muted-foreground/15 dark:text-secondary-foreground`,
            !cMs.isCurrentModelInCache
              ? 'bg-red-400/10 text-red-400! hover:bg-red-400/20 hover:text-red-400'
              : forceForegroundText
                ? 'text-foreground'
                : 'text-foreground',
            vIsOpen && 'text-foreground'
          ]}
          disabled={disabled || cMs.updating}
          onclick={() => cMs.mHandleOpenChange(true)}
          style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
        >
          <ModelsSelectorTriggerIcon class="mr-0.375 size-3.5 shrink-0" option={cSelectedOption} />

          {#if cSelectedOption}
            <ModelId
              class="min-w-0 overflow-hidden"
              hideOrgName={!vShowOrgName}
              hideQuantization
              modelId={cSelectedOption.model}
            />
          {/if}

          {#if cMs.updating}
            <Loader2 class="h-3 w-3.5 shrink-0 animate-spin" />
          {/if}
        </button>
      {/snippet}
    </Tooltip.Trigger>
  </Tooltip.fRoot>
{/if}
