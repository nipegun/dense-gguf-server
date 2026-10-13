<script lang="ts">
  import ModelsManagerModelConfiguration from './ModelsManagerModelConfiguration/ModelsManagerModelConfiguration.svelte';
  import ModelsManagerModelsTable from './ModelsManagerModelsTable.svelte';
  import {
    fDownloadGroups,
    fGroupModelQuants,
    fModelCapability,
    fModelContextLength,
    type ModelQuantGroup,
    type ModelsTableGroup,
    fSplitHiddenQuants
  } from './utils';
  import { cLocalBackendId, type ModalityKey, dModelsTableGroupLabels } from '$lib/constants';
  import { ModelCapability, ModelsTableGroupKind } from '$lib/enums';
  import { cConversationsStore, cModelsStore, cServerStore, cUiStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';
  import { fFilterModelOptions } from '$lib/utils';
  import { type Snippet, untrack } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';

  interface Props {
    class?: string;
    /** Forwarded to the table's toolbar right end. */
    toolbarEnd?: Snippet;
  }

  let { class: className, toolbarEnd }: Props = $props();

  let vFilter = $state('');
  let vContextLimit = $state(0);
  let aModalityFilter = $state<ModalityKey[]>([]);
  let aCapabilityFilter = $state<ModelCapability[]>([]);
  let vSelectedId = $state<string | null>(null);

  let aAllModels = $derived(cModelsStore.aModels);

  let vSelected = $derived(aAllModels.find((pOption) => pOption.id === vSelectedId) ?? null);
  // The pane is laid out before it is ever opened, so the first open only slides a
  // finished panel in. It renders the selection, else the model it last showed.
  let vLastPicked = $state<ModelOption | null>(null);
  let vTarget = $derived(vSelected ?? vLastPicked ?? aAllModels[0] ?? null);
  let vShownId = $state<string | null>(null);
  let vIsSwapping = $state(false);
  let vFade = $state<'open' | 'swap'>('open');
  let vShownOption = $derived(aAllModels.find((pOption) => pOption.id === vShownId) ?? null);

  $effect(() => {
    const cId = vSelectedId;

    if (!cId) return;

    // untracked: the effect must not track the state it writes
    untrack(() => {
      vLastPicked = aAllModels.find((pOption) => pOption.id === cId) ?? null;
    });
  });

  /** How long the panel takes to fade out before it swaps to another model. */
  const cSwapFadeMs = 120;

  // Another model fades the panel out, swaps it, then fades it back in.
  $effect(() => {
    const cNext = vTarget?.id ?? null;

    if (!cNext) return;

    if (vShownId === null) {
      untrack(() => (vShownId = cNext));

      return;
    }

    if (cNext === vShownId) {
      if (vSelected) {
        untrack(() => {
          vIsSwapping = false;
          vFade = 'open';
        });
      }

      return;
    }

    untrack(() => {
      vIsSwapping = true;
      vFade = 'swap';
    });

    const cTimer = setTimeout(() => {
      untrack(() => {
        vShownId = cNext;
        vIsSwapping = false;
      });
    }, cSwapFadeMs);

    return () => clearTimeout(cTimer);
  });

  let fIsFavorite = $derived((pOption: ModelOption) =>
    cModelsStore.sFavoriteModelIds.has(pOption.model)
  );
  let aMatching = $derived.by(() => {
    // the term matches what the selector search matches: name, model, aliases and tags
    const aSearched = fFilterModelOptions(aAllModels, vFilter);

    return aSearched.filter((pOption) => {
      // every capability asked for has to be there, but a model whose chat template
      // has not been read yet stays listed: the row fetches the record as it comes
      // near the viewport, so hiding it would keep it from ever answering
      if (
        aCapabilityFilter.length > 0 &&
        !aCapabilityFilter.every((pCapability) => fModelCapability(pOption, pCapability) !== false)
      ) {
        return false;
      }

      // a model whose modalities are unknown cannot be shown to match
      if (
        aModalityFilter.length > 0 &&
        !aModalityFilter.some((pKey) => pOption.modalities?.[pKey])
      ) {
        return false;
      }

      if (vContextLimit === 0) return true;

      // the context of a row arrives with its Hub record, so a model we cannot read
      // yet stays listed; only a known context below the limit rules it out
      const context = fModelContextLength(pOption);

      return context === null || context >= vContextLimit;
    });
  });

  // recently used models lead their section, the rest keep the server's order
  let vRank = $derived.by(() => {
    const cMap = new SvelteMap<string, number>();

    cModelsStore.aRecentModelIds.forEach((pId, pIndex) => cMap.set(pId, pIndex));

    return cMap;
  });

  const fRankOf = (pEntry: ModelQuantGroup) =>
    Math.min(...pEntry.quants.map((pQuant) => vRank.get(pQuant.id) ?? Number.MAX_SAFE_INTEGER));
  const fByRecency = (pList: ModelQuantGroup[]) =>
    vRank.size === 0 ? pList : [...pList].sort((pA, pB) => fRankOf(pA) - fRankOf(pB));
  // one entry per repo, so a model with several quants takes a single table row
  let entries = $derived(fByRecency(fGroupModelQuants(aMatching)));
  // tracked downloads get their own section, listed like any other model
  let aDownloads = $derived(
    fDownloadGroups(cModelsStore.status.mGetDownloadEntries(), cModelsStore.aModels)
  );
  let aGroups = $derived.by(() => {
    // a loaded quant is a model of its own: its repo keeps the quants left behind
    const fIsLoaded = (pOption: ModelOption) => cModelsStore.mIsModelLoaded(pOption.model);
    // a tracked download stands in its own section, so it is not listed twice
    const fIsDownload = (pOption: ModelOption) =>
      cModelsStore.status.mIsDownloadInProgress(pOption.model) ||
      cModelsStore.status.mIsDownloadPaused(pOption.model);
    const aLoaded: ModelQuantGroup[] = [];
    const aRest: ModelQuantGroup[] = [];

    for (const cEntry of entries) {
      const aRemaining = cEntry.quants.filter(
        (pQuant) => !fIsLoaded(pQuant) && !fIsDownload(pQuant)
      );

      for (const cQuant of cEntry.quants) {
        if (!fIsLoaded(cQuant)) continue;

        aLoaded.push({ ...cEntry, base: cQuant, key: cQuant.id, quants: [cQuant] });
      }

      if (aRemaining.length > 0) aRest.push({ ...cEntry, base: aRemaining[0], quants: aRemaining });
    }

    // a favorite quant stands on its own, listed flat like a loaded one, and the
    // quants left behind stay with their repo in the local block
    const aFavorites: ModelQuantGroup[] = [];
    const aLocalRest: ModelQuantGroup[] = [];

    for (const cEntry of aRest) {
      for (const cQuant of cEntry.quants) {
        if (cModelsStore.sFavoriteModelIds.has(cQuant.model)) {
          aFavorites.push({ ...cEntry, base: cQuant, key: cQuant.id, quants: [cQuant] });
        }
      }

      const aRemaining = cEntry.quants.filter(
        (pQuant) => !cModelsStore.sFavoriteModelIds.has(pQuant.model)
      );

      if (aRemaining.length > 0) {
        aLocalRest.push({ ...cEntry, base: aRemaining[0], quants: aRemaining });
      }
    }

    const { hidden: aHidden, local: aLocal } = fSplitHiddenQuants(aLocalRest, (pOption) =>
      cModelsStore.mIsHidden(pOption.id)
    );
    const aOrdered: ModelsTableGroup[] = [];
    // loaded models lead the table, then favorites, then the local block
    const fPushSection = (pKind: ModelsTableGroupKind, pItems: ModelQuantGroup[]): void => {
      if (pItems.length === 0) return;

      aOrdered.push({
        items: pItems,
        key: pKind === ModelsTableGroupKind.cLocal ? cLocalBackendId : pKind,
        kind: pKind,
        label: dModelsTableGroupLabels[pKind]
      });
    };

    fPushSection(ModelsTableGroupKind.cLoaded, aLoaded);
    fPushSection(ModelsTableGroupKind.cDownloading, aDownloads);
    fPushSection(ModelsTableGroupKind.cFavorites, aFavorites);
    fPushSection(ModelsTableGroupKind.cLocal, aLocal);
    fPushSection(ModelsTableGroupKind.cHidden, aHidden);

    return aOrdered;
  });

  // a caller can ask for one model to be revealed, the download rows do
  $effect(() => {
    const cFocus = cUiStore.vManageModelFocus;

    if (!cFocus) return;

    const cOption = aAllModels.find((pModel) => pModel.id === cFocus || pModel.model === cFocus);

    if (cOption) vSelectedId = cOption.id;

    cUiStore.vManageModelFocus = null;
  });
  async function fToggleLoad(pOption: ModelOption): Promise<void> {
    if (cModelsStore.mIsModelLoaded(pOption.model)) {
      await cModelsStore.status.mUnload(pOption.model);

      return;
    }

    await cModelsStore.status.load(pOption.model);
  }

  /** The chat sits behind the dialog, so the dialog closes and the composer takes focus. */
  function fReturnToChat(): void {
    cUiStore.vManageModelsOpen = false;
    cUiStore.mRequestComposerFocus();
  }

  /** Load the model when the server can take load requests, without waiting for it. */
  function fLoadInBackground(pOption: ModelOption): void {
    // only the built-in server loads on request, and only in router mode
    if (cServerStore.isRouterMode && !cModelsStore.mIsModelLoaded(pOption.model)) {
      cModelsStore.status
        .load(pOption.model)
        .catch((pError) => console.error('Failed to load model:', pError));
    }
  }

  /** Switch the open chat to this model, the way the desktop model dropdown does. */
  async function fUseInChat(pOption: ModelOption): Promise<void> {
    // a deliberate pick records the recency; useInNewChat skips it because a new
    // conversation is where the pick is exercised, not a return to a model the
    // user went looking for
    await cModelsStore.mSelectModelById(pOption.id, { recordRecent: true });

    fLoadInBackground(pOption);

    fReturnToChat();
  }

  async function fUseInNewChat(pOption: ModelOption): Promise<void> {
    await cModelsStore.mSelectModelById(pOption.id);
    await cConversationsStore.mOpenNewChat();

    fLoadInBackground(pOption);

    fReturnToChat();
  }
</script>

<div class={['relative flex min-h-0 flex-1', className]}>
  <div class="min-h-0 min-w-0 flex-1">
    <ModelsManagerModelsTable
      bind:capabilities={aCapabilityFilter}
      bind:contextLimit={vContextLimit}
      bind:filter={vFilter}
      bind:modalities={aModalityFilter}
      groups={aGroups}
      isFavorite={fIsFavorite}
      onSelect={(pOption) => (vSelectedId = pOption.id)}
      selectedId={vSelectedId}
      {toolbarEnd}
    />
  </div>

  <!-- on a phone the pane covers the whole dialog: the manager header would only
	     repeat what the pane's own header says -->
  <div
    class="pane-drawer max-md:fixed max-md:inset-0 max-md:z-[60] shrink-0"
    data-open={vSelected !== null}
  >
    <!-- the content box keeps the open width, so it never reflows with the drawer -->
    <div
      class="pane-content flex h-full min-h-0 w-[30rem] max-w-[30rem] flex-col border-l border-border/40 max-md:w-full max-md:max-w-none max-md:border-l-0 max-md:bg-background"
      data-fade={vFade}
      data-visible={vSelected !== null && !vIsSwapping}
    >
      {#if vShownOption}
        {#key vShownId}
          <ModelsManagerModelConfiguration
            onClose={() => (vSelectedId = null)}
            onToggleLoad={() => void fToggleLoad(vShownOption)}
            onUseInChat={() => void fUseInChat(vShownOption)}
            onUseInNewChat={() => void fUseInNewChat(vShownOption)}
            option={vShownOption}
          />
        {/key}
      {/if}
    </div>
  </div>
</div>

<style>
  /*
	 * The drawer moves by width because the table behind it gets that space back, so
	 * the content box inside holds the open width and only the container changes.
	 * Opening takes the iOS-like drawer curve; closing is the system responding, so
	 * it snaps back on the stronger ease-out.
	 */
  .pane-drawer {
    width: 0;
    overflow: clip;
    visibility: hidden;
    transition:
      width 120ms cubic-bezier(0.23, 1, 0.32, 1),
      visibility 120ms;
  }

  .pane-drawer[data-open='true'] {
    width: 30rem;
    visibility: visible;
    transition:
      width 200ms cubic-bezier(0.32, 0.72, 0, 1),
      visibility 200ms;
  }

  .pane-content {
    opacity: 0;
    transition: opacity 120ms cubic-bezier(0.23, 1, 0.32, 1);
  }

  .pane-content[data-visible='true'] {
    opacity: 1;
  }

  /* opening: the fade waits for the drawer to move */
  .pane-content[data-visible='true'][data-fade='open'] {
    transition: opacity 150ms cubic-bezier(0.23, 1, 0.32, 1) 80ms;
  }

  /* swapping models: out, then in, with no pause */
  .pane-content[data-visible='true'][data-fade='swap'] {
    transition: opacity 120ms cubic-bezier(0.23, 1, 0.32, 1);
  }

  /* a phone has no room beside the table: the pane covers it instead */
  @media (max-width: 767px) {
    .pane-drawer[data-open='true'] {
      width: auto;
    }
  }
</style>
