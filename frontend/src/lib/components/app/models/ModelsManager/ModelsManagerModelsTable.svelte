<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelsManagerModelRow from './ModelsManagerModelRow.svelte';
  import ModelsManagerQuantRow from './ModelsManagerQuantRow.svelte';
  import ModelsManagerRepoRow from './ModelsManagerRepoRow.svelte';
  import ModelsManagerTableToolbar from './ModelsManagerTableToolbar.svelte';
  import {
    fHasActiveFilters,
    fModelContextLength,
    type ModelQuantGroup,
    type ModelsTableGroup
  } from './utils';
  import {
    ArrowDown,
    ArrowUp,
    ChevronDown,
    ChevronUp,
    Download,
    EyeOff,
    Heart,
    Power
  } from '@lucide/svelte';
  import {
    CollapsibleRegion,
    GroupedList,
    type GroupedListGroup,
    cGroupedListUnit,
    Logo,
    ModelAvatar,
    ModelsSection
  } from '$lib/components/app';
  import { DialogConfirmDownload } from '$lib/components/app/dialogs';
  import {
    cFamilyRowWindow,
    type ModalityKey,
    cModelRowGridClass,
    cModelRowTrailingCellClass,
    cModelRowWindow,
    cModelsTableGroupStickyOffset,
    cSettingsKeys
  } from '$lib/constants';
  import {
    KeyboardKey,
    ModelCapability,
    ModelDownloadConfirmAction,
    ModelsTableGroupKind,
    ModelsTableSortKey
  } from '$lib/enums';
  import { cModelsStore, cSettingsStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';
  import { fGroupModelFamilies, type ModelFamilyGroup } from '$lib/utils/model-families';
  import type { Snippet } from 'svelte';
  import { SvelteSet } from 'svelte/reactivity';

  interface Props {
    /** Capabilities a model must have every one of. */
    capabilities?: ModelCapability[];
    /** Smallest context a model must support; 0 keeps every model. */
    contextLimit?: number;
    filter?: string;
    groups: ModelsTableGroup[];
    isFavorite: (pOption: ModelOption) => boolean;
    onSelect: (pOption: ModelOption) => void;
    /** Modalities a model must support at least one of. */
    modalities?: ModalityKey[];
    selectedId: string | null;
    /** Rendered at the toolbar's right end, past the filters. */
    toolbarEnd?: Snippet;
  }

  let {
    capabilities = $bindable<ModelCapability[]>([]),
    contextLimit = $bindable(0),
    filter = $bindable(''),
    groups,
    isFavorite,
    modalities = $bindable<ModalityKey[]>([]),
    onSelect,
    selectedId,
    toolbarEnd
  }: Props = $props();

  let vIsEmpty = $derived(groups.every((pGroup) => pGroup.items.length === 0));
  let vHasFilters = $derived(fHasActiveFilters(contextLimit, modalities, capabilities));

  /** Noun the show-more row counts in, per unit of the grouped list. */
  const dShowMoreNouns: Record<cGroupedListUnit, string> = {
    [cGroupedListUnit.ENTRIES]: 'models',
    [cGroupedListUnit.GROUPS]: 'families'
  };
  let vPendingDelete = $state('');
  let vDeleteOpen = $state(false);
  /** Repos whose quants are folded away; the rest show them. */
  const cCollapsedQuants = new SvelteSet<string>();
  /** Sections that list their models straight, without folding them into families. */
  const sFlatSections = new Set<ModelsTableGroupKind>([
    ModelsTableGroupKind.cDownloading,
    ModelsTableGroupKind.cFavorites,
    ModelsTableGroupKind.cLoaded
  ]);
  let aSections = $derived(
    groups.map((pGroup) => {
      // a flat section lists its models straight, families or not
      const cFlat =
        sFlatSections.has(pGroup.kind) ||
        !cSettingsStore.vConfig[cSettingsKeys.GROUP_MODELS_BY_FAMILY];
      // downloads keep the feed's order: their progress, not their name, moves
      const aItems =
        pGroup.kind === ModelsTableGroupKind.cDownloading
          ? pGroup.items
          : fSortEntries(pGroup.items);

      return {
        ...pGroup,
        families: cFlat ? [] : fGroupModelFamilies(aItems, (pEntry) => pEntry.base.model),
        flat: cFlat,
        items: aItems
      };
    })
  );

  function fFamilyGroups(
    pGroup: (typeof aSections)[number]
  ): GroupedListGroup<ModelFamilyGroup<ModelQuantGroup>, ModelQuantGroup>[] {
    return pGroup.families.map((pFamily) => ({
      entries: pFamily.entries,
      group: pFamily,
      key: `${pGroup.key}::${pFamily.key}`
    }));
  }

  // cancel is confirmed once for the whole list, so one dialog serves every row
  function fRequestDelete(pOption: ModelOption): void {
    vPendingDelete = pOption.model;
    vDeleteOpen = true;
  }

  function fToggleQuants(pKey: string): void {
    if (cCollapsedQuants.has(pKey)) {
      cCollapsedQuants.delete(pKey);
    } else {
      cCollapsedQuants.add(pKey);
    }
  }

  /** Column the table is ordered by; unset keeps the manager's own order. */
  let vSortKey = $state<ModelsTableSortKey | null>(null);
  let vSortAsc = $state(true);

  function fCompareEntries(pLeft: ModelQuantGroup, pRight: ModelQuantGroup): number {
    const cA = pLeft.base;
    const cB = pRight.base;

    switch (vSortKey) {
      case ModelsTableSortKey.CONTEXT:
        // a context we cannot read yet sorts with the smallest ones
        return (fModelContextLength(cA) ?? 0) - (fModelContextLength(cB) ?? 0);
      case ModelsTableSortKey.NAME:
        return cA.model.localeCompare(cB.model);
      default:
        return 0;
    }
  }

  function fSortEntries(entries: ModelQuantGroup[]): ModelQuantGroup[] {
    if (!vSortKey) return entries;

    const cDirection = vSortAsc ? 1 : -1;

    return [...entries].sort((pA, pB) => cDirection * fCompareEntries(pA, pB));
  }

  /** A click on a new column sorts lowest first, then highest first, then clears. */
  function fToggleSort(pKey: ModelsTableSortKey): void {
    if (vSortKey !== pKey) {
      vSortKey = pKey;
      vSortAsc = true;

      return;
    }

    if (vSortAsc) {
      vSortAsc = false;

      return;
    }

    vSortKey = null;
  }

  function fSortTitle(pKey: ModelsTableSortKey, pLabel: string): string {
    const name = pLabel.toLowerCase();

    if (vSortKey !== pKey) return `Sort by ${name}, lowest first`;

    return vSortAsc ? `Sort by ${name}, highest first` : `Stop sorting by ${name}`;
  }

  function fHandleFamilyKeydown(pEvent: KeyboardEvent, pToggle: () => void): void {
    if (pEvent.key === KeyboardKey.cSpace) pEvent.preventDefault();

    if (pEvent.key === KeyboardKey.cEnter || pEvent.key === KeyboardKey.cSpace) pToggle();
  }
</script>

{#snippet sortHeader(pKey: ModelsTableSortKey, pLabel: string)}
  <button
    class="inline-flex cursor-pointer items-center gap-1 uppercase transition hover:text-foreground focus:outline-none"
    onclick={() => fToggleSort(pKey)}
    title={fSortTitle(pKey, pLabel)}
    type="button"
  >
    {pLabel}

    {#if vSortKey === pKey}
      {#if vSortAsc}
        <ArrowUp class="h-3 w-3" />
      {:else}
        <ArrowDown class="h-3 w-3" />
      {/if}
    {/if}
  </button>
{/snippet}

{#snippet entryTree(pEntry: ModelQuantGroup, pIndent = 0)}
  {#if pEntry.quants.length > 1}
    <ModelsManagerRepoRow
      entry={pEntry}
      expanded={!cCollapsedQuants.has(pEntry.key)}
      indent={pIndent}
      onToggle={() => fToggleQuants(pEntry.key)}
    />

    <CollapsibleRegion open={!cCollapsedQuants.has(pEntry.key)}>
      {#each pEntry.quants as vQuant (vQuant.id)}
        <ModelsManagerQuantRow
          indent={pIndent + 24}
          {isFavorite}
          onDelete={fRequestDelete}
          {onSelect}
          option={vQuant}
          selected={selectedId === vQuant.id}
        />
      {/each}
    </CollapsibleRegion>
  {:else}
    <ModelsManagerModelRow
      indent={pIndent}
      {isFavorite}
      onDelete={fRequestDelete}
      {onSelect}
      option={pEntry.base}
      selected={selectedId === pEntry.base.id}
    />
  {/if}
{/snippet}

{#snippet familyRow({
  expanded: vExpanded,
  group: vFamily,
  toggle: fToggle
}: {
  expanded: boolean;
  group: ModelFamilyGroup<ModelQuantGroup>;
  toggle: () => void;
})}
  {@const cCountLabel = `${vFamily.entries.length} model${vFamily.entries.length === 1 ? '' : 's'}`}

  <div
    class="{cModelRowGridClass} relative group cursor-pointer rounded-md px-2 py-1 transition hover:bg-muted/40 max-md:px-3 max-md:py-2.5"
    onclick={fToggle}
    onkeydown={(pEvent) => fHandleFamilyKeydown(pEvent, fToggle)}
    role="button"
    tabindex="0"
  >
    <span class="flex min-w-0 items-center gap-3 max-md:pr-9">
      <ModelAvatar
        option={vFamily.entries[0].base}
        showBaseModelAvatar
        showQuantBadge={false}
        size="size-6"
      />

      <span class="truncate text-sm font-medium">{vFamily.label}</span>

      <span class="text-sm text-muted-foreground">{cCountLabel}</span>
    </span>

    <!-- the context and status columns stay empty here: the fold control is this
		     row's last cell, which a phone floats at the row's end -->
    <span class="max-md:hidden"></span>

    <span class="max-md:hidden"></span>

    <span
      class="flex justify-center {cModelRowTrailingCellClass} {vExpanded
        ? 'opacity-0 group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100'
        : ''}"
    >
      {#if vExpanded}
        <ChevronUp class="h-3.5 w-3.5 text-muted-foreground" />
      {:else}
        <ChevronDown class="h-3.5 w-3.5 text-muted-foreground" />
      {/if}
    </span>
  </div>
{/snippet}

{#snippet listItem({ depth: vDepth, entry: vEntry }: { depth: number; entry: ModelQuantGroup })}
  {@render entryTree(vEntry, vDepth > 0 ? 16 : 0)}
{/snippet}

{#snippet showMore({
  count: vCount,
  onMore: fOnMore,
  unit: vUnit
}: {
  count: number;
  onMore: () => void;
  unit: cGroupedListUnit;
})}
  <div class="px-2 max-md:px-3">
    <button
      class="w-full cursor-pointer rounded-md px-2 py-2 text-left text-xs text-muted-foreground transition hover:bg-muted/40"
      onclick={fOnMore}
      type="button"
    >
      {fTranslate('dgs_0df6f1cad36c')}
      {vCount}
      {fTranslate('text_187897ce0afc')}
      {dShowMoreNouns[vUnit]}
    </button>
  </div>
{/snippet}

<DialogConfirmDownload
  action={ModelDownloadConfirmAction.DELETE}
  onClose={() => (vDeleteOpen = false)}
  open={vDeleteOpen}
  repoWithTag={vPendingDelete}
/>

<div class="flex h-full min-h-0 flex-col">
  <ModelsManagerTableToolbar
    bind:capabilities
    bind:contextLimit
    bind:filter
    bind:modalities
    {toolbarEnd}
  />

  <div
    class="{cModelRowGridClass} shrink-0 border-y border-border/40 px-2 py-2 text-[11px] font-semibold tracking-wide text-muted-foreground uppercase max-md:px-3"
  >
    <span>{@render sortHeader(ModelsTableSortKey.NAME, 'Model')}</span>

    <span class="text-right whitespace-nowrap max-md:hidden">
      {@render sortHeader(ModelsTableSortKey.CONTEXT, 'Context')}
    </span>

    <span class="justify-self-center max-md:hidden">{fTranslate('dgs_920e413c7d41')}</span>

    <span class="text-center max-md:hidden">{fTranslate('dgs_ff8059dc6752')}</span>
  </div>

  <div class="min-h-0 flex-1 overflow-y-auto">
    {#each aSections as vGroup (vGroup.key)}
      {#if vGroup.items.length > 0}
        {#snippet groupIcon()}
          {#if vGroup.kind === ModelsTableGroupKind.cDownloading}
            <Download class="h-3.5 w-3.5 shrink-0" />
          {:else if vGroup.kind === ModelsTableGroupKind.cFavorites}
            <Heart class="h-3.5 w-3.5 shrink-0" />
          {:else if vGroup.kind === ModelsTableGroupKind.cLoaded}
            <Power class="h-3.5 w-3.5 shrink-0" />
          {:else if vGroup.kind === ModelsTableGroupKind.cHidden}
            <EyeOff class="h-3.5 w-3.5 shrink-0" />
          {:else if vGroup.kind === ModelsTableGroupKind.cLocal}
            <Logo class="shrink-0" style="--size: 0.875rem" />
          {/if}
        {/snippet}

        <ModelsSection
          chevronClass="mr-7"
          count={vGroup.items.length}
          defaultOpen={vGroup.kind !== ModelsTableGroupKind.cHidden}
          icon={groupIcon}
          label={vGroup.label}
          persistKey={vGroup.key}
          revealChevronOnHover
          sectionHeaderClass="m-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground select-none max-md:px-3"
          sticky
          stickyClass="sticky z-10 bg-muted/90 backdrop-blur-lg"
        >
          <GroupedList
            group={familyRow}
            groupWindow={cFamilyRowWindow}
            groups={vGroup.flat ? null : fFamilyGroups(vGroup)}
            initialCollapsed={cModelsStore.mCollapsedGroupsUnder(vGroup.key)}
            item={listItem}
            items={vGroup.flat ? vGroup.items : []}
            keyOf={(pEntry) => pEntry.key}
            more={showMore}
            onCollapsedChange={(pKey, pCollapsed) =>
              cModelsStore.mSetGroupCollapsed(vGroup.key, pKey, pCollapsed)}
            sectionWindow={cModelRowWindow}
            stickyClass="bg-muted/30 backdrop-blur-lg"
            stickyStyle={cModelsTableGroupStickyOffset}
            weightOf={(pEntry) => pEntry.quants.length}
          />
        </ModelsSection>
      {/if}
    {/each}

    {#if vIsEmpty}
      <p class="px-4 py-10 text-center text-sm text-muted-foreground">
        {vHasFilters ? 'No models match these filters.' : 'No models found.'}
      </p>
    {/if}
  </div>
</div>
