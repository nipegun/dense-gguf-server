<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelsSelectorDownloadItem from './ModelsSelectorDownloadItem.svelte';
  import { Heart, Power } from '@lucide/svelte';
  import {
    GroupedList,
    ModelAvatar,
    ModelsSection,
    ModelsSelectorOption
  } from '$lib/components/app';
  import { DialogConfirmDownload } from '$lib/components/app/dialogs';
  import Logo from '$lib/components/app/misc/Logo.svelte';
  import {
    cLocalBackendId,
    cModelRowWindow,
    cSelectorGroupStickyOffset,
    cSettingsKeys
  } from '$lib/constants';
  import { ModelDownloadConfirmAction } from '$lib/enums';
  import { cModelsStore, cSettingsStore } from '$lib/stores';
  import {
    type GroupedModelOptions,
    fGroupModelFamilies,
    type ModelFamilyGroup,
    type ModelItem,
    fWindowLocalGroups
  } from '$lib/utils';

  interface Props {
    groups: GroupedModelOptions;
    currentModel: string | null;
    activeId: string | null;
    sectionHeaderClass?: string;
    onSelect: (pModelId: string) => void;
    renderOption?: import('svelte').Snippet<[ModelItem, boolean]>;
    /** Favorite models of every backend, listed in their own section. */
    favorites?: ModelItem[];
    /** Loaded models of every dgs-compat backend, leading the list. */
    loaded?: ModelItem[];
    /** Show the organization name in every model id of the list. */
    showOrgName?: boolean;
  }

  let {
    activeId,
    currentModel,
    favorites = [],
    groups,
    loaded = [],
    onSelect,
    renderOption,
    sectionHeaderClass = 'm-0 px-2 py-2 text-[13px] font-semibold text-muted-foreground/70 select-none',
    showOrgName = true
  }: Props = $props();
  let fRender = $derived(renderOption ?? defaultOption);
  // a large catalog mounts a window at a time: the sentinel at the end of the list
  // grows the window when it scrolls into view
  let vVisibleCount = $state(cModelRowWindow);
  let vSentinelEl = $state<HTMLElement | null>(null);
  const cLocalGroups = $derived(fWindowLocalGroups(groups, vVisibleCount));
  const cLocalRowCount = $derived(
    groups.available.reduce((pCount, pGroup) => pCount + pGroup.items.length, 0)
  );
  const cHasMoreLocal = $derived(cLocalGroups.shown < cLocalRowCount);

  $effect(() => {
    const cSentinel = vSentinelEl;

    if (!cSentinel || !cHasMoreLocal) return;

    // scroll does not bubble, so listen in the capture phase and ask the
    // sentinel where it is instead of guessing which ancestor scrolls
    const fOnScroll = () => {
      const cRect = cSentinel.getBoundingClientRect();

      // grow only when the end of the list is on screen: otherwise every scroll
      // event of the page would mount the whole catalog
      if (cRect.top > window.innerHeight || cRect.bottom < 0) return;

      vVisibleCount += cModelRowWindow;
    };

    document.addEventListener('scroll', fOnScroll, { capture: true, passive: true });
    fOnScroll();

    return () => document.removeEventListener('scroll', fOnScroll, { capture: true });
  });
  // section headers stick below the dropdown's search block: `--dropdown-sticky-height`
  // comes from DropdownMenuSearchable and falls back to 0 in the mobile sheet
  let vHeaderClass = $derived(`${sectionHeaderClass} sticky z-10 bg-popover`);
  const cHeaderStyle = 'top: var(--dropdown-sticky-height, 0px)';

  /** In-flight / paused downloads, tracked by the status feed. */
  // only a running download belongs in the selector; a paused one waits in the
  // manager table, where its partial files are resumed or dropped
  let aDownloadEntries = $derived(
    cModelsStore.status.mGetDownloadEntries().filter((pEntry) => !pEntry.isPaused)
  );

  // cancel is confirmed once for the whole list, so one dialog serves every row
  // The target is kept while the dialog closes so its copy stays rendered.
  let vPendingCancel = $state('');
  let vCancelOpen = $state(false);

  function fRequestCancel(pRepoWithTag: string) {
    vPendingCancel = pRepoWithTag;
    vCancelOpen = true;
  }
</script>

{#snippet familyHeading({ group: vFamily }: { group: ModelFamilyGroup<ModelItem> })}
  <div class="flex items-center gap-2 px-2 py-1.5 select-none max-md:py-2.5">
    <ModelAvatar
      option={vFamily.entries[0].option}
      showBaseModelAvatar
      showQuantBadge={false}
      size="size-6"
    />

    <span class="truncate text-[13px] font-semibold text-muted-foreground/70">
      {vFamily.label}
    </span>

    <span class="text-xs text-muted-foreground">
      {vFamily.entries.length}
      {fTranslate('dgs_9372c470eead')}{vFamily.entries.length === 1 ? '' : 's'}
    </span>
  </div>
{/snippet}

{#snippet listItem({ depth: vDepth, entry: vEntry }: { depth: number; entry: ModelItem })}
  <div style="padding-left: {vDepth * 16}px">{@render fRender(vEntry, !showOrgName)}</div>
{/snippet}

{#snippet listRows(pItems: ModelItem[], pPrefix: string)}
  {#if cSettingsStore.vConfig[cSettingsKeys.GROUP_MODELS_BY_FAMILY]}
    <GroupedList
      group={familyHeading}
      groups={fGroupModelFamilies(pItems, (pRow) => pRow.option.model).map((pFamily) => ({
        entries: pFamily.entries,
        group: pFamily,
        key: `${pPrefix}-${pFamily.key}`
      }))}
      initialCollapsed={cModelsStore.mCollapsedGroupsUnder(pPrefix)}
      item={listItem}
      keyOf={(pRow) => `${pPrefix}-${pRow.option.id}`}
      onCollapsedChange={(pKey, pCollapsed) =>
        cModelsStore.mSetGroupCollapsed(pPrefix, pKey, pCollapsed)}
      stickyStyle={cSelectorGroupStickyOffset}
    />
  {:else}
    <GroupedList item={listItem} items={pItems} keyOf={(pRow) => `${pPrefix}-${pRow.option.id}`} />
  {/if}
{/snippet}

{#snippet defaultOption(pItem: ModelItem, pHideOrgName: boolean)}
  {@const { option: vOption } = pItem}
  {@const cIsSelected = currentModel === vOption.model || activeId === vOption.id}
  {@const cIsFav = cModelsStore.sFavoriteModelIds.has(vOption.model)}

  <ModelsSelectorOption
    hideOrgName={pHideOrgName}
    isFav={cIsFav}
    isHighlighted={false}
    isSelected={cIsSelected}
    onKeyDown={() => {}}
    onMouseEnter={() => {}}
    {onSelect}
    option={vOption}
  />
{/snippet}

{#if loaded.length > 0}
  <ModelsSection
    count={loaded.length}
    label={fTranslate('text_fb65b87d1b5e')}
    persistKey="loaded"
    revealChevronOnHover
    sticky
  >
    {#snippet icon()}
      <Power class="h-3.5 w-3.5 shrink-0" />
    {/snippet}

    {#each loaded as vItem (vItem.option.id)}
      {@render fRender(vItem, !showOrgName)}
    {/each}
  </ModelsSection>
{/if}

{#if favorites.length > 0}
  <!-- Favorites come first; the sections below skip them -->
  <ModelsSection
    label={fTranslate('dgs_7a1f2a83aca9')}
    persistKey="favorites"
    revealChevronOnHover
    sticky
  >
    {#snippet icon()}
      <Heart class="h-3.5 w-3.5 shrink-0" />
    {/snippet}

    {#each favorites as vItem (`fav-${vItem.option.id}`)}
      {@render fRender(vItem, !showOrgName)}
    {/each}
  </ModelsSection>
{/if}

{#if aDownloadEntries.length > 0}
  <p class={vHeaderClass} style={cHeaderStyle}>{fTranslate('dgs_37b345555d7e')}</p>

  {#each aDownloadEntries as vEntry (vEntry.repoWithTag)}
    <ModelsSelectorDownloadItem entry={vEntry} onRequestCancel={fRequestCancel} {showOrgName} />
  {/each}
{/if}

{#snippet localRows()}
  {@render listRows(
    cLocalGroups.available.flatMap((pGroup) => pGroup.items),
    'local'
  )}

  {#if cHasMoreLocal}
    <div bind:this={vSentinelEl} aria-hidden="true" class="h-px"></div>
  {/if}
{/snippet}

{#if groups.available.length > 0}
  <ModelsSection
    label={fTranslate('dgs_8e4bf4368b03')}
    persistKey={cLocalBackendId}
    revealChevronOnHover
    sticky
  >
    {#snippet icon()}
      <Logo class="shrink-0" style="--size: 0.875rem" />
    {/snippet}

    {@render localRows()}
  </ModelsSection>
{/if}

<DialogConfirmDownload
  action={ModelDownloadConfirmAction.cCancel}
  onClose={() => (vCancelOpen = false)}
  open={vCancelOpen}
  repoWithTag={vPendingCancel}
/>
