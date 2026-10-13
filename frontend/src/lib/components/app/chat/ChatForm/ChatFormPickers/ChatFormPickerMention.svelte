<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { File, Folder } from '@lucide/svelte';
  import { ChatFormPickerList, ChatFormPickerListItem } from '$lib/components/app/chat';
  import HighlightedMatch from '$lib/components/app/forms/HighlightedMatch.svelte';
  import * as Popover from '$lib/components/ui/popover';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cFileGlobSearchPickers, cHomeTilde, SEARCH } from '$lib/constants';
  import {
    BuiltInTool,
    FileMentionEntryType,
    GlobSearchType,
    KeyboardKey,
    ToolSource
  } from '$lib/enums';
  import { fUseDebouncedSearch } from '$lib/hooks/use-debounced-search.svelte';
  import { fUsePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
  import { cConversationsStore, cDeviceStore, cSettingsStore, cToolsStore } from '$lib/stores';
  import type { FileMentionEntry, GlobEntryResult } from '$lib/types';
  import { fAbbreviateHome, fRunGlobSearchWithChildren } from '$lib/utils';

  /**
   * Floating file/folder mention picker. The chat input is the search
   * surface: `query` (typed after `@`) drives a `file_glob_search` tool
   * call scoped to `scopePath`. The parent owns the "dismissed token,
   * don't re-open until it changes" snapshot.
   */
  interface Props {
    class?: string;
    isOpen: boolean;
    query: string;
    customAnchor?: HTMLElement | null;
    scopePath?: string | null;
    onClose: () => void;
    onSelect: (pEntry: FileMentionEntry) => void;
    /** Fired when `isOpen` becomes true, so the host can keep focus on the chat input. */
    onOpened?: () => void;
  }

  let {
    class: className = '',
    customAnchor = null,
    isOpen,
    onClose,
    onOpened,
    onSelect,
    query,
    scopePath = null
  }: Props = $props();

  const cNav = fUsePickerNavigation({
    count: () => aDisplayedItems.length,
    isOpen: () => isOpen,
    onClose: () => onClose(),
    onSelect: (pIndex) => fHandleSelect(aDisplayedItems[pIndex])
  });

  // When the server does not expose file_glob_search (started without
  // --tools) or the user disabled it, the picker still opens but explains
  // why instead of firing searches that would only fail.
  const cFileSearchKey = $derived(cToolsStore.mGetPermissionKey(BuiltInTool.cServerFileGlobSearch));
  // effective policy: the active conversation's tool policy, or global defaults
  const cFileSearchEnabled = $derived(
    cFileSearchKey !== null &&
      cConversationsStore.preferences.mIsToolEnabled(cFileSearchKey) &&
      cConversationsStore.preferences.mIsCategoryEnabled(ToolSource.SERVER)
  );

  let aSearchResults = $state<FileMentionEntry[]>([]);
  let vSearchError = $state<string | null>(null);

  // Coerce the depth setting to a positive integer; an invalid value
  // would otherwise reach the server as max_depth 0 = unlimited.
  const cSearchDepth = $derived.by(() => {
    const cN = Number(cSettingsStore.vConfig.mentionSearchMaxDepth);

    return Number.isInteger(cN) && cN > 0 ? cN : cFileGlobSearchPickers.DEFAULT_SEARCH_DEPTH;
  });

  const cHome = $derived(cToolsStore.serverHome);

  // A smaller window than the WD picker suffices: entries are ranked client-side.
  const cMentionSearchLimit = 50;

  const search = fUseDebouncedSearch({
    canRun: () => isOpen && cFileSearchEnabled,
    debounceMs: SEARCH.DEBOUNCE_MS,
    getQuery: () => cTrimmedQuery,
    mRun: async (pQuery, pSignal, pIsCurrent) => {
      try {
        // A trailing path separator targets a directory, so also list its
        // children. Accept both `/` and `\`.
        const cRes = await fRunGlobSearchWithChildren(
          pQuery,
          scopePath ?? cHome ?? cHomeTilde,
          cSearchDepth,
          cMentionSearchLimit,
          pSignal,
          { descendOnTrailingSeparator: true, type: GlobSearchType.cAll }
        );

        if (!pIsCurrent()) return;

        if (cRes.error) {
          aSearchResults = [];
          vSearchError = cRes.error;

          return;
        }

        const fToEntry = (pE: GlobEntryResult): FileMentionEntry => ({
          name: pE.name,
          path: pE.path,
          type: pE.type === 'dir' ? FileMentionEntryType.cDirectory : FileMentionEntryType.FILE
        });

        aSearchResults = cRes.entries.map(fToEntry);
        vSearchError = null;
      } catch (vErr) {
        if (!pIsCurrent() || pSignal.aborted) return;

        aSearchResults = [];
        vSearchError = vErr instanceof Error ? vErr.message : String(vErr);
      }
    }
  });

  const cTrimmedQuery = $derived((query ?? '').trim());
  const aDisplayedItems = $derived(aSearchResults);

  const cEmptyMessage = $derived.by(() => {
    if (cFileSearchKey === null) {
      return 'File search is unavailable on this server (started without --tools)';
    }

    if (!cFileSearchEnabled) {
      return 'File search is disabled - enable "Search files" in Settings > Tools to use @-mentions';
    }

    return vSearchError ? `Search failed - ${vSearchError}` : 'No matching files or folders';
  });

  const cShowTooltip = $derived(!cDeviceStore.isMobile);

  $effect(() => {
    if (typeof window === 'undefined') return;

    void cToolsStore.mResolveServerHome();
  });

  $effect(() => {
    if (isOpen) {
      cNav.reset(0);
    }
  });

  $effect(() => {
    if (isOpen) onOpened?.();
  });

  $effect(() => {
    const cQ = (query ?? '').trim();

    if (!isOpen || !cQ || !cFileSearchEnabled) {
      search.cancel();
      aSearchResults = [];
      vSearchError = null;

      return;
    }

    search.mSetLoading(true);
    search.mRun(cQ);
  });

  function fHandleSelect(pEntry: FileMentionEntry) {
    onSelect(pEntry);
    onClose();
  }

  export function handleKeydown(pEvent: KeyboardEvent): boolean {
    // Always consume Enter while the picker is open - even with no
    // result yet (skeletons) or no matches - so the chat form's
    // Enter-to-submit never fires mid-search.
    if (isOpen && pEvent.key === KeyboardKey.cEnter) {
      pEvent.preventDefault();

      if (cNav.hoveredIndex >= 0 && aDisplayedItems[cNav.hoveredIndex]) {
        fHandleSelect(aDisplayedItems[cNav.hoveredIndex]);
      }

      return true;
    }

    return cNav.handleKeydown(pEvent);
  }
</script>

<Popover.Root
  onOpenChange={(pOpen) => {
    if (!pOpen) onClose();
  }}
  open={isOpen}
>
  <!-- Invisible form-wide trigger: stops bits-ui's outside-click detector
	     from closing the picker when the user clicks inside the textarea.
	     We open programmatically via `open={isOpen}`, so it is inert
	     (tabindex=-1 + pointer-events-none + opacity-0 + aria-hidden).
	     Positioning comes from `customAnchor` at the form's top edge. -->
  <Popover.Trigger
    aria-hidden="true"
    class="pointer-events-none absolute inset-0 opacity-0"
    tabindex={-1}
  >
    <span class="sr-only">{fTranslate('dgs_413527207d18')}</span>
  </Popover.Trigger>

  <Popover.Content
    align="start"
    class={[
      'w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl',
      className
    ]}
    {customAnchor}
    onCloseAutoFocus={(pEvent) => pEvent.preventDefault()}
    onOpenAutoFocus={(pEvent) => pEvent.preventDefault()}
    onkeydown={handleKeydown}
    preventScroll={false}
    side="top"
    sideOffset={12}
  >
    <ChatFormPickerList
      emptyMessage={cEmptyMessage}
      isLoading={search.isSearching}
      itemKey={(pEntry) => pEntry.type + ':' + pEntry.path}
      items={aDisplayedItems}
      scrollTrigger={cNav.scrollTrigger}
      searchQuery={query ?? ''}
      selectedIndex={cNav.hoveredIndex}
      showSearchInput={false}
    >
      {#snippet item(pEntry, pIndex, pIsSelected)}
        <ChatFormPickerListItem
          dataIndex={pIndex}
          isSelected={pIsSelected}
          onclick={() => fHandleSelect(pEntry)}
          onmouseenter={() => cNav.setHover(pIndex)}
        >
          {@const Icon = pEntry.type === FileMentionEntryType.cDirectory ? Folder : File}
          <Icon
            class={[
              'mt-0.5 h-4 w-4 shrink-0',
              pEntry.type === FileMentionEntryType.cDirectory
                ? 'text-amber-500'
                : 'text-muted-foreground'
            ]}
          />

          <div class="flex min-w-0 flex-1 flex-col">
            <div class="flex min-w-0 items-center gap-2">
              {#if cShowTooltip}
                <Tooltip.fRoot>
                  <Tooltip.Trigger>
                    {#snippet child({ props: dProps })}
                      <span {...dProps} class="truncate text-sm font-medium">{pEntry.name}</span>
                    {/snippet}
                  </Tooltip.Trigger>

                  <Tooltip.Content>
                    <p>{pEntry.path}</p>
                  </Tooltip.Content>
                </Tooltip.fRoot>
              {:else}
                <span class="truncate text-sm font-medium">{pEntry.name}</span>
              {/if}

              <span
                class="shrink-0 rounded-full bg-muted px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wide text-muted-foreground"
              >
                {pEntry.type}
              </span>
            </div>

            <span class="min-w-0 flex-1 truncate font-mono text-left text-xs">
              <HighlightedMatch query={cTrimmedQuery} text={fAbbreviateHome(pEntry.path, cHome)} />
            </span>
          </div>
        </ChatFormPickerListItem>
      {/snippet}
    </ChatFormPickerList>
  </Popover.Content>
</Popover.Root>
