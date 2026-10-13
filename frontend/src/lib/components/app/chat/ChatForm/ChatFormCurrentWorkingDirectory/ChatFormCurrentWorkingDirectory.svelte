<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ChatFormCurrentWorkingDirectoryChip from './ChatFormCurrentWorkingDirectoryChip.svelte';
  import ChatFormCurrentWorkingDirectoryResultsList from './ChatFormCurrentWorkingDirectoryResultsList.svelte';
  import { FolderOpen } from '@lucide/svelte';
  import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
  import * as Popover from '$lib/components/ui/popover';
  import { cDefaultMobileBreakpoint, cHomeTilde, SEARCH, cUiDataAttrs } from '$lib/constants';
  import { BuiltInTool, GlobSearchType, KeyboardKey, ToolSource } from '$lib/enums';
  import { fUseDebouncedSearch } from '$lib/hooks/use-debounced-search.svelte';
  import { fUsePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
  import { fUseScrollActiveRow } from '$lib/hooks/use-scroll-active-row.svelte';
  import { ToolsService } from '$lib/services/tools.service';
  import { cConversationsStore, cToolsStore } from '$lib/stores';
  import type { GlobEntry } from '$lib/types';
  import {
    fAbbreviateHome,
    fBuildCaseInsensitiveGlob,
    fJoinPath,
    fLastPathSegment,
    fRunGlobSearchWithChildren
  } from '$lib/utils';

  // Microtask delay so the popover's focus scope tears down first.
  const cFocusDelayMs = 0;

  interface Props {
    class?: string;
    disabled?: boolean;
    directory?: string | null;
    /** Controlled open state; the host owns it so the chip click and the
     * `/cwd` slash command open the picker through the same path. */
    isOpen: boolean;
    /** Two-way bound query, kept in sync with the text after `/cwd `. */
    query: string;
    /** Anchor at the form's top edge so the popover floats above the box. */
    customAnchor?: HTMLElement | null;
    onChange?: (pDirectory: string | null) => void;
    /** Lets the host refocus the chat input after the popover closes. */
    onClose?: () => void;
    /** Fired when the chip is clicked so the host can open the picker. */
    onOpen?: () => void;
  }

  let {
    class: className = '',
    customAnchor = null,
    directory = null,
    disabled = false,
    isOpen,
    onChange,
    onClose,
    onOpen,
    query = $bindable('')
  }: Props = $props();

  // File System Access API is opt-in (Chrome / Edge / Opera): the popover
  // exposes a "Browse" button only when available.
  const cPickerSupported =
    typeof window !== 'undefined' && typeof window.showDirectoryPicker === 'function';

  // When the server does not serve file_glob_search or the user disabled
  // it, the picker still opens for manual entry but explains why search is
  // unavailable instead of firing searches that would only fail. Browse is
  // hidden too: it resolves the picked folder name through the same tool.
  const cFileSearchKey = $derived(cToolsStore.mGetPermissionKey(BuiltInTool.cServerFileGlobSearch));
  // effective policy: the active conversation's tool policy, or global defaults
  const cFileSearchEnabled = $derived(
    cFileSearchKey !== null &&
      cConversationsStore.preferences.mIsToolEnabled(cFileSearchKey) &&
      cConversationsStore.preferences.mIsCategoryEnabled(ToolSource.SERVER)
  );
  const cSearchUnavailableMessage = $derived(
    cFileSearchKey === null
      ? 'File search is unavailable on this server - type a full path and press Enter'
      : 'File search is disabled - type a full path and press Enter, or enable "Search files" in Settings > Tools'
  );

  let vSearchInputRef: HTMLInputElement | null = $state(null);

  let aQueryResults = $state<string[]>([]);
  let vSearchError = $state<string | null>(null);
  let vListContainer = $state<HTMLDivElement | null>(null);

  const cNav = fUsePickerNavigation({
    count: () => aQueryResults.length,
    isOpen: () => isOpen,
    onClose: fClosePicker,
    onSelect: (pIndex) => fCommit(aQueryResults[pIndex])
  });

  let vHomeBase = $derived(cToolsStore.serverHome);

  // Resolve home eagerly so the chip can abbreviate before the picker opens.
  $effect(() => {
    if (typeof window === 'undefined') return;

    void cToolsStore.mResolveServerHome();
  });

  // HTML `autofocus` is unreliable on dynamically shown elements.
  $effect(() => {
    if (!isOpen) return;

    setTimeout(() => vSearchInputRef?.focus(), cFocusDelayMs);
  });

  $effect(() => {
    if (!isOpen) return;

    const cQ = query.trim();

    cNav.reset(-1);

    if (cQ && cFileSearchEnabled) {
      search.mRun(cQ);
    } else {
      search.cancel();
      aQueryResults = [];
      vSearchError = null;
      cNav.reset(-1);
      vSearchScope = vHomeBase ?? cHomeTilde;
    }
  });

  fUseScrollActiveRow({
    dataAttr: cUiDataAttrs.RESULT_INDEX,
    getContainer: () => vListContainer,
    getCount: () => aQueryResults.length,
    getIndex: () => cNav.hoveredIndex,
    getTrigger: () => cNav.scrollTrigger
  });

  let vSearchScope = $state(cHomeTilde);

  // An exactly-typed directory is "entered": the shared search lists its
  // children too, so path navigation does not require a trailing slash.
  const search = fUseDebouncedSearch({
    canRun: () => isOpen && cFileSearchEnabled,
    debounceMs: SEARCH.DEBOUNCE_MS,
    getQuery: () => query.trim(),
    mRun: async (pQ, pSignal, pIsCurrent) => {
      const cTrimmed = pQ.trim();

      if (!cTrimmed) {
        aQueryResults = [];
        vSearchError = null;
        cNav.reset(-1);
        vSearchScope = vHomeBase ?? cHomeTilde;

        return;
      }

      try {
        // Generous limit: ranking is client-side, only the top
        // MAX_RESULTS_SHOWN are shown.
        const cRes = await fRunGlobSearchWithChildren(
          cTrimmed,
          vHomeBase ?? cHomeTilde,
          SEARCH.MAX_DEPTH,
          SEARCH.LIMIT,
          pSignal,
          { type: GlobSearchType.cDir }
        );

        if (!pIsCurrent()) return;

        if (cRes.error) {
          aQueryResults = [];
          cNav.reset(-1);
          vSearchError = cRes.error;

          return;
        }

        vSearchScope = cRes.exactDir ?? cRes.args.path;
        aQueryResults = cRes.entries.map((pE) => pE.path).slice(0, SEARCH.MAX_RESULTS_SHOWN);

        if (aQueryResults.length > 0) {
          cNav.reset(0);
          cNav.bumpScroll(); // scroll the list back to the top (first item is hovered)
        } else {
          cNav.reset(-1);
        }

        vSearchError = null;
      } catch (vErr) {
        if (!pIsCurrent() || pSignal.aborted) return;

        aQueryResults = [];
        cNav.reset(-1);
        vSearchError = vErr instanceof Error ? vErr.message : String(vErr);
      }
    }
  });
  // Single funnel for every local close so the host refocus always fires.
  function fClosePicker() {
    onClose?.();
  }

  function fCommit(pPath: string) {
    onChange?.(pPath);
    fClosePicker();
  }

  function fSetDirectory(pValue: string) {
    const cTrimmed = pValue.trim();

    if (!cTrimmed) return;

    onChange?.(cTrimmed);
  }

  // Resolve a browser-picked folder name (which exposes only the leaf name)
  // to a server-side absolute path; null when the server cannot locate it,
  // so the caller fails visibly instead of committing a bare leaf name.
  async function fResolveNativeName(pName: string): Promise<string | null> {
    try {
      const dRes = await ToolsService.mExecuteToolRaw(BuiltInTool.cServerFileGlobSearch, {
        include: fBuildCaseInsensitiveGlob(pName),
        limit: SEARCH.NATIVE_LIMIT,
        max_depth: SEARCH.NATIVE_MAX_DEPTH,
        path: vHomeBase ?? cHomeTilde,
        type: GlobSearchType.cDir
      });
      const cBase = typeof dRes.base === 'string' ? dRes.base : '';
      const entries = Array.isArray(dRes.entries) ? (dRes.entries as GlobEntry[]) : [];
      const cMatch = entries.find(
        (pE) => fLastPathSegment(pE.path).toLowerCase() === pName.toLowerCase()
      );

      return cMatch ? fJoinPath(cBase, cMatch.path) : null;
    } catch {
      return null;
    }
  }

  async function fBrowseNative() {
    if (disabled || !window.showDirectoryPicker) return;

    try {
      const cHandle = await window.showDirectoryPicker();
      const cPath = await fResolveNativeName(cHandle.name);

      if (cPath) {
        fSetDirectory(cPath);
        fClosePicker();
      } else {
        // keep the previous cwd and fail visibly instead of committing a
        // bare leaf name that would resolve against the server cwd
        vSearchError = `Could not resolve "${cHandle.name}" to a server path`;
      }
    } catch (vErr) {
      // user cancelled - silently ignore; other errors are logged
      if (vErr instanceof DOMException && vErr.name === 'AbortError') return;

      console.error('[ChatFormCurrentWorkingDirectory] showDirectoryPicker failed:', vErr);
    }
  }

  function fHandleSubmit() {
    const cValue = query.trim();

    if (!cValue) {
      fClosePicker();

      return;
    }

    fSetDirectory(cValue);
    fClosePicker();
  }

  function fHandleKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === KeyboardKey.cEnter) {
      pEvent.preventDefault();

      if (cNav.hoveredIndex >= 0 && aQueryResults[cNav.hoveredIndex]) {
        fCommit(aQueryResults[cNav.hoveredIndex]);
      } else if (aQueryResults.length === 0) {
        fHandleSubmit();
      }
    } else if (pEvent.key === KeyboardKey.cArrowDown) {
      if (aQueryResults.length > 0) {
        pEvent.preventDefault();
        cNav.move(1);
      }
    } else if (pEvent.key === KeyboardKey.cArrowUp) {
      if (aQueryResults.length > 0) {
        pEvent.preventDefault();
        cNav.move(-1);
      }
    }
  }

  function fClearDirectory(pEvent?: MouseEvent) {
    // Stop the click from bubbling into the chip button and re-opening
    // the picker on top of the now-cleared state.
    pEvent?.stopPropagation();
    pEvent?.preventDefault();
    onChange?.(null);
    fClosePicker();
  }

  function fHandleDismiss(pEvent?: MouseEvent) {
    pEvent?.stopPropagation();
    pEvent?.preventDefault();

    if (directory) {
      fClearDirectory(pEvent);
    }
  }

  function fHandleOpenChange(pOpen: boolean) {
    if (pOpen) {
      void cToolsStore.mResolveServerHome();
    } else {
      search.cancel();
      // bits-ui-initiated close (Escape on the content, outside-click) -
      // the only path that bypasses closePicker().
      onClose?.();
    }
  }

  let vInnerWidth = $state(0);
  const cShowTooltip = $derived(vInnerWidth > cDefaultMobileBreakpoint);
</script>

<button
  class={[
    'justify-self-start flex min-w-0 w-auto items-center gap-1 mt-1.5 py-1 px-2 backdrop-blur-2xl rounded-md',
    className
  ]}
  {disabled}
  onclick={onOpen}
  type="button"
>
  <ChatFormCurrentWorkingDirectoryChip
    {directory}
    {disabled}
    homeBase={vHomeBase}
    onClear={fHandleDismiss}
    showTooltip={cShowTooltip}
  />
</button>

<Popover.Root onOpenChange={fHandleOpenChange} open={isOpen}>
  <Popover.Trigger
    aria-hidden="true"
    class="pointer-events-none absolute inset-0 opacity-0"
    tabindex={-1}
  >
    <span class="sr-only">{fTranslate('dgs_58ee76db840b')}</span>
  </Popover.Trigger>

  <Popover.Content
    align="start"
    class="w-[var(--bits-popover-anchor-width)] max-w-none rounded-xl border-border/50 p-0 shadow-xl"
    {customAnchor}
    onCloseAutoFocus={(pEvent) => pEvent.preventDefault()}
    onOpenAutoFocus={(pEvent) => pEvent.preventDefault()}
    onkeydown={fHandleKeydown}
    preventScroll={false}
    side="top"
    sideOffset={12}
  >
    <div class="p-2 min-h-22 flex flex-col justify-between">
      <SearchInput
        bind:ref={vSearchInputRef}
        bind:value={query}
        class="w-full"
        onClose={fClosePicker}
        placeholder={fTranslate('dgs_f8df672c258b')}
      />

      {#if !cFileSearchEnabled}
        <div class="px-2 py-1.5 text-sm text-muted-foreground">{cSearchUnavailableMessage}</div>
      {:else if query.trim() && (search.isSearching || aQueryResults.length > 0 || vSearchError)}
        <ChatFormCurrentWorkingDirectoryResultsList
          bind:container={vListContainer}
          error={vSearchError}
          hoveredIndex={cNav.hoveredIndex}
          isSearching={search.isSearching}
          onCommit={fCommit}
          onHover={(pIndex) => cNav.setHover(pIndex)}
          rawQuery={query}
          results={aQueryResults}
        />
      {/if}

      {#if cPickerSupported && cFileSearchEnabled}
        <button
          class="-mt-1 flex cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none hover:bg-accent hover:text-accent-foreground"
          onclick={fBrowseNative}
          type="button"
        >
          <FolderOpen class="size-4 shrink-0 text-muted-foreground" />

          <span>{fTranslate('dgs_3227aa966625')}</span>
        </button>
      {/if}

      {#if vHomeBase && cFileSearchEnabled}
        <div aria-hidden="true" class="-mx-2 my-2 h-px bg-border/20"></div>

        <span class="px-2 py-1.5 font-mono text-[10px]">
          {fTranslate('dgs_40de76ef13dd')}

          <span class="truncate text-muted-foreground/70" title={vSearchScope}
            >{fAbbreviateHome(vSearchScope, vHomeBase)}</span
          >
        </span>
      {/if}
    </div>
  </Popover.Content>
</Popover.Root>

<svelte:window bind:innerWidth={vInnerWidth} />
