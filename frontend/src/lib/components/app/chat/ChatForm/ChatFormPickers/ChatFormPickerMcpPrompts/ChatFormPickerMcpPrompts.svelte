<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import {
    ChatFormPickerItemHeader,
    ChatFormPickerList,
    ChatFormPickerListItem,
    ChatFormPickerListItemSkeleton,
    ChatFormPickerPopover,
    ChatFormPromptPickerArgumentForm
  } from '$lib/components/app/chat';
  import Badge from '$lib/components/ui/badge/badge.svelte';
  import { KeyboardKey } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';
  import type { GetPromptResult, MCPPromptInfo, MCPServerSettingsEntry } from '$lib/types';
  import { fDebounce, fUuid } from '$lib/utils';
  import { SvelteMap } from 'svelte/reactivity';

  interface Props {
    class?: string;
    isOpen?: boolean;
    searchQuery?: string;
    onClose?: () => void;
    onPromptLoadStart?: (
      pPlaceholderId: string,
      pPromptInfo: MCPPromptInfo,
      pArgs?: Record<string, string>
    ) => void;
    onPromptLoadComplete?: (pPlaceholderId: string, pResult: GetPromptResult) => void;
    onPromptLoadError?: (pPlaceholderId: string, pError: string) => void;
  }

  let {
    class: className = '',
    isOpen = false,
    onClose,
    onPromptLoadComplete,
    onPromptLoadError,
    onPromptLoadStart,
    searchQuery = ''
  }: Props = $props();

  let aPrompts = $state<MCPPromptInfo[]>([]);
  let isLoading = $state(false);
  let vSelectedPrompt = $state<MCPPromptInfo | null>(null);
  let dPromptArgs = $state<Record<string, string>>({});
  let vSelectedIndex = $state(0);
  let vInternalSearchQuery = $state('');
  // Bumped on ArrowUp/ArrowDown only, so the list scrolls on keyboard
  // nav but not on hover or result changes.
  let vScrollTrigger = $state(0);
  let vPromptError = $state<string | null>(null);
  let vSelectedIndexBeforeArgumentForm = $state<number | null>(null);

  let dSuggestions = $state<Record<string, string[]>>({});
  let dLoadingSuggestions = $state<Record<string, boolean>>({});
  let vActiveAutocomplete = $state<string | null>(null);
  let vAutocompleteIndex = $state(0);

  let vServerSettingsMap = $derived.by(() => {
    const aServers = cMcpStore.mGetServers();
    const cMap = new SvelteMap<string, MCPServerSettingsEntry>();

    for (const server of aServers) {
      cMap.set(server.id, server);
    }

    return cMap;
  });

  $effect(() => {
    if (isOpen) {
      fLoadPrompts();
      vSelectedIndex = 0;
    } else {
      vSelectedPrompt = null;
      dPromptArgs = {};
      vPromptError = null;
    }
  });

  $effect(() => {
    if (aFilteredPrompts.length > 0 && vSelectedIndex >= aFilteredPrompts.length) {
      vSelectedIndex = 0;
    }
  });

  async function fLoadPrompts() {
    isLoading = true;

    try {
      const cInitialized = await cMcpStore.mEnsureInitialized();

      if (!cInitialized) {
        aPrompts = [];

        return;
      }

      aPrompts = await cMcpStore.mGetAllPrompts();
    } catch (error) {
      console.error('[ChatFormPickerMcpPrompts] Failed to load prompts:', error);
      aPrompts = [];
    } finally {
      isLoading = false;
    }
  }

  function fHandlePromptClick(pPrompt: MCPPromptInfo) {
    const aArgs = pPrompt.arguments ?? [];

    if (aArgs.length > 0) {
      vSelectedIndexBeforeArgumentForm = vSelectedIndex;
      vSelectedPrompt = pPrompt;
      dPromptArgs = {};
      vPromptError = null;

      requestAnimationFrame(() => {
        const cFirstInput = document.querySelector(`#arg-${aArgs[0].name}`) as HTMLInputElement;

        if (cFirstInput) {
          cFirstInput.focus();
        }
      });
    } else {
      fExecutePrompt(pPrompt, {});
    }
  }

  async function fExecutePrompt(pPrompt: MCPPromptInfo, pArgs: Record<string, string>) {
    vPromptError = null;

    const cPlaceholderId = fUuid();
    const cNonEmptyArgs = Object.fromEntries(
      Object.entries(pArgs).filter(([, vValue]) => vValue.trim() !== '')
    );
    const cArgsToPass = Object.keys(cNonEmptyArgs).length > 0 ? cNonEmptyArgs : undefined;

    onPromptLoadStart?.(cPlaceholderId, pPrompt, cArgsToPass);
    onClose?.();

    try {
      const cResult = await cMcpStore.mGetPrompt(pPrompt.serverName, pPrompt.name, pArgs);

      onPromptLoadComplete?.(cPlaceholderId, cResult);
    } catch (error) {
      const cErrorMessage =
        error instanceof Error ? error.message : 'Unknown error executing prompt';

      onPromptLoadError?.(cPlaceholderId, cErrorMessage);
    }
  }

  function fHandleArgumentSubmit(pEvent: SubmitEvent) {
    pEvent.preventDefault();

    if (vSelectedPrompt) {
      fExecutePrompt(vSelectedPrompt, dPromptArgs);
    }
  }

  const fFetchCompletions = fDebounce(async (pArgName: string, pValue: string) => {
    if (!vSelectedPrompt || pValue.length < 1) {
      dSuggestions[pArgName] = [];

      return;
    }

    if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
      console.log('[ChatFormPickerMcpPrompts] Fetching completions for:', {
        argName: pArgName,
        promptName: vSelectedPrompt.name,
        serverName: vSelectedPrompt.serverName,
        value: pValue
      });
    }

    dLoadingSuggestions[pArgName] = true;

    try {
      const cResult = await cMcpStore.mGetPromptCompletions(
        vSelectedPrompt.serverName,
        vSelectedPrompt.name,
        pArgName,
        pValue
      );

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log('[ChatFormPickerMcpPrompts] Autocomplete result:', {
          argName: pArgName,
          result: cResult,
          suggestionsCount: cResult?.values.length ?? 0,
          value: pValue
        });
      }

      if (cResult && cResult.values.length > 0) {
        // Filter out empty strings from suggestions
        const aFilteredValues = cResult.values.filter((pV) => pV.trim() !== '');

        if (aFilteredValues.length > 0) {
          dSuggestions[pArgName] = aFilteredValues;
          vActiveAutocomplete = pArgName;
          vAutocompleteIndex = 0;
        } else {
          dSuggestions[pArgName] = [];
        }
      } else {
        dSuggestions[pArgName] = [];
      }
    } catch (error) {
      console.error('[ChatFormPickerMcpPrompts] Failed to fetch completions:', error);
      dSuggestions[pArgName] = [];
    } finally {
      dLoadingSuggestions[pArgName] = false;
    }
  }, 200);

  function fHandleArgInput(pArgName: string, pValue: string) {
    dPromptArgs[pArgName] = pValue;
    fFetchCompletions(pArgName, pValue);
  }

  function fSelectSuggestion(pArgName: string, pValue: string) {
    dPromptArgs[pArgName] = pValue;
    dSuggestions[pArgName] = [];
    vActiveAutocomplete = null;
  }

  function fHandleArgKeydown(pEvent: KeyboardEvent, pArgName: string) {
    const aArgSuggestions = dSuggestions[pArgName] ?? [];

    // Handle Escape - return to prompt selection list
    if (pEvent.key === KeyboardKey.cEscape) {
      pEvent.preventDefault();
      pEvent.stopPropagation();
      fHandleCancelArgumentForm();

      return;
    }

    if (aArgSuggestions.length === 0 || vActiveAutocomplete !== pArgName) return;

    if (pEvent.key === KeyboardKey.cArrowDown) {
      pEvent.preventDefault();
      vAutocompleteIndex = Math.min(vAutocompleteIndex + 1, aArgSuggestions.length - 1);
    } else if (pEvent.key === KeyboardKey.cArrowUp) {
      pEvent.preventDefault();
      vAutocompleteIndex = Math.max(vAutocompleteIndex - 1, 0);
    } else if (pEvent.key === KeyboardKey.cEnter && aArgSuggestions[vAutocompleteIndex]) {
      pEvent.preventDefault();
      pEvent.stopPropagation();
      fSelectSuggestion(pArgName, aArgSuggestions[vAutocompleteIndex]);
    }
  }

  function fHandleArgBlur(pArgName: string) {
    // Delay to allow click on suggestion
    setTimeout(() => {
      if (vActiveAutocomplete === pArgName) {
        dSuggestions[pArgName] = [];
        vActiveAutocomplete = null;
      }
    }, 150);
  }

  function fHandleArgFocus(pArgName: string) {
    if ((dSuggestions[pArgName]?.length ?? 0) > 0) {
      vActiveAutocomplete = pArgName;
    }
  }

  function fHandleCancelArgumentForm() {
    // Restore the previously selected prompt index
    if (vSelectedIndexBeforeArgumentForm !== null) {
      vSelectedIndex = vSelectedIndexBeforeArgumentForm;
      vSelectedIndexBeforeArgumentForm = null;
    }

    vSelectedPrompt = null;
    dPromptArgs = {};
    vPromptError = null;
  }

  export function handleKeydown(pEvent: KeyboardEvent): boolean {
    if (!isOpen) return false;

    if (pEvent.key === KeyboardKey.cEscape) {
      pEvent.preventDefault();

      if (vSelectedPrompt) {
        // Return to prompt selection list, keeping the selected prompt active
        fHandleCancelArgumentForm();
      } else {
        onClose?.();
      }

      return true;
    }

    if (pEvent.key === KeyboardKey.cArrowDown) {
      pEvent.preventDefault();

      if (aFilteredPrompts.length > 0) {
        vSelectedIndex = (vSelectedIndex + 1) % aFilteredPrompts.length;
        vScrollTrigger++;
      }

      return true;
    }

    if (pEvent.key === KeyboardKey.cArrowUp) {
      pEvent.preventDefault();

      if (aFilteredPrompts.length > 0) {
        vSelectedIndex = vSelectedIndex === 0 ? aFilteredPrompts.length - 1 : vSelectedIndex - 1;
        vScrollTrigger++;
      }

      return true;
    }

    if (pEvent.key === KeyboardKey.cEnter && !vSelectedPrompt) {
      pEvent.preventDefault();

      if (aFilteredPrompts[vSelectedIndex]) {
        fHandlePromptClick(aFilteredPrompts[vSelectedIndex]);
      }

      return true;
    }

    return false;
  }

  let aFilteredPrompts = $derived.by(() => {
    const aSortedServers = cMcpStore.mGetServers();
    const dServerOrderMap = new Map(aSortedServers.map((pServer, pIndex) => [pServer.id, pIndex]));
    const aSortedPrompts = [...aPrompts].sort((pA, pB) => {
      const cOrderA = dServerOrderMap.get(pA.serverName) ?? Number.MAX_SAFE_INTEGER;
      const cOrderB = dServerOrderMap.get(pB.serverName) ?? Number.MAX_SAFE_INTEGER;

      return cOrderA - cOrderB;
    });
    const cQuery = (searchQuery || vInternalSearchQuery).toLowerCase();

    if (!cQuery) return aSortedPrompts;

    return aSortedPrompts.filter(
      (pPrompt) =>
        pPrompt.name.toLowerCase().includes(cQuery) ||
        pPrompt.title?.toLowerCase().includes(cQuery) ||
        pPrompt.description?.toLowerCase().includes(cQuery)
    );
  });

  let vShowSearchInput = $derived(aPrompts.length > 3);
</script>

<ChatFormPickerPopover
  bind:isOpen
  class={className}
  {onClose}
  onKeydown={handleKeydown}
  srLabel="Open prompt picker"
>
  {#if vSelectedPrompt}
    {@const prompt = vSelectedPrompt}
    {@const server = vServerSettingsMap.get(prompt.serverName)}
    {@const cServerLabel = server ? cMcpStore.mGetServerLabel(server) : prompt.serverName}

    <div class="p-4">
      <ChatFormPickerItemHeader
        description={prompt.description}
        {server}
        serverLabel={cServerLabel}
        title={prompt.title || prompt.name}
      >
        {#snippet titleExtra()}
          {#if prompt.arguments?.length}
            <Badge variant="secondary">
              {prompt.arguments.length}
              {fTranslate('text_b25f03dedd69')}{prompt.arguments.length > 1 ? 's' : ''}
            </Badge>
          {/if}
        {/snippet}
      </ChatFormPickerItemHeader>

      <ChatFormPromptPickerArgumentForm
        activeAutocomplete={vActiveAutocomplete}
        autocompleteIndex={vAutocompleteIndex}
        loadingSuggestions={dLoadingSuggestions}
        onArgBlur={fHandleArgBlur}
        onArgFocus={fHandleArgFocus}
        onArgInput={fHandleArgInput}
        onArgKeydown={fHandleArgKeydown}
        onCancel={fHandleCancelArgumentForm}
        onSelectSuggestion={fSelectSuggestion}
        onSubmit={fHandleArgumentSubmit}
        prompt={vSelectedPrompt}
        promptArgs={dPromptArgs}
        promptError={vPromptError}
        suggestions={dSuggestions}
      />
    </div>
  {:else}
    <ChatFormPickerList
      bind:searchQuery={vInternalSearchQuery}
      emptyMessage="No MCP prompts available"
      {isLoading}
      itemKey={(pPrompt) => pPrompt.serverName + ':' + pPrompt.name}
      items={aFilteredPrompts}
      scrollTrigger={vScrollTrigger}
      searchPlaceholder="Search prompts..."
      selectedIndex={vSelectedIndex}
      showSearchInput={vShowSearchInput}
    >
      {#snippet item(pPrompt, pIndex, pIsSelected)}
        {@const server = vServerSettingsMap.get(pPrompt.serverName)}
        {@const cServerLabel = server ? cMcpStore.mGetServerLabel(server) : pPrompt.serverName}

        <ChatFormPickerListItem
          dataIndex={pIndex}
          isSelected={pIsSelected}
          onclick={() => fHandlePromptClick(pPrompt)}
        >
          <ChatFormPickerItemHeader
            description={pPrompt.description}
            {server}
            serverLabel={cServerLabel}
            title={pPrompt.title || pPrompt.name}
          >
            {#snippet titleExtra()}
              {#if pPrompt.arguments?.length}
                <Badge variant="secondary">
                  {pPrompt.arguments.length}
                  {fTranslate('text_b25f03dedd69')}{pPrompt.arguments.length > 1 ? 's' : ''}
                </Badge>
              {/if}
            {/snippet}
          </ChatFormPickerItemHeader>
        </ChatFormPickerListItem>
      {/snippet}

      {#snippet skeleton()}
        <ChatFormPickerListItemSkeleton showBadge titleWidth="w-32" />
      {/snippet}
    </ChatFormPickerList>
  {/if}
</ChatFormPickerPopover>
