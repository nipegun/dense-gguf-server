<script lang="ts">
  import {
    ChatFormPickerList,
    ChatFormPickerListItem,
    ChatFormPickerPopover
  } from '$lib/components/app/chat';
  import { dChatFormCommandIcons } from '$lib/constants';
  import { fUsePickerNavigation } from '$lib/hooks/use-picker-navigation.svelte';
  import type { ChatFormCommand } from '$lib/types';

  /**
   * Slash-command picker; `query` (typed after `/`) filters the commands.
   * The parent owns the "dismissed token, don't act until it changes"
   * snapshot, so this picker just renders and reports selection.
   */
  interface Props {
    class?: string;
    isOpen: boolean;
    query: string;
    commands: ChatFormCommand[];
    onClose: () => void;
    onSelect: (pCommand: ChatFormCommand) => void;
  }

  let { class: className = '', commands, isOpen, onClose, onSelect, query }: Props = $props();

  const cTrimmedQuery = $derived((query ?? '').trim().toLowerCase());

  const aFilteredCommands = $derived(
    cTrimmedQuery
      ? commands.filter(
          (pC) =>
            pC.name.toLowerCase().includes(cTrimmedQuery) ||
            pC.description.toLowerCase().includes(cTrimmedQuery) ||
            (pC.keywords ?? []).some((pK) => pK.toLowerCase().includes(cTrimmedQuery))
        )
      : commands
  );

  function fFirstEnabledIndex(): number {
    return aFilteredCommands.findIndex((pC) => !pC.disabled);
  }

  function fStepEnabled(pFrom: number, pDir: number): number {
    const cN = aFilteredCommands.length;

    if (cN === 0) return -1;

    for (let vI = 1; vI <= cN; vI++) {
      const cIdx = (pFrom + pDir * vI + cN) % cN;

      if (!aFilteredCommands[cIdx].disabled) return cIdx;
    }

    return -1;
  }

  const cNav = fUsePickerNavigation({
    count: () => aFilteredCommands.length,
    isOpen: () => isOpen,
    onClose: () => onClose(),
    onSelect: (pIndex) => fHandleSelect(aFilteredCommands[pIndex]),
    step: (pFrom, pDir) => (pFrom < 0 ? fFirstEnabledIndex() : fStepEnabled(pFrom, pDir))
  });

  $effect(() => {
    if (isOpen) {
      cNav.reset(fFirstEnabledIndex());
    }
  });

  $effect(() => {
    if (cNav.hoveredIndex < 0 || cNav.hoveredIndex >= aFilteredCommands.length) {
      cNav.reset(fFirstEnabledIndex());

      return;
    }

    if (aFilteredCommands[cNav.hoveredIndex].disabled) {
      cNav.reset(fFirstEnabledIndex());
    }
  });

  function fHandleSelect(pCommand: ChatFormCommand) {
    if (pCommand.disabled) return;

    onSelect(pCommand);
    onClose();
  }

  export function handleKeydown(pEvent: KeyboardEvent): boolean {
    return cNav.handleKeydown(pEvent);
  }
</script>

<ChatFormPickerPopover
  bind:isOpen
  class={className}
  {onClose}
  onKeydown={handleKeydown}
  srLabel="Open command picker"
>
  <ChatFormPickerList
    emptyMessage="No matching command"
    isLoading={false}
    itemKey={(pCommand) => pCommand.name}
    items={aFilteredCommands}
    scrollTrigger={cNav.scrollTrigger}
    searchQuery={query ?? ''}
    selectedIndex={cNav.hoveredIndex}
    showSearchInput={false}
  >
    {#snippet item(pCommand, pIndex, pIsSelected)}
      {@const Icon = dChatFormCommandIcons[pCommand.action]}
      <ChatFormPickerListItem
        dataIndex={pIndex}
        disabled={pCommand.disabled}
        isSelected={pIsSelected}
        onclick={() => fHandleSelect(pCommand)}
        onmouseenter={() => {
          if (!pCommand.disabled) cNav.setHover(pIndex);
        }}
      >
        <Icon class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />

        <div class="flex min-w-0 flex-1 flex-col">
          <span class="font-mono text-sm font-medium">/{pCommand.name}</span>

          <span class="min-w-0 flex-1 truncate text-left text-xs text-muted-foreground">
            {pCommand.description}
          </span>
        </div>
      </ChatFormPickerListItem>
    {/snippet}
  </ChatFormPickerList>
</ChatFormPickerPopover>
