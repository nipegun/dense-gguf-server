<script lang="ts">
  import ChatFormPickerCommand from './ChatFormPickerCommand.svelte';
  import ChatFormPickerMcpPrompts from './ChatFormPickerMcpPrompts/ChatFormPickerMcpPrompts.svelte';
  import ChatFormPickerMention from './ChatFormPickerMention.svelte';
  import type {
    ChatFormCommand,
    FileMentionEntry,
    GetPromptResult,
    MCPPromptInfo
  } from '$lib/types';

  interface Props {
    isCommandPickerOpen?: boolean;
    commandQuery?: string;
    commands?: ChatFormCommand[];
    isPromptPickerOpen?: boolean;
    promptSearchQuery?: string;
    isMentionPickerOpen?: boolean;
    mentionQuery?: string;
    mentionAnchor?: HTMLElement | null;
    scopePath?: string | null;
    onCommandPickerClose?: () => void;
    onCommandSelect?: (pCommand: ChatFormCommand) => void;
    onPromptPickerClose?: () => void;
    onMentionPickerClose?: () => void;
    onMentionOpened?: () => void;
    onMentionSelect?: (pEntry: FileMentionEntry) => void;
    onPromptLoadStart?: (
      pPlaceholderId: string,
      pPromptInfo: MCPPromptInfo,
      pArgs?: Record<string, string>
    ) => void;
    onPromptLoadComplete?: (pPlaceholderId: string, pResult: GetPromptResult) => void;
    onPromptLoadError?: (pPlaceholderId: string, pError: string) => void;
  }

  let {
    commandQuery,
    commands = [],
    isCommandPickerOpen,
    isMentionPickerOpen,
    isPromptPickerOpen,
    mentionAnchor,
    mentionQuery,
    onCommandPickerClose,
    onCommandSelect,
    onMentionOpened,
    onMentionPickerClose,
    onMentionSelect,
    onPromptLoadComplete,
    onPromptLoadError,
    onPromptLoadStart,
    onPromptPickerClose,
    promptSearchQuery,
    scopePath
  }: Props = $props();

  let vCommandPickerRef: ChatFormPickerCommand | undefined = $state(undefined);
  let vPromptPickerRef: ChatFormPickerMcpPrompts | undefined = $state(undefined);
  let vMentionPickerRef: ChatFormPickerMention | undefined = $state(undefined);

  /** Delegate keyboard events to the active picker child; true if handled. */
  export function handleKeydown(pEvent: KeyboardEvent): boolean {
    if (isCommandPickerOpen && vCommandPickerRef?.handleKeydown(pEvent)) {
      return true;
    }

    if (isPromptPickerOpen && vPromptPickerRef?.handleKeydown(pEvent)) {
      return true;
    }

    if (isMentionPickerOpen && vMentionPickerRef?.handleKeydown(pEvent)) {
      return true;
    }

    return false;
  }
</script>

<ChatFormPickerCommand
  bind:this={vCommandPickerRef}
  {commands}
  isOpen={isCommandPickerOpen ?? false}
  onClose={onCommandPickerClose ?? (() => {})}
  onSelect={onCommandSelect ?? (() => {})}
  query={commandQuery ?? ''}
/>

<ChatFormPickerMcpPrompts
  bind:this={vPromptPickerRef}
  isOpen={isPromptPickerOpen}
  onClose={onPromptPickerClose}
  {onPromptLoadComplete}
  {onPromptLoadError}
  {onPromptLoadStart}
  searchQuery={promptSearchQuery}
/>

<ChatFormPickerMention
  bind:this={vMentionPickerRef}
  customAnchor={mentionAnchor}
  isOpen={isMentionPickerOpen ?? false}
  onClose={onMentionPickerClose ?? (() => {})}
  onOpened={onMentionOpened}
  onSelect={onMentionSelect ?? (() => {})}
  query={mentionQuery ?? ''}
  scopePath={scopePath ?? null}
/>
