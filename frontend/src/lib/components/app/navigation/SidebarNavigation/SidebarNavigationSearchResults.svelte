<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';
  import { fBuildConversationTree } from '$lib/utils';

  interface Props {
    class?: string;
    searchQuery: string;
    filteredConversations: DatabaseConversation[];
    currentChatId: string | undefined;
    isSelectionMode?: boolean;
    selectedIds?: Set<string>;
    onSelect: (pId: string) => void;
    onEdit: (pId: string) => void;
    onDelete: (pId: string) => void;
    onStop: (pId: string) => void;
    onToggleSelect?: (pId: string) => void;
    onEnterSelectionMode?: (pId: string) => void;
    onSelectionClick?: (pId: string, pOptions: { shiftKey: boolean }) => void;
    onRowMouseDown?: (pId: string, pEvent: MouseEvent) => void;
  }

  let {
    class: className = '',
    currentChatId,
    filteredConversations,
    isSelectionMode = false,
    onDelete,
    onEdit,
    onEnterSelectionMode,
    onRowMouseDown,
    onSelect,
    onSelectionClick,
    onStop,
    onToggleSelect,
    searchQuery,
    selectedIds = new Set<string>()
  }: Props = $props();

  let aTree = $derived(fBuildConversationTree(filteredConversations));

  const cHasQuery = $derived(searchQuery.trim().length > 0);
  const cShowHeader = $derived(cHasQuery && filteredConversations.length > 0);

  const cEmptyMessage = $derived(cHasQuery ? 'No results found' : 'Start typing to see results');
</script>

<div class="flex min-h-0 flex-1 flex-col gap-2 whitespace-nowrap {className}">
  {#if cShowHeader}
    <div
      class="text-muted-foreground flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium"
    >
      {fTranslate('text_e978b00de465')}
    </div>
  {/if}

  <div class="min-h-0 flex-1 overflow-y-auto">
    <ul class="flex w-full min-w-0 flex-col gap-1">
      {#each aTree as { conversation: vConversation, depth: vDepth } (vConversation.id)}
        <li class="group/item relative mb-1 p-0">
          <SidebarNavigationConversationItem
            conversation={{
              currNode: vConversation.currNode,
              forkedFromConversationId: vConversation.forkedFromConversationId,
              id: vConversation.id,
              lastModified: vConversation.lastModified,
              name: vConversation.name,
              pinned: vConversation.pinned
            }}
            depth={vDepth}
            isActive={currentChatId === vConversation.id}
            isSelected={selectedIds.has(vConversation.id)}
            {isSelectionMode}
            {onDelete}
            {onEdit}
            {onEnterSelectionMode}
            {onRowMouseDown}
            {onSelect}
            {onSelectionClick}
            {onStop}
            {onToggleSelect}
          />
        </li>
      {/each}

      {#if aTree.length === 0}
        <li class="px-2 py-4 text-center">
          <p class="mb-4 p-4 text-sm text-muted-foreground">
            {cEmptyMessage}
          </p>
        </li>
      {/if}
    </ul>
  </div>
</div>
