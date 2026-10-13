<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import SidebarNavigationConversationItem from './SidebarNavigationConversationItem.svelte';
  import SidebarNavigationSearchResults from './SidebarNavigationSearchResults.svelte';
  import SidebarNavigationSelectionBar from './SidebarNavigationSelectionBar.svelte';
  import { Pin } from '@lucide/svelte';
  import { fBuildConversationTree } from '$lib/utils';

  interface Props {
    class: string;
    filteredConversations: DatabaseConversation[];
    currentChatId: string | undefined;
    isSearchModeActive: boolean;
    searchQuery: string;
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
    visibleCount: number;
    allVisibleSelected: boolean;
    someVisibleSelected: boolean;
    allSelectedArePinned: boolean;
    pinStateIsMixed: boolean;
    onSelectAllToggle: () => void;
    onBulkPinToggle: () => void;
    onBulkExport: () => void;
    onBulkDelete: () => void;
    onCloseSelection: () => void;
  }

  let {
    allSelectedArePinned,
    allVisibleSelected,
    class: className,
    currentChatId,
    filteredConversations,
    isSearchModeActive,
    isSelectionMode = false,
    onBulkDelete,
    onBulkExport,
    onBulkPinToggle,
    onCloseSelection,
    onDelete,
    onEdit,
    onEnterSelectionMode,
    onRowMouseDown,
    onSelect,
    onSelectAllToggle,
    onSelectionClick,
    onStop,
    onToggleSelect,
    pinStateIsMixed,
    searchQuery,
    selectedIds = new Set<string>(),
    someVisibleSelected,
    visibleCount
  }: Props = $props();

  let aConversationTree = $derived(fBuildConversationTree(filteredConversations));

  let aPinnedConversations = $derived(
    aConversationTree.filter(({ conversation: vConversation }) => vConversation.pinned)
  );

  let aUnpinnedConversations = $derived(
    aConversationTree.filter(({ conversation: vConversation }) => !vConversation.pinned)
  );

  const cRecentEmptyMessage = $derived(
    searchQuery.length > 0 ? 'No results found' : 'No conversations yet'
  );
</script>

<div class="flex min-h-0 flex-1 flex-col">
  {#if isSearchModeActive}
    <SidebarNavigationSearchResults
      class={className}
      {currentChatId}
      {filteredConversations}
      {isSelectionMode}
      {onDelete}
      {onEdit}
      {onEnterSelectionMode}
      {onRowMouseDown}
      {onSelect}
      {onSelectionClick}
      {onStop}
      {onToggleSelect}
      {searchQuery}
      {selectedIds}
    />
  {:else}
    {#if aPinnedConversations.length > 0}
      <div class="py-2 flex whitespace-nowrap {className}">
        <div
          class="text-muted-foreground inline-flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium gap-1"
        >
          <Pin class="h-3.5 w-3.5" />

          <span>{fTranslate('text_f20c87946555')}</span>
        </div>
      </div>

      <ul class="flex w-full min-w-0 flex-col gap-4 md:gap-1 {className}">
        {#each aPinnedConversations as { conversation: vConversation, depth: vDepth } (vConversation.id)}
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
      </ul>
    {/if}

    <div class="mt-2 flex min-h-0 flex-1 flex-col gap-4 md:gap-0 whitespace-nowrap {className}">
      {#if filteredConversations.length > 0}
        <div
          class="text-muted-foreground flex h-8 shrink-0 items-center rounded-md px-2 text-xs font-medium"
        >
          {fTranslate('text_529ef4c48988')}
        </div>
      {/if}

      <div class="min-h-0 flex-1 md:overflow-y-auto">
        <ul class="flex w-full min-w-0 flex-col gap-4 md:gap-0">
          {#each aUnpinnedConversations as { conversation: vConversation, depth: vDepth } (vConversation.id)}
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

          {#if aUnpinnedConversations.length === 0}
            <li class="px-2 py-4 text-center">
              <p class="mb-4 p-4 text-sm text-muted-foreground">
                {cRecentEmptyMessage}
              </p>
            </li>
          {/if}
        </ul>
      </div>
    </div>

    {#if isSelectionMode}
      <SidebarNavigationSelectionBar
        {allVisibleSelected}
        class="sticky top-0 z-10 m-2 mt-0"
        {onBulkDelete}
        {onBulkExport}
        {onBulkPinToggle}
        onClose={onCloseSelection}
        {onSelectAllToggle}
        {pinStateIsMixed}
        selectedCount={selectedIds.size}
        someSelectedPinned={allSelectedArePinned}
        {someVisibleSelected}
        {visibleCount}
      />
    {/if}
  {/if}
</div>
