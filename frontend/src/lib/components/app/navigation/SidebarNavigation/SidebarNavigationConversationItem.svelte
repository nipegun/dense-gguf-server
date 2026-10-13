<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import {
    Download,
    GitBranch,
    ListChecks,
    Loader2,
    MoreHorizontal,
    Pencil,
    Pin,
    PinOff,
    Square,
    Trash2
  } from '@lucide/svelte';
  import { DropdownMenuActions } from '$lib/components/app';
  import { TruncatedText } from '$lib/components/app';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cForkTreeDepthPadding, cIconClassDefault, cUiDataAttrs } from '$lib/constants';
  import { RouterService } from '$lib/services/router.service';
  import { cChatStore, cConversationsStore } from '$lib/stores';
  import { onMount } from 'svelte';

  interface Props {
    isActive?: boolean;
    depth?: number;
    conversation: DatabaseConversation;
    isSelectionMode?: boolean;
    isSelected?: boolean;
    onDelete?: (pId: string) => void;
    onEdit?: (pId: string) => void;
    onSelect?: (pId: string) => void;
    onStop?: (pId: string) => void;
    onToggleSelect?: (pId: string) => void;
    onEnterSelectionMode?: (pId: string) => void;
    onSelectionClick?: (pId: string, pOptions: { shiftKey: boolean }) => void;
    onRowMouseDown?: (pId: string, pEvent: MouseEvent) => void;
  }

  let {
    conversation,
    depth = 0,
    isActive = false,
    isSelected = false,
    isSelectionMode = false,
    onDelete,
    onEdit,
    onEnterSelectionMode,
    onRowMouseDown,
    onSelect,
    onSelectionClick,
    onStop,
    onToggleSelect
  }: Props = $props();

  let vRenderActionsDropdown = $state(false);
  let vDropdownOpen = $state(false);

  let isLoading = $derived(cChatStore.mGetAllLoadingChats().includes(conversation.id));

  function fHandleEdit(pEvent: Event) {
    pEvent.stopPropagation();
    onEdit?.(conversation.id);
  }

  function fHandleDelete(pEvent: Event) {
    pEvent.stopPropagation();
    onDelete?.(conversation.id);
  }

  function fHandleStop(pEvent: Event) {
    pEvent.stopPropagation();
    onStop?.(conversation.id);
  }

  function fHandleTogglePin() {
    cConversationsStore.mToggleConversationPin(conversation.id);
  }

  function fHandleEnterSelectionMode(pEvent: Event) {
    pEvent.stopPropagation();
    onEnterSelectionMode?.(conversation.id);
  }

  function fHandleGlobalEditEvent(pEvent: Event) {
    const cCustomEvent = pEvent as CustomEvent<{ conversationId: string }>;

    if (cCustomEvent.detail.conversationId === conversation.id && isActive) {
      fHandleEdit(pEvent);
    }
  }

  function fHandleMouseLeave() {
    if (!vDropdownOpen) {
      vRenderActionsDropdown = false;
    }
  }

  function fHandleMouseOver() {
    if (isSelectionMode) return;

    vRenderActionsDropdown = true;
  }

  function fHandleSelect(pEvent: MouseEvent) {
    if (isSelectionMode) {
      onSelectionClick?.(conversation.id, { shiftKey: pEvent.shiftKey });
    } else {
      onSelect?.(conversation.id);
    }
  }

  function fHandleCheckboxClick(pEvent: MouseEvent) {
    pEvent.stopPropagation();

    if (isSelectionMode) {
      onSelectionClick?.(conversation.id, { shiftKey: pEvent.shiftKey });
    } else {
      onToggleSelect?.(conversation.id);
    }
  }

  function fHandleRowMouseDown(pEvent: MouseEvent) {
    onRowMouseDown?.(conversation.id, pEvent);
  }

  function fHandleCheckboxKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key !== ' ' && pEvent.key !== 'Enter') return;

    pEvent.stopPropagation();
    pEvent.preventDefault();

    if (isSelectionMode) {
      onSelectionClick?.(conversation.id, { shiftKey: pEvent.shiftKey });
    } else {
      onToggleSelect?.(conversation.id);
    }
  }

  $effect(() => {
    if (!vDropdownOpen) {
      vRenderActionsDropdown = false;
    }
  });

  onMount(() => {
    document.addEventListener('edit-active-conversation', fHandleGlobalEditEvent as EventListener);

    return () => {
      document.removeEventListener(
        'edit-active-conversation',
        fHandleGlobalEditEvent as EventListener
      );
    };
  });
</script>

<button
  class="group flex min-h-9 w-full cursor-pointer items-center justify-between space-x-3 rounded-lg py-1.5 text-left transition-colors hover:bg-foreground/10 {isActive
    ? 'bg-foreground/5 text-accent-foreground'
    : ''} {isSelected ? 'bg-primary/10 hover:bg-primary/15' : ''} {isSelectionMode
    ? 'is-selection-mode'
    : ''} px-2"
  {...{ [cUiDataAttrs.CONVERSATION_ROW]: conversation.id }}
  onclick={(pE) => fHandleSelect(pE)}
  onfocusin={fHandleMouseOver}
  onfocusout={(pE) => {
    if (!pE.currentTarget.contains(pE.relatedTarget as Node | null)) {
      fHandleMouseLeave();
    }
  }}
  onmousedown={(pE) => fHandleRowMouseDown(pE)}
  onmouseleave={fHandleMouseLeave}
  onmouseover={fHandleMouseOver}
>
  <div
    style:padding-left="{depth * cForkTreeDepthPadding}{fTranslate('dgs_6ee2cc105a67')}"
    class="flex min-w-0 flex-1 items-center gap-2"
  >
    {#if isSelectionMode}
      <div
        aria-checked={isSelected}
        aria-label={isSelected ? `Deselect ${conversation.name}` : `Select ${conversation.name}`}
        class="shrink-0"
        onclick={(pE) => fHandleCheckboxClick(pE)}
        onkeydown={fHandleCheckboxKeydown}
        role="checkbox"
        tabindex="-1"
      >
        <Checkbox
          aria-label={isSelected ? `Deselect ${conversation.name}` : `Select ${conversation.name}`}
          checked={isSelected}
        />
      </div>
    {/if}

    {#if depth > 0}
      <Tooltip.fRoot>
        <Tooltip.Trigger>
          <!-- prevent another nested button element -->
          {#snippet child({ props: dProps })}
            <a
              {...dProps}
              class="flex shrink-0 items-center text-muted-foreground transition-colors hover:text-foreground"
              href={RouterService.mChat(conversation.forkedFromConversationId)}
            >
              <GitBranch class="h-3.5 w-3.5" />
            </a>
          {/snippet}
        </Tooltip.Trigger>

        <Tooltip.Content>
          <p>{fTranslate('text_4d71ef1fdda5')}</p>
        </Tooltip.Content>
      </Tooltip.fRoot>
    {/if}

    {#if isLoading}
      <Tooltip.fRoot>
        <Tooltip.Trigger>
          <div
            aria-label={fTranslate('text_d4cb298c2663')}
            class="stop-button flex {cIconClassDefault} shrink-0 cursor-pointer items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
            onclick={fHandleStop}
            onkeydown={(pE) => pE.key === 'Enter' && fHandleStop(pE)}
            role="button"
            tabindex="0"
          >
            <Loader2 class="loading-icon h-3.5 w-3.5 animate-spin" />

            <Square class="stop-icon hidden h-3 w-3 fill-current text-destructive" />
          </div>
        </Tooltip.Trigger>

        <Tooltip.Content>
          <p>{fTranslate('text_d4cb298c2663')}</p>
        </Tooltip.Content>
      </Tooltip.fRoot>
    {/if}

    <TruncatedText class="text-sm font-medium" showTooltip={false} text={conversation.name} />
  </div>

  {#if !isSelectionMode && vRenderActionsDropdown}
    <div class="actions flex items-center">
      <DropdownMenuActions
        bind:open={vDropdownOpen}
        actions={[
          {
            icon: conversation.pinned ? PinOff : Pin,
            label: conversation.pinned ? 'Unpin' : 'Pin',
            onclick: (pE: Event) => {
              pE.stopPropagation();
              fHandleTogglePin();
            }
          },
          {
            icon: Pencil,
            label: 'Edit',
            onclick: fHandleEdit,
            shortcut: ['shift', 'cmd', 'e']
          },
          {
            icon: Download,
            label: 'Export',
            onclick: (pE: Event) => {
              pE.stopPropagation();
              cConversationsStore.mDownloadConversation(conversation.id);
            },
            shortcut: ['shift', 'cmd', 's']
          },
          {
            icon: ListChecks,
            label: 'Select',
            onclick: fHandleEnterSelectionMode
          },
          {
            icon: Trash2,
            label: 'Delete',
            onclick: fHandleDelete,
            separator: true,
            shortcut: ['shift', 'cmd', 'd'],
            variant: 'destructive'
          }
        ]}
        triggerIcon={MoreHorizontal}
        triggerTooltip="More actions"
      />
    </div>
  {/if}
</button>

<style>
  button {
    :global([data-slot='dropdown-menu-trigger']:not([data-state='open'])) {
      opacity: 0;
    }

    &:is(:hover) :global([data-slot='dropdown-menu-trigger']),
    &:focus-within :global([data-slot='dropdown-menu-trigger']) {
      opacity: 1;
    }
    @media (max-width: 768px) {
      :global([data-slot='dropdown-menu-trigger']) {
        opacity: 1 !important;
      }
    }

    &.is-selection-mode :global([data-slot='dropdown-menu-trigger']) {
      display: none !important;
    }

    .stop-button {
      :global(.stop-icon) {
        display: none;
      }

      :global(.loading-icon) {
        display: block;
      }
    }

    &:is(:hover) .stop-button {
      :global(.stop-icon) {
        display: block;
      }

      :global(.loading-icon) {
        display: none;
      }
    }
  }
</style>
