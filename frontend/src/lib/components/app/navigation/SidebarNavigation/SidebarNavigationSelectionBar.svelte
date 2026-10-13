<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Download, Pin, PinOff, Trash2, X } from '@lucide/svelte';
  import { ActionIcon, DialogConfirmation } from '$lib/components/app';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { TooltipSide } from '$lib/enums';

  interface Props {
    class?: string;
    selectedCount: number;
    visibleCount: number;
    allVisibleSelected: boolean;
    someVisibleSelected: boolean;
    someSelectedPinned: boolean;
    pinStateIsMixed: boolean;
    onSelectAllToggle: () => void;
    onBulkPinToggle: () => void;
    onBulkExport: () => void;
    onBulkDelete: () => void;
    onClose: () => void;
  }

  let {
    allVisibleSelected,
    class: className = '',
    onBulkDelete,
    onBulkExport,
    onBulkPinToggle,
    onClose,
    onSelectAllToggle,
    pinStateIsMixed,
    selectedCount,
    someSelectedPinned,
    someVisibleSelected,
    visibleCount
  }: Props = $props();

  let vShowDeleteDialog = $state(false);

  function fHandleDeleteClick() {
    vShowDeleteDialog = true;
  }

  function fHandleDeleteConfirm() {
    vShowDeleteDialog = false;
    onBulkDelete();
  }

  function fHandleDeleteCancel() {
    vShowDeleteDialog = false;
  }

  const cHasSelection = $derived(selectedCount > 0);
  const cIsMasterChecked = $derived(allVisibleSelected);
  const cIsMasterIndeterminate = $derived(!allVisibleSelected && someVisibleSelected);

  const cPinTooltip = $derived(
    cHasSelection
      ? pinStateIsMixed
        ? 'Unavailable for mixed state selection'
        : someSelectedPinned
          ? selectedCount === 1
            ? 'Unpin'
            : 'Unpin all'
          : selectedCount === 1
            ? 'Pin'
            : 'Pin all'
      : 'Pin'
  );

  const cPinDisabled = $derived(!cHasSelection || pinStateIsMixed);
</script>

<div
  aria-label={fTranslate('text_5667f96ffb3d')}
  class="flex items-center gap-1.5 rounded-xl border border-border/50 bg-background/50 px-2 py-1.5 shadow-sm backdrop-blur-xl {className}"
  role="toolbar"
>
  <label class="flex min-w-0 cursor-pointer items-center gap-2">
    <Checkbox
      aria-label={cIsMasterChecked ? 'Deselect all' : 'Select all'}
      checked={cIsMasterChecked}
      indeterminate={cIsMasterIndeterminate}
      onCheckedChange={onSelectAllToggle}
    />

    <span class="truncate text-xs font-medium text-muted-foreground">
      {selectedCount} / {visibleCount}
      {fTranslate('text_d7cbbb688b2e')}
    </span>
  </label>

  <div class="ml-auto flex items-center gap-0.75">
    <ActionIcon
      ariaLabel={cPinTooltip}
      class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent! {cPinDisabled
        ? 'cursor-not-allowed'
        : ''} {!cPinDisabled ? 'opacity-100' : 'opacity-40'}"
      disabled={cPinDisabled}
      icon={someSelectedPinned ? PinOff : Pin}
      iconSize="h-3.5 w-3.5"
      onclick={onBulkPinToggle}
      size="sm"
      tooltip={cPinTooltip}
      tooltipSide={TooltipSide.TOP}
    />

    <ActionIcon
      ariaLabel="Export selected"
      class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent! {cHasSelection
        ? 'opacity-100'
        : 'opacity-40'}"
      disabled={!cHasSelection}
      icon={Download}
      iconSize="h-3.5 w-3.5"
      onclick={onBulkExport}
      size="sm"
      tooltip={cHasSelection ? 'Export' : 'Export'}
      tooltipSide={TooltipSide.TOP}
    />

    <ActionIcon
      ariaLabel="Delete selected"
      class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-destructive/10! dark:hover:bg-destructive/20! disabled:hover:bg-transparent {cHasSelection
        ? 'opacity-100'
        : 'opacity-40'}"
      disabled={!cHasSelection}
      icon={Trash2}
      iconSize="h-3.5 w-3.5 text-destructive"
      onclick={fHandleDeleteClick}
      size="sm"
      tooltip={fTranslate('text_d2ab5d46fed4')}
      tooltipSide={TooltipSide.TOP}
    />

    <div aria-hidden="true" class="mx-1 h-4 w-px bg-border"></div>

    <ActionIcon
      ariaLabel="Exit bulk selection mode"
      class="h-7 w-7 rounded-md bg-transparent backdrop-blur-none hover:bg-accent!"
      icon={X}
      iconSize="h-3.5 w-3.5"
      onclick={onClose}
      size="sm"
      tooltip={fTranslate('text_0a6603b65249')}
      tooltipSide={TooltipSide.TOP}
    />
  </div>
</div>

<DialogConfirmation
  bind:open={vShowDeleteDialog}
  cancelText="Cancel"
  confirmText={selectedCount === 1 ? 'Delete' : `Delete ${selectedCount}`}
  description="This action cannot be undone. The selected conversation{selectedCount === 1
    ? ''
    : 's'} and {selectedCount === 1
    ? 'its'
    : 'their'} messages will be permanently removed, including any forks."
  icon={Trash2}
  onCancel={fHandleDeleteCancel}
  onConfirm={fHandleDeleteConfirm}
  title="Delete {selectedCount} conversation{selectedCount === 1 ? '' : 's'}"
  variant="destructive"
/>
