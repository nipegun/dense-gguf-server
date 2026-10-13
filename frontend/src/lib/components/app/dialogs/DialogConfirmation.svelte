<script lang="ts">
  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { KeyboardKey } from '$lib/enums';
  import type { Component, Snippet } from 'svelte';

  interface Props {
    open: boolean;
    title: string;
    description: string;
    confirmText?: string;
    cancelText?: string;
    variant?: 'default' | 'destructive';
    icon?: Component;
    onConfirm: () => void;
    onCancel: () => void;
    onKeydown?: (pEvent: KeyboardEvent) => void;
    children?: Snippet;
  }

  let {
    cancelText = 'Cancel',
    children,
    confirmText = 'Confirm',
    description,
    icon,
    onCancel,
    onConfirm,
    onKeydown,
    open = $bindable(),
    title,
    variant = 'default'
  }: Props = $props();

  function fHandleKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === KeyboardKey.cEnter) {
      pEvent.preventDefault();

      onConfirm();
    }

    onKeydown?.(pEvent);
  }

  function fHandleOpenChange(pNewOpen: boolean) {
    if (!pNewOpen) {
      onCancel();
    }
  }
</script>

<AlertDialog.fRoot onOpenChange={fHandleOpenChange} {open}>
  <AlertDialog.Content onkeydown={fHandleKeydown}>
    <AlertDialog.Header>
      <AlertDialog.Title class="flex items-center gap-2">
        {#if icon}
          {@const IconComponent = icon}

          <IconComponent class="h-5 w-5 {variant === 'destructive' ? 'text-destructive' : ''}" />
        {/if}
        {title}
      </AlertDialog.Title>

      <AlertDialog.Description>
        {description}
      </AlertDialog.Description>
    </AlertDialog.Header>

    {#if children}
      {@render children()}
    {/if}

    <AlertDialog.Footer>
      <AlertDialog.Cancel onclick={onCancel}>{cancelText}</AlertDialog.Cancel>

      <AlertDialog.Action
        class={variant === 'destructive' ? 'bg-destructive text-white hover:bg-destructive/80' : ''}
        onclick={onConfirm}
      >
        {confirmText}
      </AlertDialog.Action>
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.fRoot>
