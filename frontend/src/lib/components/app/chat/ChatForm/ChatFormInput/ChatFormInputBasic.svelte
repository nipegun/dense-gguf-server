<script lang="ts">
  import { cDeviceStore } from '$lib/stores';
  import { fAutoResizeTextarea } from '$lib/utils';
  import { onMount } from 'svelte';

  interface Props {
    class?: string;
    disabled?: boolean;
    onInput?: () => void;
    onKeydown?: (pEvent: KeyboardEvent) => void;
    onPaste?: (pEvent: ClipboardEvent) => void;
    placeholder?: string;
    value?: string;
  }

  let {
    class: className = '',
    disabled = false,
    onInput,
    onKeydown,
    onPaste,
    placeholder = 'Ask anything...',
    value = $bindable('')
  }: Props = $props();

  let vTextareaElement: HTMLTextAreaElement | undefined;

  onMount(() => {
    if (vTextareaElement) {
      fAutoResizeTextarea(vTextareaElement);
      vTextareaElement.focus({ preventScroll: true });
    }
  });

  export function getElement() {
    return vTextareaElement;
  }

  export function focus() {
    if (cDeviceStore.isMobile) return;

    vTextareaElement?.focus({ preventScroll: true });
  }

  export function resetHeight() {
    if (vTextareaElement) {
      vTextareaElement.style.height = '1rem';
    }
  }

  // Plain-text caret offsets, shared with the rich chat form input variant so
  // the picker/paste flows can address either renderer through one handle.
  export function getCaretOffset(): number {
    if (!vTextareaElement) return 0;

    return vTextareaElement.selectionStart ?? vTextareaElement.value.length;
  }

  export function setCaretOffset(pOffset: number) {
    vTextareaElement?.setSelectionRange(pOffset, pOffset);
  }
</script>

<div class="flex-1 {className}">
  <textarea
    bind:this={vTextareaElement}
    bind:value
    class={[
      'text-md min-h-12 w-full resize-none border-0 bg-transparent p-0 leading-6 outline-none placeholder:text-muted-foreground focus-visible:ring-0 focus-visible:ring-offset-0',
      disabled && 'cursor-not-allowed'
    ]}
    {disabled}
    oninput={(pEvent) => {
      fAutoResizeTextarea(pEvent.currentTarget);
      onInput?.();
    }}
    onkeydown={onKeydown}
    onpaste={onPaste}
    {placeholder}
    style="max-height: var(--max-message-height);"
  ></textarea>
</div>
