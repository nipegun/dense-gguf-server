<script lang="ts">
  import ChatFormInputBasic from './ChatFormInputBasic.svelte';
  import ChatFormInputRich from './ChatFormInputRich.svelte';

  interface Props {
    class?: string;
    disabled?: boolean;
    onInput?: () => void;
    onKeydown?: (pEvent: KeyboardEvent) => void;
    onPaste?: (pEvent: ClipboardEvent) => void;
    placeholder?: string;
    value?: string;
    useRichInput?: boolean;
  }

  let {
    class: className = '',
    disabled = false,
    onInput,
    onKeydown,
    onPaste,
    placeholder = 'Ask anything...',
    useRichInput = false,
    value = $bindable('')
  }: Props = $props();

  let vBasicRef: ChatFormInputBasic | undefined = $state();
  let vRichRef: ChatFormInputRich | undefined = $state();

  // The two renderers share one imperative handle (focus/caret/height), so
  // the parent can drive whichever variant is mounted through this one.
  export function getElement() {
    return useRichInput ? vRichRef?.getElement() : vBasicRef?.getElement();
  }

  export function focus() {
    if (useRichInput) vRichRef?.focus();
    else vBasicRef?.focus();
  }

  export function resetHeight() {
    if (useRichInput) vRichRef?.resetHeight();
    else vBasicRef?.resetHeight();
  }

  export function getCaretOffset(): number {
    return useRichInput ? (vRichRef?.getCaretOffset() ?? 0) : (vBasicRef?.getCaretOffset() ?? 0);
  }

  export function setCaretOffset(pOffset: number) {
    if (useRichInput) vRichRef?.setCaretOffset(pOffset);
    else vBasicRef?.setCaretOffset(pOffset);
  }
</script>

{#if useRichInput}
  <ChatFormInputRich
    bind:this={vRichRef}
    bind:value
    class={className}
    {disabled}
    {onInput}
    {onKeydown}
    {onPaste}
    {placeholder}
  />
{:else}
  <ChatFormInputBasic
    bind:this={vBasicRef}
    bind:value
    class={className}
    {disabled}
    {onInput}
    {onKeydown}
    {onPaste}
    {placeholder}
  />
{/if}
