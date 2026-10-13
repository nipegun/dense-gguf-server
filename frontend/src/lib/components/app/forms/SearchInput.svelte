<script lang="ts">
  import { Search, X } from '@lucide/svelte';
  import { Input } from '$lib/components/ui/input';
  import { cIconClassDefault } from '$lib/constants';

  interface Props {
    autofocus?: boolean;
    value?: string;
    placeholder?: string;
    onInput?: (pValue: string) => void;
    onClose?: () => void;
    onKeyDown?: (pEvent: KeyboardEvent) => void;
    class?: string;
    /** Input height, matching the small size of the other controls. */
    size?: 'sm' | 'default';
    id?: string;
    ref?: HTMLInputElement | null;
    isCancelAlwaysVisible?: boolean;
  }

  let {
    autofocus,
    class: className,
    id,
    isCancelAlwaysVisible = false,
    onClose,
    onInput,
    onKeyDown,
    placeholder = 'Search...',
    ref = $bindable(null),
    size = 'default',
    value = $bindable('')
  }: Props = $props();

  let vShowClearButton = $derived(isCancelAlwaysVisible || !!value || !!onClose);

  function fHandleInput(pEvent: Event) {
    const cTarget = pEvent.target as HTMLInputElement;

    value = cTarget.value;
    onInput?.(cTarget.value);
  }

  function fHandleClear() {
    if (value) {
      value = '';
      onInput?.('');
      ref?.focus({ preventScroll: true });
    } else {
      onClose?.();
    }
  }
</script>

<div class="relative {className}">
  <Search
    class="absolute top-1/2 left-3 z-10 {cIconClassDefault} -translate-y-1/2 transform text-muted-foreground"
  />

  <Input
    bind:ref
    bind:value
    {autofocus}
    class="max-md:h-10 pl-9 {vShowClearButton ? 'pr-9' : ''} {size === 'sm' ? 'h-8' : ''}"
    {id}
    oninput={fHandleInput}
    onkeydown={onKeyDown}
    {placeholder}
    type="search"
  />

  {#if vShowClearButton}
    <button
      aria-label={value ? 'Clear search' : 'Close'}
      class="absolute top-1/2 right-3 -translate-y-1/2 transform cursor-pointer text-muted-foreground transition-colors hover:text-foreground"
      onclick={fHandleClear}
      type="button"
    >
      <X class={cIconClassDefault} />
    </button>
  {/if}
</div>
