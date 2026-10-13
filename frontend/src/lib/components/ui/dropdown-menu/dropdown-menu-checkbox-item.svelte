<script lang="ts">
  import CheckIcon from '@lucide/svelte/icons/check';
  import MinusIcon from '@lucide/svelte/icons/minus';
  import { fCn, type WithoutChildrenOrChild } from '$lib/components/ui/utils.js';
  import { DropdownMenu as DropdownMenuPrimitive } from 'bits-ui';
  import type { Snippet } from 'svelte';

  let {
    checked = $bindable(false),
    children: childrenProp,
    class: className,
    indeterminate = $bindable(false),
    ref = $bindable(null),
    ...restProps
  }: WithoutChildrenOrChild<DropdownMenuPrimitive.CheckboxItemProps> & {
    children?: Snippet;
  } = $props();
</script>

<DropdownMenuPrimitive.CheckboxItem
  bind:checked
  bind:indeterminate
  bind:ref
  class={fCn(
    "relative flex cursor-default items-center gap-2 rounded-sm py-1.5 pr-2 pl-8 text-sm outline-hidden select-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50 [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
    className
  )}
  data-slot="dropdown-menu-checkbox-item"
  {...restProps}
>
  {#snippet children({ checked: vChecked, indeterminate: vIndeterminate })}
    <span class="pointer-events-none absolute left-2 flex size-3.5 items-center justify-center">
      {#if vIndeterminate}
        <MinusIcon class="size-4" />
      {:else}
        <CheckIcon class={fCn('size-4', !vChecked && 'text-transparent')} />
      {/if}
    </span>

    {@render childrenProp?.()}
  {/snippet}
</DropdownMenuPrimitive.CheckboxItem>
