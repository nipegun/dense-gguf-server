<script lang="ts">
  import { fCn, type WithElementRef } from '$lib/components/ui/utils.js';
  import type { HTMLTdAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    ...restProps
  }: WithElementRef<HTMLTdAttributes> = $props();
</script>

<td
  bind:this={ref}
  class={fCn(
    'bg-clip-padding p-2 align-middle whitespace-nowrap [&:has([role=checkbox])]:pe-0',
    className
  )}
  data-slot="table-cell"
  {...restProps}
>
  {@render children?.()}
</td>
