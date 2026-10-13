<script lang="ts">
  import { fCn, type WithElementRef } from '$lib/components/ui/utils.js';
  import type { HTMLThAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    ...restProps
  }: WithElementRef<HTMLThAttributes> = $props();
</script>

<th
  bind:this={ref}
  class={fCn(
    'h-10 bg-clip-padding px-2 text-left align-middle font-medium whitespace-nowrap text-foreground [&:has([role=checkbox])]:pe-0',
    className
  )}
  data-slot="table-head"
  {...restProps}
>
  {@render children?.()}
</th>
