<script lang="ts">
  import { fCn, type WithElementRef } from '$lib/components/ui/utils.js';
  import type { HTMLAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    ...restProps
  }: WithElementRef<HTMLAttributes<HTMLTableSectionElement>> = $props();
</script>

<tbody
  bind:this={ref}
  class={fCn('[&_tr:last-child]:border-0', className)}
  data-slot="table-body"
  {...restProps}
>
  {@render children?.()}
</tbody>
