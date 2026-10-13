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

<thead
  bind:this={ref}
  class={fCn('[&_tr]:border-b', className)}
  data-slot="table-header"
  {...restProps}
>
  {@render children?.()}
</thead>
