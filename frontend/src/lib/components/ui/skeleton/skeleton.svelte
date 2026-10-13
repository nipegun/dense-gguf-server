<script lang="ts">
  import { fCn, type WithElementRef, type WithoutChildren } from '$lib/components/ui/utils.js';
  import type { HTMLAttributes } from 'svelte/elements';

  let {
    class: className,
    ref = $bindable(null),
    ...restProps
  }: WithoutChildren<WithElementRef<HTMLAttributes<HTMLDivElement>>> = $props();
</script>

<div
  bind:this={ref}
  class={fCn('animate-pulse rounded-md bg-accent', className)}
  data-slot="skeleton"
  {...restProps}
></div>
