<script lang="ts">
  import { fCn, type WithElementRef } from '$lib/components/ui/utils';
  import type { HTMLAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    ...restProps
  }: WithElementRef<HTMLAttributes<HTMLDivElement>> = $props();
</script>

<div bind:this={ref} class={fCn('px-6', className)} data-slot="card-content" {...restProps}>
  {@render children?.()}
</div>
