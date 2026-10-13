<script lang="ts">
  import { fCn, type WithElementRef } from '$lib/components/ui/utils.js';
  import type { HTMLAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    ...restProps
  }: WithElementRef<HTMLAttributes<HTMLDivElement>> & {} = $props();
</script>

<div
  bind:this={ref}
  class={fCn('px-2 py-1.5 text-xs text-muted-foreground', className)}
  data-slot="select-label"
  {...restProps}
>
  {@render children?.()}
</div>
