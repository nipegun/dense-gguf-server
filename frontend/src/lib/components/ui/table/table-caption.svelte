<script lang="ts">
  import { fCn, type WithElementRef } from '$lib/components/ui/utils.js';
  import type { HTMLAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    ...restProps
  }: WithElementRef<HTMLAttributes<HTMLElement>> = $props();
</script>

<caption
  bind:this={ref}
  class={fCn('mt-4 text-sm text-muted-foreground', className)}
  data-slot="table-caption"
  {...restProps}
>
  {@render children?.()}
</caption>
