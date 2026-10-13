<script lang="ts">
  import { fCn, type WithElementRef } from '$lib/components/ui/utils.js';
  import type { HTMLAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    ...restProps
  }: WithElementRef<HTMLAttributes<HTMLDivElement>> = $props();
</script>

<div
  bind:this={ref}
  class={fCn(
    'col-start-2 grid justify-items-start gap-1 text-sm text-muted-foreground [&_p]:leading-relaxed',
    className
  )}
  data-slot="alert-description"
  {...restProps}
>
  {@render children?.()}
</div>
