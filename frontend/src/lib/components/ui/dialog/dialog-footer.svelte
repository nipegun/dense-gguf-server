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

<div
  bind:this={ref}
  class={fCn('flex flex-col-reverse gap-2 sm:flex-row sm:justify-end', className)}
  data-slot="dialog-footer"
  {...restProps}
>
  {@render children?.()}
</div>
