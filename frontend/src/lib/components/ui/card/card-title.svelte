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
  class={fCn('leading-none font-semibold', className)}
  data-slot="card-title"
  {...restProps}
>
  {@render children?.()}
</div>
