<script lang="ts">
  import { fCn } from '$lib/components/ui/utils.js';
  import { Drawer as DrawerPrimitive } from 'vaul-svelte';

  let {
    class: className,
    ref = $bindable(null),
    ...restProps
  }: DrawerPrimitive.TitleProps = $props();
</script>

<DrawerPrimitive.Title
  bind:ref
  class={fCn('text-foreground text-base font-medium', className)}
  data-slot="drawer-title"
  {...restProps}
/>
