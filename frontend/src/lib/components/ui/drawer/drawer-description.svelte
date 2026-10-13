<script lang="ts">
  import { fCn } from '$lib/components/ui/utils.js';
  import { Drawer as DrawerPrimitive } from 'vaul-svelte';

  let {
    class: className,
    ref = $bindable(null),
    ...restProps
  }: DrawerPrimitive.DescriptionProps = $props();
</script>

<DrawerPrimitive.Description
  bind:ref
  class={fCn('text-muted-foreground text-sm', className)}
  data-slot="drawer-description"
  {...restProps}
/>
