<script lang="ts">
  import { fCn } from '$lib/components/ui/utils.js';
  import { Separator as SeparatorPrimitive } from 'bits-ui';

  let {
    class: className,
    ref = $bindable(null),
    ...restProps
  }: SeparatorPrimitive.RootProps = $props();
</script>

<SeparatorPrimitive.Root
  bind:ref
  class={fCn(
    'shrink-0 bg-border data-[orientation=horizontal]:h-px data-[orientation=horizontal]:w-full data-[orientation=vertical]:h-full data-[orientation=vertical]:w-px',
    className
  )}
  data-slot="separator"
  {...restProps}
/>
