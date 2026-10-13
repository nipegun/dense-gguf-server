<script lang="ts">
  import { fGetToggleGroupCtx } from './toggle-group.svelte';
  import { type ToggleVariants, fToggleVariants } from '$lib/components/ui/toggle/index.js';
  import { fCn } from '$lib/components/ui/utils.js';
  import { ToggleGroup as ToggleGroupPrimitive } from 'bits-ui';

  let {
    class: className,
    ref = $bindable(null),
    size,
    value = $bindable(),
    variant,
    ...restProps
  }: ToggleGroupPrimitive.ItemProps & ToggleVariants = $props();

  const cCtx = fGetToggleGroupCtx();
</script>

<ToggleGroupPrimitive.Item
  bind:ref
  class={fCn(
    'group-data-[spacing=0]/toggle-group:rounded-none group-data-[spacing=0]/toggle-group:px-2 group-data-[spacing=0]/toggle-group:has-data-[icon=inline-end]:pr-1.5 group-data-[spacing=0]/toggle-group:has-data-[icon=inline-start]:pl-1.5 group-data-horizontal/toggle-group:data-[spacing=0]:first:rounded-l-md group-data-vertical/toggle-group:data-[spacing=0]:first:rounded-t-md group-data-horizontal/toggle-group:data-[spacing=0]:last:rounded-r-md group-data-vertical/toggle-group:data-[spacing=0]:last:rounded-b-md shrink-0 focus:z-10 focus-visible:z-10 group-data-horizontal/toggle-group:data-[spacing=0]:data-[variant=outline]:border-l-0 group-data-vertical/toggle-group:data-[spacing=0]:data-[variant=outline]:border-t-0 group-data-horizontal/toggle-group:data-[spacing=0]:data-[variant=outline]:first:border-l group-data-vertical/toggle-group:data-[spacing=0]:data-[variant=outline]:first:border-t',
    fToggleVariants({
      size: cCtx.size || size,
      variant: cCtx.variant || variant
    }),
    className
  )}
  data-size={cCtx.size || size}
  data-slot="toggle-group-item"
  data-spacing={cCtx.spacing}
  data-variant={cCtx.variant || variant}
  {value}
  {...restProps}
/>
