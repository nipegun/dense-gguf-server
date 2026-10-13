<script lang="ts" module>
  import { fToggleVariants } from '$lib/components/ui/toggle/index.js';
  import { getContext, setContext } from 'svelte';
  import type { VariantProps } from 'tailwind-variants';

  type ToggleVariants = VariantProps<typeof fToggleVariants>;

  interface ToggleGroupContext extends ToggleVariants {
    spacing?: number;
    orientation?: 'horizontal' | 'vertical';
  }

  export function fSetToggleGroupCtx(pProps: ToggleGroupContext) {
    setContext('toggleGroup', pProps);
  }

  export function fGetToggleGroupCtx() {
    return getContext<Required<ToggleGroupContext>>('toggleGroup');
  }
</script>

<script lang="ts">
  import { fCn } from '$lib/components/ui/utils.js';
  import { ToggleGroup as ToggleGroupPrimitive } from 'bits-ui';

  let {
    class: className,
    orientation = 'horizontal',
    ref = $bindable(null),
    size = 'default',
    spacing = 0,
    value = $bindable(),
    variant = 'default',
    ...restProps
  }: ToggleGroupPrimitive.RootProps &
    ToggleVariants & {
      spacing?: number;
      orientation?: 'horizontal' | 'vertical';
    } = $props();

  fSetToggleGroupCtx({
    get orientation() {
      return orientation;
    },
    get size() {
      return size;
    },
    get spacing() {
      return spacing;
    },
    get variant() {
      return variant;
    }
  });
</script>

<!--
Discriminated Unions + Destructing (required for bindable) do not
get along, so we shut typescript up by casting `value` to `never`.
-->
<ToggleGroupPrimitive.Root
  bind:ref
  bind:value={value as never}
  class={fCn(
    'rounded-md group/toggle-group flex w-fit flex-row items-center gap-[--spacing(var(--gap))] data-vertical:flex-col data-vertical:items-stretch',
    className
  )}
  data-size={size}
  data-slot="toggle-group"
  data-spacing={spacing}
  data-variant={variant}
  {orientation}
  style={`--gap: ${spacing}`}
  {...restProps}
/>
