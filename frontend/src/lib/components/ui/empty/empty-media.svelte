<script lang="ts" module>
  import { tv, type VariantProps } from 'tailwind-variants';

  export const fEmptyMediaVariants = tv({
    base: 'mb-2 flex shrink-0 items-center justify-center [&_svg]:pointer-events-none [&_svg]:shrink-0',
    defaultVariants: {
      variant: 'default'
    },
    variants: {
      variant: {
        default: 'bg-transparent',
        icon: "bg-muted text-foreground flex size-8 shrink-0 items-center justify-center rounded-lg [&_svg:not([class*='size-'])]:size-4"
      }
    }
  });

  export type EmptyMediaVariant = VariantProps<typeof fEmptyMediaVariants>['variant'];
</script>

<script lang="ts">
  import { fCn, type WithElementRef } from '$lib/components/ui/utils.js';
  import type { HTMLAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    variant = 'default',
    ...restProps
  }: WithElementRef<HTMLAttributes<HTMLDivElement>> & { variant?: EmptyMediaVariant } = $props();
</script>

<div
  bind:this={ref}
  class={fCn(fEmptyMediaVariants({ variant }), className)}
  data-slot="empty-icon"
  data-variant={variant}
  {...restProps}
>
  {@render children?.()}
</div>
