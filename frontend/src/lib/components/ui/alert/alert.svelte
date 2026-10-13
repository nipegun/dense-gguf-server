<script lang="ts" module>
  import { tv, type VariantProps } from 'tailwind-variants';

  export const fAlertVariants = tv({
    base: 'relative grid w-full grid-cols-[0_1fr] items-start gap-y-0.5 rounded-lg border px-4 py-3 text-sm has-[>svg]:grid-cols-[calc(var(--spacing)*4)_1fr] has-[>svg]:gap-x-3 [&>svg]:size-4 [&>svg]:translate-y-0.5 [&>svg]:text-current',
    defaultVariants: {
      variant: 'default'
    },
    variants: {
      variant: {
        default: 'bg-card text-card-foreground',
        destructive:
          'text-destructive bg-card *:data-[slot=alert-description]:text-destructive/90 [&>svg]:text-current'
      }
    }
  });

  export type AlertVariant = VariantProps<typeof fAlertVariants>['variant'];
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
  }: WithElementRef<HTMLAttributes<HTMLDivElement>> & {
    variant?: AlertVariant;
  } = $props();
</script>

<div
  bind:this={ref}
  class={fCn(fAlertVariants({ variant }), className)}
  data-slot="alert"
  {...restProps}
  role="alert"
>
  {@render children?.()}
</div>
