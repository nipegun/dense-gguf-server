<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import XIcon from '@lucide/svelte/icons/x';
  import { fCn, type WithElementRef } from '$lib/components/ui/utils';
  import { cPanelCloseMobileClass } from '$lib/constants';
  import { Dialog as DialogPrimitive } from 'bits-ui';
  import type { HTMLAttributes } from 'svelte/elements';

  let {
    children,
    class: className,
    ref = $bindable(null),
    showCloseButton = true,
    ...restProps
  }: WithElementRef<HTMLAttributes<HTMLDivElement>> & {
    showCloseButton?: boolean;
  } = $props();
</script>

<!--
	Header is `sticky`, so it stays at the top while the dialog body scrolls. The close
	button lives here (not in the body) so it sticks together with the title. `sticky`
	makes it the containing block, so the close can be absolutely placed at its corner.
-->
<div
  bind:this={ref}
  class={fCn(
    'flex flex-col gap-2 text-center sm:text-left sticky top-0 z-50 bg-background md:bg-transparent',
    className
  )}
  data-slot="dialog-header"
  {...restProps}
>
  {@render children?.()}

  {#if showCloseButton}
    <DialogPrimitive.Close
      class="absolute top-0 right-0 rounded-xs opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-hidden disabled:pointer-events-none max-md:-top-3 max-md:-right-3 {cPanelCloseMobileClass} [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4"
    >
      <XIcon />

      <span class="sr-only">{fTranslate('text_7d9eb7acb13e')}</span>
    </DialogPrimitive.Close>
  {/if}
</div>
