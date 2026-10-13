<script lang="ts">
  import SelectScrollDownButton from './select-scroll-down-button.svelte';
  import SelectScrollUpButton from './select-scroll-up-button.svelte';
  import { fCn, type WithoutChild } from '$lib/components/ui/utils.js';
  import { Select as SelectPrimitive } from 'bits-ui';
  import { onDestroy, onMount } from 'svelte';

  let {
    children,
    class: className,
    portalProps,
    ref = $bindable(null),
    sideOffset = 4,
    ...restProps
  }: WithoutChild<SelectPrimitive.ContentProps> & {
    portalProps?: SelectPrimitive.PortalProps;
  } = $props();

  let vCleanupInternalListeners: (() => void) | undefined;

  onMount(() => {
    const cListenerOptions: AddEventListenerOptions = { passive: false };
    const fBlockOutsideWheel = (pEvent: WheelEvent) => {
      if (!ref) {
        return;
      }

      const cTarget = pEvent.target as Node | null;

      if (!cTarget || !ref.contains(cTarget)) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
      }
    };
    const fBlockOutsideTouchMove = (pEvent: TouchEvent) => {
      if (!ref) {
        return;
      }

      const cTarget = pEvent.target as Node | null;

      if (!cTarget || !ref.contains(cTarget)) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
      }
    };

    document.addEventListener('wheel', fBlockOutsideWheel, cListenerOptions);
    document.addEventListener('touchmove', fBlockOutsideTouchMove, cListenerOptions);

    return () => {
      document.removeEventListener('wheel', fBlockOutsideWheel, cListenerOptions);
      document.removeEventListener('touchmove', fBlockOutsideTouchMove, cListenerOptions);
    };
  });

  $effect(() => {
    const cElement = ref;

    vCleanupInternalListeners?.();

    if (!cElement) {
      return;
    }

    const fStopWheelPropagation = (pEvent: WheelEvent) => {
      pEvent.stopPropagation();
    };
    const fStopTouchPropagation = (pEvent: TouchEvent) => {
      pEvent.stopPropagation();
    };

    cElement.addEventListener('wheel', fStopWheelPropagation);
    cElement.addEventListener('touchmove', fStopTouchPropagation);

    vCleanupInternalListeners = () => {
      cElement.removeEventListener('wheel', fStopWheelPropagation);
      cElement.removeEventListener('touchmove', fStopTouchPropagation);
    };
  });

  onDestroy(() => {
    vCleanupInternalListeners?.();
  });
</script>

<SelectPrimitive.Portal {...portalProps}>
  <SelectPrimitive.Content
    bind:ref
    class={fCn(
      'relative z-[var(--layer-popover,1000000)] max-h-(--bits-select-content-available-height) min-w-[8rem] origin-(--bits-select-content-transform-origin) overflow-x-hidden overflow-y-auto rounded-md border bg-popover text-popover-foreground shadow-md data-[side=bottom]:translate-y-1 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:-translate-x-1 data-[side=left]:slide-in-from-right-2 data-[side=right]:translate-x-1 data-[side=right]:slide-in-from-left-2 data-[side=top]:-translate-y-1 data-[side=top]:slide-in-from-bottom-2 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:fill-mode-forwards data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95',
      className
    )}
    data-slot="select-content"
    {sideOffset}
    {...restProps}
  >
    <SelectScrollUpButton />

    <SelectPrimitive.Viewport
      class={fCn(
        'h-(--bits-select-anchor-height) w-full min-w-(--bits-select-anchor-width) scroll-my-1 p-1'
      )}
    >
      {@render children?.()}
    </SelectPrimitive.Viewport>

    <SelectScrollDownButton />
  </SelectPrimitive.Content>
</SelectPrimitive.Portal>
