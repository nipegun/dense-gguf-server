<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ChevronLeft, ChevronRight } from '@lucide/svelte';
  import { fCn } from '$lib/components/ui/utils';
  import { cIconClassDefault } from '$lib/constants';
  import { ScrollCarouselVariant } from '$lib/enums';
  import { fUseScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
  import type { Snippet } from 'svelte';

  interface Props {
    children: Snippet;
    /** External carousel hook for callers that need to drive it (e.g. scrollToCenter). */
    carousel?: ReturnType<typeof fUseScrollCarousel>;
    /** Classes for the outer relative wrapper. */
    class?: string;
    /** Classes for the scrollable overflow container. */
    containerClass?: string;
    /** Classes for the min-w-max content wrapper. */
    innerClass?: string;
    /** Tailwind gap class applied to the content wrapper. */
    gapSize?: string;
    /** Show the arrows whenever the content overflows, even without hover. */
    alwaysShowArrows?: boolean;
    /** Arrow placement and styling. */
    variant?: ScrollCarouselVariant;
  }

  let {
    alwaysShowArrows = false,
    carousel: externalCarousel,
    children,
    class: className = '',
    containerClass = '',
    gapSize = '3',
    innerClass = '',
    variant = ScrollCarouselVariant.TOP
  }: Props = $props();

  const cInternalCarousel = fUseScrollCarousel();
  const cCarousel = $derived(externalCarousel ?? cInternalCarousel);

  const cIsCenter = $derived(variant === ScrollCarouselVariant.CENTER);

  function fScrollLeft(pEvent?: MouseEvent) {
    pEvent?.stopPropagation();
    pEvent?.preventDefault();

    const cContainer = cCarousel.scrollContainer;

    if (!cContainer) return;

    cContainer.scrollBy({ behavior: 'smooth', left: -(cContainer.clientWidth * 0.67) });
  }

  function fScrollRight(pEvent?: MouseEvent) {
    pEvent?.stopPropagation();
    pEvent?.preventDefault();

    const cContainer = cCarousel.scrollContainer;

    if (!cContainer) return;

    cContainer.scrollBy({ behavior: 'smooth', left: cContainer.clientWidth * 0.67 });
  }

  export function resetScroll() {
    const cContainer = cCarousel.scrollContainer;

    if (!cContainer) return;

    cContainer.scrollLeft = 0;
    setTimeout(() => cCarousel.updateScrollButtons(), 0);
  }
</script>

<div
  class={fCn('group relative', !cIsCenter && 'flex items-center', className)}
  style={!cIsCenter ? 'scroll-padding: 1rem;' : undefined}
>
  <button
    class={fCn(
      'absolute z-10 flex h-6 w-6 items-center justify-center rounded-full shadow-md transition-opacity',
      cIsCenter
        ? 'top-1/2 left-4 -translate-y-1/2 bg-background/25 backdrop-blur-xs hover:bg-background/45 disabled:pointer-events-none disabled:opacity-0'
        : 'left-2 bg-muted backdrop-blur-sm hover:bg-accent',
      !cIsCenter &&
        (cCarousel.canScrollLeft
          ? alwaysShowArrows
            ? 'opacity-100'
            : 'opacity-0 group-hover:opacity-100'
          : 'pointer-events-none opacity-0')
    )}
    {...cIsCenter ? { disabled: !cCarousel.canScrollLeft } : {}}
    aria-label={fTranslate('text_35dfa1045cb2')}
    onclick={fScrollLeft}
  >
    <ChevronLeft class={cIconClassDefault} />
  </button>

  <div
    bind:this={cCarousel.scrollContainer}
    class={fCn('scrollbar-hide overflow-x-auto', containerClass)}
    onscroll={cCarousel.updateScrollButtons}
  >
    <div
      bind:this={cCarousel.contentContainer}
      class={fCn('flex min-w-max', cIsCenter && 'items-start', `gap-${gapSize}`, innerClass)}
    >
      {@render children?.()}
    </div>
  </div>

  <button
    class={fCn(
      'absolute z-10 flex h-6 w-6 items-center justify-center rounded-full shadow-md transition-opacity',
      cIsCenter
        ? 'top-1/2 right-4 -translate-y-1/2 bg-background/25 backdrop-blur-xs hover:bg-background/45 disabled:pointer-events-none disabled:opacity-0'
        : 'right-2 bg-muted backdrop-blur-sm hover:bg-accent',
      !cIsCenter &&
        (cCarousel.canScrollRight
          ? alwaysShowArrows
            ? 'opacity-100'
            : 'opacity-0 group-hover:opacity-100'
          : 'pointer-events-none opacity-0')
    )}
    {...cIsCenter ? { disabled: !cCarousel.canScrollRight } : {}}
    aria-label={fTranslate('text_366bd578350a')}
    onclick={fScrollRight}
  >
    <ChevronRight class={cIconClassDefault} />
  </button>
</div>
