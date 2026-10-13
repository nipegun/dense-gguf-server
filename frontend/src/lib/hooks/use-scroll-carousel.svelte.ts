export function fUseScrollCarousel() {
  let vCanScrollLeft = $state(false);
  let vCanScrollRight = $state(false);
  let vScrollContainer = $state<HTMLDivElement | undefined>();
  let vContentContainer = $state<HTMLDivElement | undefined>();

  function fScrollToCenter(pElement: HTMLElement) {
    if (!vScrollContainer) return;

    const cContainerRect = vScrollContainer.getBoundingClientRect();
    const cElementRect = pElement.getBoundingClientRect();
    const cElementCenter = cElementRect.left + cElementRect.width / 2;
    const cContainerCenter = cContainerRect.left + cContainerRect.width / 2;
    const cScrollOffset = cElementCenter - cContainerCenter;

    vScrollContainer.scrollBy({ behavior: 'smooth', left: cScrollOffset });
  }

  function fUpdateScrollButtons() {
    if (!vScrollContainer) return;

    const {
      clientWidth: vClientWidth,
      scrollLeft: vSl,
      scrollWidth: vScrollWidth
    } = vScrollContainer;

    vCanScrollLeft = vSl > 0;
    vCanScrollRight = vSl < vScrollWidth - vClientWidth - 1;
  }

  // Re-evaluate arrow visibility whenever the container or its content resizes,
  // otherwise the arrows may not appear when overflowing items are added (e.g. new
  // tabs/attachments) and the user has not scrolled yet.
  $effect(() => {
    if (!vScrollContainer) return;

    fUpdateScrollButtons();

    const cObserver = new ResizeObserver(() => fUpdateScrollButtons());

    cObserver.observe(vScrollContainer);

    if (vContentContainer) cObserver.observe(vContentContainer);

    return () => cObserver.disconnect();
  });

  return {
    get canScrollLeft() {
      return vCanScrollLeft;
    },
    get canScrollRight() {
      return vCanScrollRight;
    },
    get contentContainer() {
      return vContentContainer;
    },
    set contentContainer(pEl: HTMLDivElement | undefined) {
      vContentContainer = pEl;
    },
    get scrollContainer() {
      return vScrollContainer;
    },
    set scrollContainer(pEl: HTMLDivElement | undefined) {
      vScrollContainer = pEl;
    },
    scrollToCenter: fScrollToCenter,
    updateScrollButtons: fUpdateScrollButtons
  };
}
