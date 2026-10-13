import type { ActionReturn } from 'svelte/action';

/** Margin around the viewport that already counts as "near". */
const cNearViewportMargin = '200px';

/**
 * Calls back once the element comes near the viewport, so per-row work that costs
 * a request (avatars, Hub details) does not fire on mount.
 */
export function fNearViewport(node: HTMLElement, pOnNear: () => void): ActionReturn {
  if (typeof IntersectionObserver === 'undefined') {
    pOnNear();

    return {};
  }

  const cObserver = new IntersectionObserver(
    (entries) => {
      if (entries.some((pEntry) => pEntry.isIntersecting)) {
        cObserver.disconnect();
        pOnNear();
      }
    },
    { rootMargin: cNearViewportMargin }
  );

  cObserver.observe(node);

  return {
    destroy() {
      cObserver.disconnect();
    }
  };
}
