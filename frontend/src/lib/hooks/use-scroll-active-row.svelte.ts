import { untrack } from 'svelte';

/**
 * Scrolls the highlighted row of a picker list into view when the scroll
 * trigger is bumped, without scrolling on mouse hover or result
 * replacement.
 */
export interface UseScrollActiveRowOptions {
  /** Counter bumped by keyboard nav; `undefined` disables the effect. */
  getTrigger: () => number | undefined;
  getContainer: () => HTMLDivElement | null;
  getIndex: () => number;
  getCount: () => number;
  /** Full data attribute marking the row, e.g. `data-picker-index`. */
  dataAttr: string;
}

export function fUseScrollActiveRow(pOpts: UseScrollActiveRowOptions) {
  let vLastTrigger: number | null = null;

  $effect(() => {
    const cTrigger = pOpts.getTrigger();

    if (cTrigger === undefined) return;

    // Skip the initial run on mount: the list opens with the first row
    // already in view, and scrolling here fires before the popover is
    // positioned, which would scroll the whole page to the top.
    if (vLastTrigger === null) {
      vLastTrigger = cTrigger;

      return;
    }

    if (cTrigger === vLastTrigger) return;

    vLastTrigger = cTrigger;
    untrack(() => {
      const cContainer = pOpts.getContainer();
      const cIndex = pOpts.getIndex();

      if (!cContainer || cIndex < 0 || cIndex >= pOpts.getCount()) return;

      const cRow = cContainer.querySelector(
        `[${pOpts.dataAttr}="${cIndex}"]`
      ) as HTMLElement | null;

      cRow?.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    });
  });
}

export type UseScrollActiveRowReturn = ReturnType<typeof fUseScrollActiveRow>;
