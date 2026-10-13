import { KeyboardKey } from '$lib/enums';

/**
 * Shared keyboard navigation state for the chat-form pickers: a highlighted
 * row, a scroll trigger, and Arrow/Escape/Enter handling.
 */
export interface UsePickerNavigationOptions {
  /** Gates all key handling. */
  isOpen: () => boolean;
  count: () => number;
  /**
   * Resolve the row to highlight for a movement step, or -1 when no move
   * is possible. Defaults to plain wraparound across `count()`.
   */
  step?: (pFrom: number, pDir: 1 | -1) => number;
  onClose: () => void;
  /** Called on Enter when `hoveredIndex` points at a selectable row. */
  onSelect: (pIndex: number) => void;
}

function fWrapStep(pFrom: number, pDir: 1 | -1, pCount: number): number {
  return pDir === 1 ? (pFrom + 1) % pCount : pFrom <= 0 ? pCount - 1 : pFrom - 1;
}

export function fUsePickerNavigation(pOpts: UsePickerNavigationOptions) {
  let vHoveredIndex = $state(-1);
  let vScrollTrigger = $state(0);

  function fResolve(pFrom: number, pDir: 1 | -1): number {
    const cN = pOpts.count();

    if (cN === 0) return -1;

    if (pOpts.step) return pOpts.step(pFrom, pDir);

    return fWrapStep(pFrom, pDir, cN);
  }

  function fMove(pDir: 1 | -1) {
    const cNext = fResolve(vHoveredIndex, pDir);

    if (cNext >= 0) {
      vHoveredIndex = cNext;
      vScrollTrigger++;
    }
  }

  /** Reset the highlight without bumping the scroll trigger. */
  function fReset(pIndex: number) {
    vHoveredIndex = pIndex;
  }

  /** Bump the scroll trigger without moving the highlight. */
  function fBumpScroll() {
    vScrollTrigger++;
  }

  /** Mouse hover highlights a row but must NOT bump the scroll trigger. */
  function fSetHover(pIndex: number) {
    vHoveredIndex = pIndex;
  }

  function fHandleKeydown(pEvent: KeyboardEvent): boolean {
    if (!pOpts.isOpen()) return false;

    if (pEvent.key === KeyboardKey.cEscape) {
      pEvent.preventDefault();
      pOpts.onClose();

      return true;
    }

    if (pEvent.key === KeyboardKey.cArrowDown) {
      pEvent.preventDefault();
      fMove(1);

      return true;
    }

    if (pEvent.key === KeyboardKey.cArrowUp) {
      pEvent.preventDefault();
      fMove(-1);

      return true;
    }

    if (pEvent.key === KeyboardKey.cEnter) {
      if (vHoveredIndex >= 0 && vHoveredIndex < pOpts.count()) {
        pEvent.preventDefault();
        pOpts.onSelect(vHoveredIndex);

        return true;
      }

      // No selectable row - let the caller's Enter-to-submit run.
      return false;
    }

    return false;
  }

  return {
    bumpScroll: fBumpScroll,
    handleKeydown: fHandleKeydown,
    get hoveredIndex() {
      return vHoveredIndex;
    },
    move: fMove,
    reset: fReset,
    get scrollTrigger() {
      return vScrollTrigger;
    },
    setHover: fSetHover
  };
}

export type UsePickerNavigationReturn = ReturnType<typeof fUsePickerNavigation>;
