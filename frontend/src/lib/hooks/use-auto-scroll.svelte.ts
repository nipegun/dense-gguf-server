import { cAutoScrollAtBottomThreshold, cAutoScrollInterval } from '$lib/constants';

export interface AutoScrollOptions {
  disabled?: boolean;
}

/**
 * Creates an auto-scroll controller for a scrollable container.
 *
 * Features:
 * - Auto-scrolls to bottom during streaming/loading
 * - Stops auto-scroll when user manually scrolls up
 * - Resumes auto-scroll when user scrolls back to bottom
 */
export class AutoScrollController {
  private vAutoScrollEnabled = $state(true);
  private vContainer: HTMLElement | undefined;
  private _disabled: boolean;
  private _lastScrollTop = $state(0);
  private vMutationObserver: MutationObserver | null = null;
  private vObserverEnabled = false;
  private vRafPending = false;
  private vScrollInterval: ReturnType<typeof setInterval> | undefined;
  private _userScrolledUp = $state(false);
  get autoScrollEnabled(): boolean {
    return this.vAutoScrollEnabled;
  }

  get userScrolledUp(): boolean {
    return this._userScrolledUp;
  }

  constructor(pOptions: AutoScrollOptions = {}) {
    this._disabled = pOptions.disabled ?? false;
  }

  /**
   * Cleans up resources. Call this in onDestroy or when the component unmounts.
   */
  mDestroy(): void {
    this.mStopInterval();
    this.mDoStopObserving();
  }

  /**
   * Enables auto-scroll (e.g., when user sends a message).
   */
  mEnable(): void {
    if (this._disabled) return;

    this._userScrolledUp = false;
    this.vAutoScrollEnabled = true;
  }

  /**
   * Handles scroll events to detect user scroll direction and toggle auto-scroll.
   */
  mHandleScroll(): void {
    if (this._disabled || !this.vContainer) return;

    const {
      clientHeight: vClientHeight,
      scrollHeight: vScrollHeight,
      scrollTop: vScrollTop
    } = this.vContainer;
    const cDistanceFromBottom = vScrollHeight - vClientHeight - vScrollTop;
    const cIsScrollingUp = vScrollTop < this._lastScrollTop;
    const cIsAtBottom = cDistanceFromBottom < cAutoScrollAtBottomThreshold;

    if (cIsScrollingUp && !cIsAtBottom) {
      this._userScrolledUp = true;
      this.vAutoScrollEnabled = false;
    } else if (cIsAtBottom && this._userScrolledUp) {
      this._userScrolledUp = false;
      this.vAutoScrollEnabled = true;
    }

    this._lastScrollTop = vScrollTop;
  }

  /**
   * Resets scroll state when switching conversations.
   */
  mResetScrollState(): void {
    this._userScrolledUp = false;
    this.vAutoScrollEnabled = !this._disabled;

    if (this.vContainer) {
      this._lastScrollTop = this.vContainer.scrollTop;
    }
  }

  /**
   * Scrolls the container to the bottom instantly.
   */
  mScrollToBottom(): void {
    if (this._disabled || !this.vContainer) return;

    this.vContainer.scrollTop = this.vContainer.scrollHeight;
  }

  /**
   * Binds the controller to a scrollable container element.
   */
  mSetContainer(pContainer: HTMLElement | undefined): void {
    this.mDoStopObserving();
    this.vContainer = pContainer;

    if (this.vObserverEnabled && pContainer && !this._disabled) {
      this.mDoStartObserving();
    }
  }

  /**
   * Updates the disabled state.
   */
  mSetDisabled(pDisabled: boolean): void {
    if (this._disabled === pDisabled) return;

    this._disabled = pDisabled;

    if (pDisabled) {
      this.vAutoScrollEnabled = false;
      this.mStopInterval();
      this.mDoStopObserving();
    } else if (this.vObserverEnabled && this.vContainer && !this.vMutationObserver) {
      this.mDoStartObserving();
    }
  }

  /**
   * Starts the auto-scroll interval for continuous scrolling during streaming.
   */
  mStartInterval(): void {
    if (this._disabled || this.vScrollInterval) return;

    this.vScrollInterval = setInterval(() => {
      this.mScrollToBottom();
    }, cAutoScrollInterval);
  }

  /**
   * Starts a MutationObserver on the container that auto-scrolls to bottom
   * on content changes. More responsive than interval-based polling.
   */
  mStartObserving(): void {
    this.vObserverEnabled = true;

    if (this.vContainer && !this._disabled && !this.vMutationObserver) {
      this.mDoStartObserving();
    }
  }

  /**
   * Stops the auto-scroll interval.
   */
  mStopInterval(): void {
    if (this.vScrollInterval) {
      clearInterval(this.vScrollInterval);
      this.vScrollInterval = undefined;
    }
  }

  /**
   * Stops the MutationObserver.
   */
  mStopObserving(): void {
    this.vObserverEnabled = false;
    this.mDoStopObserving();
  }

  /**
   * Updates the auto-scroll interval based on streaming state.
   * Call this in a $effect to automatically manage the interval.
   */
  mUpdateInterval(pIsStreaming: boolean): void {
    if (this._disabled) {
      this.mStopInterval();

      return;
    }

    if (pIsStreaming && this.vAutoScrollEnabled) {
      if (!this.vScrollInterval) {
        this.mStartInterval();
      }
    } else {
      this.mStopInterval();
    }
  }

  private mDoStartObserving(): void {
    if (!this.vContainer || this.vMutationObserver) return;

    this.vMutationObserver = new MutationObserver(() => {
      if (!this.vAutoScrollEnabled || this.vRafPending) return;

      this.vRafPending = true;
      requestAnimationFrame(() => {
        this.vRafPending = false;

        if (this.vAutoScrollEnabled && this.vContainer) {
          this.vContainer.scrollTop = this.vContainer.scrollHeight;
        }
      });
    });

    this.vMutationObserver.observe(this.vContainer, {
      characterData: true,
      childList: true,
      subtree: true
    });
  }

  private mDoStopObserving(): void {
    if (this.vMutationObserver) {
      this.vMutationObserver.disconnect();
      this.vMutationObserver = null;
    }

    this.vRafPending = false;
  }
}

/**
 * Creates a new AutoScrollController instance.
 */
export function fCreateAutoScrollController(
  pOptions: AutoScrollOptions = {}
): AutoScrollController {
  return new AutoScrollController(pOptions);
}
