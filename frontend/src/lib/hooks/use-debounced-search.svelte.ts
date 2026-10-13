import { fDebounce } from '$lib/utils/debounce';

/**
 * Shared debounced async-search machinery for the chat-form pickers:
 * AbortController + sequence counter to discard stale responses, a
 * debounce, and a live `isSearching` flag.
 */

export interface UseDebouncedSearchOptions {
  debounceMs: number;
  /** Fire-time guard: a scheduled call that outlives a reset is dropped. */
  canRun: () => boolean;
  /** Live query, used to drop a scheduled call whose query changed. */
  getQuery: () => string;
  /** Perform the search and commit results; bail out when `isCurrent()` is false. */
  mRun: (pQuery: string, pSignal: AbortSignal, pIsCurrent: () => boolean) => void | Promise<void>;
}

export function fUseDebouncedSearch(pOpts: UseDebouncedSearchOptions) {
  let vController: AbortController | null = null;
  let vSearchSeq = 0;
  let vIsSearching = $state(false);

  function fIsCurrent(pSeq: number) {
    return pSeq === vSearchSeq;
  }

  function fCancel() {
    vController?.abort();
    vSearchSeq++;
    vIsSearching = false;
  }

  const fSchedule = fDebounce((pQuery: string) => {
    if (!pOpts.canRun() || pQuery !== pOpts.getQuery().trim()) return;

    void fStart(pQuery);
  }, pOpts.debounceMs);

  async function fStart(pQuery: string) {
    fCancel();
    const cFresh = new AbortController();

    vController = cFresh;
    const cMySeq = ++vSearchSeq;

    vIsSearching = true;
    try {
      await pOpts.mRun(pQuery, cFresh.signal, () => fIsCurrent(cMySeq));
    } finally {
      if (fIsCurrent(cMySeq)) vIsSearching = false;
    }
  }

  return {
    cancel: fCancel,
    get isSearching() {
      return vIsSearching;
    },
    mRun(pQuery: string) {
      fSchedule(pQuery);
    },
    /** Bump the loading flag synchronously (e.g. before the debounce fires). */
    mSetLoading(pValue: boolean) {
      vIsSearching = pValue;
    }
  };
}

export type UseDebouncedSearchReturn = ReturnType<typeof fUseDebouncedSearch>;
