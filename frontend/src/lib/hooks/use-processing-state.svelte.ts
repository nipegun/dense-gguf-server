import { cStatsUnits } from '$lib/constants';
import { cChatStore } from '$lib/stores';
import type { ApiProcessingState, LiveGenerationStats, LiveProcessingStats } from '$lib/types';

export interface UseProcessingStateReturn {
  readonly processingState: ApiProcessingState | null;
  mGetProcessingDetails(): string[];
  mGetTechnicalDetails(): string[];
  mGetProcessingMessage(): string;
  mGetPromptProgressText(): string | null;
  mGetLiveProcessingStats(): LiveProcessingStats | null;
  mGetLiveGenerationStats(): LiveGenerationStats | null;
  mShouldShowDetails(): boolean;
  mStartMonitoring(): void;
  mStopMonitoring(): void;
}

/**
 * useProcessingState - Reactive processing state hook
 *
 * This hook provides reactive access to the processing state of the server.
 * It directly reads from chatStore's reactive state and provides
 * formatted processing details for UI display.
 *
 * **Features:**
 * - Real-time processing state via direct reactive state binding
 * - Context and output token tracking
 * - Tokens per second calculation
 * - Automatic updates when streaming data arrives
 * - Supports multiple concurrent conversations
 *
 * @returns Hook interface with processing state and control methods
 */
export function fUseProcessingState(): UseProcessingStateReturn {
  let vIsMonitoring = $state(false);
  let vLastKnownState = $state<ApiProcessingState | null>(null);
  let vLastKnownProcessingStats = $state<LiveProcessingStats | null>(null);

  // Derive processing state reactively from chatStore's direct state
  const cProcessingState = $derived.by(() => {
    if (!vIsMonitoring) {
      return vLastKnownState;
    }

    // Read directly from the reactive state
    return cChatStore.processing.vActiveState;
  });

  $effect(() => {
    if (cProcessingState && vIsMonitoring) {
      vLastKnownState = cProcessingState;
    }
  });

  // Track last known processing stats for when promptProgress disappears
  $effect(() => {
    if (cProcessingState?.promptProgress) {
      const {
        cache: vCache,
        processed: vProcessed,
        time_ms: vTimeMs,
        total: vTotal
      } = cProcessingState.promptProgress;
      const cActualProcessed = vProcessed - vCache;
      const cActualTotal = vTotal - vCache;

      if (cActualProcessed > 0 && vTimeMs > 0) {
        const cTokensPerSecond = cActualProcessed / (vTimeMs / 1000);

        vLastKnownProcessingStats = {
          timeMs: vTimeMs,
          tokensPerSecond: cTokensPerSecond,
          tokensProcessed: cActualProcessed,
          totalTokens: cActualTotal
        };
      }
    }
  });

  function fGetETASecs(pDone: number, pTotal: number, pElapsedMs: number): number | undefined {
    const cElapsedSecs = pElapsedMs / 1000;
    const cProgressETASecs =
      pDone === 0 || cElapsedSecs < 0.5
        ? undefined // can be the case for the 0% progress report
        : cElapsedSecs * (pTotal / pDone - 1);

    return cProgressETASecs;
  }

  function fStartMonitoring(): void {
    if (vIsMonitoring) return;

    vIsMonitoring = true;
  }

  function fStopMonitoring(): void {
    if (!vIsMonitoring) return;

    vIsMonitoring = false;
  }

  function fGetProcessingMessage(): string {
    if (!cProcessingState) {
      return 'Processing...';
    }

    switch (cProcessingState.status) {
      case 'initializing':
        return 'Initializing...';
      case 'preparing':
        if (cProcessingState.progressPercent !== undefined) {
          return `Processing (${cProcessingState.progressPercent}%)`;
        }

        return 'Preparing response...';
      case 'generating':
        return '';
      default:
        return 'Processing...';
    }
  }

  function fGetProcessingDetails(): string[] {
    // Use current processing state or fall back to last known state
    const cStateToUse = cProcessingState || vLastKnownState;

    if (!cStateToUse) {
      return [];
    }

    const aDetails: string[] = [];

    // Show prompt processing progress with ETA during preparation phase
    if (cStateToUse.promptProgress) {
      const {
        cache: vCache,
        processed: vProcessed,
        time_ms: vTimeMs,
        total: vTotal
      } = cStateToUse.promptProgress;
      const cActualProcessed = vProcessed - vCache;
      const cActualTotal = vTotal - vCache;

      if (cActualProcessed < cActualTotal && cActualProcessed > 0) {
        const cPercent = Math.round((cActualProcessed / cActualTotal) * 100);
        const cEta = fGetETASecs(cActualProcessed, cActualTotal, vTimeMs);

        if (cEta !== undefined) {
          const cEtaSecs = Math.ceil(cEta);

          aDetails.push(`Processing ${cPercent}% (ETA: ${cEtaSecs}s)`);
        } else {
          aDetails.push(`Processing ${cPercent}%`);
        }
      }
    }

    // Always show context info when we have valid data
    if (
      typeof cStateToUse.contextTotal === 'number' &&
      cStateToUse.contextUsed >= 0 &&
      cStateToUse.contextTotal > 0
    ) {
      const cContextPercent = Math.round(
        (cStateToUse.contextUsed / cStateToUse.contextTotal) * 100
      );

      aDetails.push(
        `Context: ${cStateToUse.contextUsed}/${cStateToUse.contextTotal} (${cContextPercent}%)`
      );
    }

    if (cStateToUse.outputTokensUsed > 0) {
      // Handle infinite max_tokens (-1) case
      if (cStateToUse.outputTokensMax <= 0) {
        aDetails.push(`Output: ${cStateToUse.outputTokensUsed}/∞`);
      } else {
        const cOutputPercent = Math.round(
          (cStateToUse.outputTokensUsed / cStateToUse.outputTokensMax) * 100
        );

        aDetails.push(
          `Output: ${cStateToUse.outputTokensUsed}/${cStateToUse.outputTokensMax} (${cOutputPercent}%)`
        );
      }
    }

    if (cStateToUse.tokensPerSecond && cStateToUse.tokensPerSecond > 0) {
      aDetails.push(`${cStateToUse.tokensPerSecond.toFixed(1)} ${cStatsUnits.TOKENS_PER_SECOND}`);
    }

    if (cStateToUse.speculative) {
      aDetails.push('Speculative decoding enabled');
    }

    return aDetails;
  }

  /**
   * Returns technical details without the progress message (for bottom bar)
   */
  function fGetTechnicalDetails(): string[] {
    const cStateToUse = cProcessingState || vLastKnownState;

    if (!cStateToUse) {
      return [];
    }

    const aDetails: string[] = [];

    // Always show context info when we have valid data
    if (
      typeof cStateToUse.contextTotal === 'number' &&
      cStateToUse.contextUsed >= 0 &&
      cStateToUse.contextTotal > 0
    ) {
      const cContextPercent = Math.round(
        (cStateToUse.contextUsed / cStateToUse.contextTotal) * 100
      );

      aDetails.push(
        `Context: ${cStateToUse.contextUsed}/${cStateToUse.contextTotal} (${cContextPercent}%)`
      );
    }

    if (cStateToUse.outputTokensUsed > 0) {
      // Handle infinite max_tokens (-1) case
      if (cStateToUse.outputTokensMax <= 0) {
        aDetails.push(`Output: ${cStateToUse.outputTokensUsed}/∞`);
      } else {
        const cOutputPercent = Math.round(
          (cStateToUse.outputTokensUsed / cStateToUse.outputTokensMax) * 100
        );

        aDetails.push(
          `Output: ${cStateToUse.outputTokensUsed}/${cStateToUse.outputTokensMax} (${cOutputPercent}%)`
        );
      }
    }

    if (cStateToUse.tokensPerSecond && cStateToUse.tokensPerSecond > 0) {
      aDetails.push(`${cStateToUse.tokensPerSecond.toFixed(1)} ${cStatsUnits.TOKENS_PER_SECOND}`);
    }

    if (cStateToUse.speculative) {
      aDetails.push('Speculative decoding enabled');
    }

    return aDetails;
  }

  function fShouldShowDetails(): boolean {
    return cProcessingState !== null && cProcessingState.status !== 'idle';
  }

  /**
   * Returns a short progress message with percent
   */
  function fGetPromptProgressText(): string | null {
    if (!cProcessingState?.promptProgress) return null;

    const { cache: vCache, processed: vProcessed, total: vTotal } = cProcessingState.promptProgress;
    const cActualProcessed = vProcessed - vCache;
    const cActualTotal = vTotal - vCache;
    const cPercent = Math.round((cActualProcessed / cActualTotal) * 100);
    const cEta = fGetETASecs(
      cActualProcessed,
      cActualTotal,
      cProcessingState.promptProgress.time_ms
    );

    if (cEta !== undefined) {
      const cEtaSecs = Math.ceil(cEta);

      return `Processing ${cPercent}% (ETA: ${cEtaSecs}s)`;
    }

    return `Processing ${cPercent}%`;
  }

  /**
   * Returns live processing statistics for display (prompt processing phase)
   * Returns last known stats when promptProgress becomes unavailable
   */
  function fGetLiveProcessingStats(): LiveProcessingStats | null {
    if (cProcessingState?.promptProgress) {
      const {
        cache: vCache,
        processed: vProcessed,
        time_ms: vTimeMs,
        total: vTotal
      } = cProcessingState.promptProgress;
      const cActualProcessed = vProcessed - vCache;
      const cActualTotal = vTotal - vCache;

      if (cActualProcessed > 0 && vTimeMs > 0) {
        const cTokensPerSecond = cActualProcessed / (vTimeMs / 1000);

        return {
          timeMs: vTimeMs,
          tokensPerSecond: cTokensPerSecond,
          tokensProcessed: cActualProcessed,
          totalTokens: cActualTotal
        };
      }
    }

    // Return last known stats if promptProgress is no longer available
    return vLastKnownProcessingStats;
  }

  /**
   * Returns live generation statistics for display (token generation phase)
   */
  function fGetLiveGenerationStats(): LiveGenerationStats | null {
    if (!cProcessingState) return null;

    const { tokensDecoded: vTokensDecoded, tokensPerSecond: vTokensPerSecond } = cProcessingState;

    if (vTokensDecoded <= 0) return null;

    // Calculate time from tokens and speed
    const cTimeMs =
      vTokensPerSecond && vTokensPerSecond > 0 ? (vTokensDecoded / vTokensPerSecond) * 1000 : 0;

    return {
      timeMs: cTimeMs,
      tokensGenerated: vTokensDecoded,
      tokensPerSecond: vTokensPerSecond || 0
    };
  }

  return {
    mGetLiveGenerationStats: fGetLiveGenerationStats,
    mGetLiveProcessingStats: fGetLiveProcessingStats,
    mGetProcessingDetails: fGetProcessingDetails,
    mGetProcessingMessage: fGetProcessingMessage,
    mGetPromptProgressText: fGetPromptProgressText,
    mGetTechnicalDetails: fGetTechnicalDetails,
    get processingState() {
      return cProcessingState;
    },
    mShouldShowDetails: fShouldShowDetails,
    mStartMonitoring: fStartMonitoring,
    mStopMonitoring: fStopMonitoring
  };
}
