/**
 * chatProcessingStore - Per-conversation processing state
 *
 * Owns the live processing snapshot shown while a conversation streams:
 * token counts, tokens/sec, prompt progress. Updated from stream timings,
 * restored from persisted message timings when a conversation loads.
 *
 * Composed under chatStore.processing; not exported from the stores barrel.
 */

import { MessageRole } from '$lib/enums';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cModelsStore } from '$lib/stores/models/index.svelte';
import { cServerStore } from '$lib/stores/server.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import type {
  ApiProcessingState,
  ChatMessagePromptProgress,
  ChatMessageTimings,
  DatabaseMessage
} from '$lib/types';
import { SvelteMap } from 'svelte/reactivity';

interface ProcessingTimingData {
  cache_n: number;
  predicted_n: number;
  predicted_per_second: number;
  prompt_ms?: number;
  prompt_n: number;
  prompt_progress?: ChatMessagePromptProgress;
}

export class ChatProcessingStore {
  private vActiveConversationId = $state<string | null>(null);
  private vStates = new SvelteMap<string, ApiProcessingState>();

  /** Processing state of the conversation currently shown in the UI. */
  vActiveState = $derived(
    this.vActiveConversationId ? (this.vStates.get(this.vActiveConversationId) ?? null) : null
  );

  get activeConversationId(): string | null {
    return this.vActiveConversationId;
  }

  /**
   * Applies a stream timings event (tokens/sec + token counts) to the given
   * conversation's processing state. Shared by the chat and continue flows.
   */
  mApplyStreamTimings(
    pTimings?: ChatMessageTimings,
    pPromptProgress?: ChatMessagePromptProgress,
    pConversationId?: string
  ): void {
    const cTokensPerSecond =
      pTimings?.predicted_ms && pTimings?.predicted_n
        ? (pTimings.predicted_n / pTimings.predicted_ms) * 1000
        : 0;

    this.mUpdateFromTimings(
      {
        cache_n: pTimings?.cache_n || 0,
        predicted_n: pTimings?.predicted_n || 0,
        predicted_per_second: cTokensPerSecond,
        prompt_ms: pTimings?.prompt_ms,
        prompt_n: pTimings?.prompt_n || 0,
        prompt_progress: pPromptProgress
      },
      pConversationId
    );
  }

  mGetConversationIds(): string[] {
    return Array.from(this.vStates.keys());
  }

  mGetState(pConversationId: string): ApiProcessingState | null {
    return this.vStates.get(pConversationId) ?? null;
  }

  mRestoreFromMessages(pMessages: DatabaseMessage[], pConversationId: string): void {
    for (let vI = pMessages.length - 1; vI >= 0; vI--) {
      const message = pMessages[vI];

      if (message.role === MessageRole.cAssistant && message.timings) {
        this.mSetState(
          pConversationId,
          this.mParseTimingData({
            cache_n: message.timings.cache_n || 0,
            predicted_n: message.timings.predicted_n || 0,
            predicted_per_second:
              message.timings.predicted_n && message.timings.predicted_ms
                ? (message.timings.predicted_n / message.timings.predicted_ms) * 1000
                : 0,
            prompt_ms: message.timings.prompt_ms,
            prompt_n: message.timings.prompt_n || 0
          })
        );

        return;
      }
    }
  }

  mSetActiveConversation(pConversationId: string | null): void {
    this.vActiveConversationId = pConversationId;
  }

  /** Passing null clears the state for the conversation. */
  mSetState(pConversationId: string, pState: ApiProcessingState | null): void {
    if (pState === null) this.vStates.delete(pConversationId);
    else this.vStates.set(pConversationId, pState);
  }

  mUpdateFromTimings(pTimingData: ProcessingTimingData, pConversationId?: string): void {
    const cTargetId = pConversationId || this.vActiveConversationId;

    if (cTargetId) {
      this.mSetState(cTargetId, this.mParseTimingData(pTimingData));
    }
  }

  private mGetContextTotal(): number | null {
    const cActiveConvId = this.vActiveConversationId;
    const cActiveState = cActiveConvId ? this.mGetState(cActiveConvId) : null;

    if (
      cActiveState &&
      typeof cActiveState.contextTotal === 'number' &&
      cActiveState.contextTotal > 0
    )
      return cActiveState.contextTotal;

    if (cServerStore.isRouterMode) {
      const cModelContextSize = cModelsStore.selectedModelContextSize;

      if (typeof cModelContextSize === 'number' && cModelContextSize > 0) {
        return cModelContextSize;
      }
    } else {
      const cPropsContextSize = cServerStore.contextSize;

      if (typeof cPropsContextSize === 'number' && cPropsContextSize > 0) {
        return cPropsContextSize;
      }
    }

    return null;
  }

  private mParseTimingData(pTimingData: ProcessingTimingData): ApiProcessingState {
    const cCacheTokens = pTimingData.cache_n || 0,
      cPredictedTokens = pTimingData.predicted_n || 0,
      cPromptMs = pTimingData.prompt_ms || undefined,
      cPromptTokens = pTimingData.prompt_n || 0,
      cTokensPerSecond = pTimingData.predicted_per_second || 0;
    const cPromptProgress = pTimingData.prompt_progress;
    const cContextTotal = this.mGetContextTotal();
    const cCurrentConfig = cSettingsStore.vConfig;
    const cOutputTokensMax = cCurrentConfig.max_tokens || -1;
    const cContextUsed = cPromptTokens + cCacheTokens + cPredictedTokens,
      cOutputTokensUsed = cPredictedTokens;
    const cProgressCache = cPromptProgress?.cache || 0,
      cProgressActualDone = (cPromptProgress?.processed ?? 0) - cProgressCache,
      cProgressActualTotal = (cPromptProgress?.total ?? 0) - cProgressCache;
    const cProgressPercent = cPromptProgress
      ? Math.round((cProgressActualDone / cProgressActualTotal) * 100)
      : undefined;

    return {
      cacheTokens: cCacheTokens,
      contextTotal: cContextTotal,
      contextUsed: cContextUsed,
      hasNextToken: cPredictedTokens > 0,
      outputTokensMax: cOutputTokensMax,
      outputTokensUsed: cOutputTokensUsed,
      progressPercent: cProgressPercent,
      promptMs: cPromptMs,
      promptProgress: cPromptProgress,
      promptTokens: cPromptTokens,
      speculative: false,
      status: cPredictedTokens > 0 ? 'generating' : cPromptProgress ? 'preparing' : 'idle',
      temperature: cCurrentConfig.temperature ?? 0.8,
      tokensDecoded: cPredictedTokens,
      tokensPerSecond: cTokensPerSecond,
      tokensRemaining: cOutputTokensMax - cPredictedTokens,
      topP: cCurrentConfig.top_p ?? 0.95
    };
  }
}

export const cChatProcessingStore = new ChatProcessingStore();
