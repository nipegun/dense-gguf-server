/**
 * ContextStatsStore - Context window usage stats for the active conversation
 *
 * Combines token usage persisted in message timings metadata with
 * server-originating data: model context size from /props (modelsStore)
 * and live processing state while streaming (chatStore).
 */

import { MessageRole } from '$lib/enums';
// direct imports between stores, not via the barrel, to avoid circular deps
import { cAgenticStore } from '$lib/stores/agentic/index.svelte';
import { cChatStore } from '$lib/stores/chat/index.svelte';
import { cConversationsStore } from '$lib/stores/conversations/index.svelte';
import { cModelsStore } from '$lib/stores/models/index.svelte';
import { cServerStore } from '$lib/stores/server.svelte';
import type {
  ApiProcessingState,
  ChatMessageAgenticTimings,
  ChatMessageTimings,
  DatabaseMessage
} from '$lib/types';

interface LiveStats {
  freshTokens: number;
  promptTokens: number;
  cacheTokens: number;
  outputTokens: number;
}

interface AssistantTimingsSummary {
  lastAgenticLlm: ChatMessageAgenticTimings['llm'] | undefined;
  lastTimings: ChatMessageTimings | undefined;
  cacheTotal: number;
  output: number;
  outputMs: number;
  read: number;
}

/**
 * One forward pass over the messages computing everything the deriveds
 * below need: the last assistant timings (per-turn gauges), the last
 * agentic llm totals (cumulative gauge) and the cumulative sums. During
 * streaming activeMessages churns every chunk, and each of these used to be
 * its own O(n) scan re-run per chunk.
 */
function fSummarizeAssistantTimings(pMessages: DatabaseMessage[]): AssistantTimingsSummary {
  let vLastAgenticLlm: ChatMessageAgenticTimings['llm'] | undefined;
  let vLastTimings: ChatMessageTimings | undefined;
  let vRead = 0;
  let vCacheTotal = 0;
  let vOutput = 0;
  let vOutputMs = 0;

  for (const cM of pMessages) {
    if (cM.role !== MessageRole.cAssistant || !cM.timings) continue;

    vLastTimings = cM.timings;

    if (cM.timings.agentic?.llm?.predicted_n != null) {
      vLastAgenticLlm = cM.timings.agentic.llm;
    }

    vRead += cM.timings.prompt_n ?? 0;
    vCacheTotal += cM.timings.cache_n ?? 0;
    vOutput += cM.timings.predicted_n ?? 0;
    vOutputMs += cM.timings.predicted_ms ?? 0;
  }

  return {
    cacheTotal: vCacheTotal,
    lastAgenticLlm: vLastAgenticLlm,
    lastTimings: vLastTimings,
    output: vOutput,
    outputMs: vOutputMs,
    read: vRead
  };
}

function fDeriveLiveStats(pState: ApiProcessingState | null): LiveStats | null {
  if (!pState || (pState.status !== 'preparing' && pState.status !== 'generating')) {
    return null;
  }

  const cPromptTokens = pState.promptTokens ?? 0;
  const cCacheTokens = pState.cacheTokens ?? 0;

  return {
    cacheTokens: cCacheTokens,
    freshTokens: cPromptTokens,
    outputTokens: pState.outputTokensUsed ?? 0,
    promptTokens: cPromptTokens + cCacheTokens
  };
}

class ContextStatsStore {
  // The canonical resolution lives in modelsStore.activeModelId.
  vActiveModelId = $derived(cModelsStore.activeModelId);

  // shared by currentRead/Fresh/Cache/Output and cumulative so a per-chunk
  // churn of activeMessages triggers exactly one scan instead of one per
  // derived
  private vAssistantTimings = $derived.by(() =>
    fSummarizeAssistantTimings(cConversationsStore.aActiveMessages as DatabaseMessage[])
  );

  private vCumulative = $derived.by(() => {
    const cConvId = cConversationsStore.vActiveConversation?.id;
    // A running agentic flow stamps llm totals on messages only when it
    // exits, so read its live session totals instead.
    const cLiveLlm = cConvId ? cAgenticStore.mGetLiveLlmTotals(cConvId) : null;

    if (cLiveLlm) {
      const cOutputMs = cLiveLlm.predicted_ms;
      const cAverageTokensPerSecond =
        cOutputMs > 0 && cLiveLlm.predicted_n > 0
          ? (cLiveLlm.predicted_n / cOutputMs) * 1000
          : null;

      return {
        averageTokensPerSecond: cAverageTokensPerSecond,
        cacheTotal: 0,
        output: cLiveLlm.predicted_n,
        read: cLiveLlm.prompt_n
      };
    }

    const {
      cacheTotal: vCacheTotal,
      lastAgenticLlm: vLastAgenticLlm,
      output: vOutput,
      outputMs: vOutputMs,
      read: vRead
    } = this.vAssistantTimings;

    // Agentic sessions stamp the same agentic.llm totals onto every
    // assistant message; cache_n is never per-turn so cache_total stays 0.
    if (vLastAgenticLlm) {
      const cAverageTokensPerSecond =
        vLastAgenticLlm.predicted_ms > 0 && vLastAgenticLlm.predicted_n > 0
          ? (vLastAgenticLlm.predicted_n / vLastAgenticLlm.predicted_ms) * 1000
          : null;

      return {
        averageTokensPerSecond: cAverageTokensPerSecond,
        cacheTotal: 0,
        output: vLastAgenticLlm.predicted_n ?? 0,
        read: vLastAgenticLlm.prompt_n ?? 0
      };
    }

    const cAverageTokensPerSecond =
      vOutputMs > 0 && vOutput > 0 ? (vOutput / vOutputMs) * 1000 : null;

    return {
      averageTokensPerSecond: cAverageTokensPerSecond,
      cacheTotal: vCacheTotal,
      output: vOutput,
      read: vRead
    };
  });

  vAverageTokensPerSecond = $derived(this.vCumulative.averageTokensPerSecond);

  vContextTotal = $derived.by(() => {
    void cModelsStore.props.vCacheVersion;

    return this.vActiveModelId
      ? cModelsStore.props.mGetModelContextSize(this.vActiveModelId)
      : null;
  });

  private vLiveStats = $derived(fDeriveLiveStats(cChatStore.processing.vActiveState));

  vCurrentOutput = $derived.by(() => {
    if (this.vLiveStats && this.vLiveStats.outputTokens > 0) return this.vLiveStats.outputTokens;

    return this.vAssistantTimings.lastTimings?.predicted_n ?? 0;
  });

  vCurrentRead = $derived.by(() => {
    const cTimings = this.vAssistantTimings.lastTimings;

    let vRead = 0;

    if (cTimings) {
      vRead = (cTimings.prompt_n ?? 0) + (cTimings.cache_n ?? 0);
    }

    // live.promptTokens is already the combined reading (prompt + cache),
    // so do not also add live.cacheTokens.
    if (this.vLiveStats && this.vLiveStats.promptTokens > 0) {
      vRead = Math.max(vRead, this.vLiveStats.promptTokens);
    }

    return vRead;
  });

  vContextUsed = $derived(this.vCurrentRead + this.vCurrentOutput);

  vContextAvailable = $derived(
    this.vContextTotal !== null ? this.vContextTotal - this.vContextUsed : null
  );

  vContextPercent = $derived.by(() => {
    if (this.vContextTotal === null || this.vContextTotal <= 0) return null;

    return Math.round((this.vContextUsed / this.vContextTotal) * 100);
  });

  vCumulativeCacheTotal = $derived(this.vCumulative.cacheTotal);

  vCumulativeOutput = $derived(this.vCumulative.output);

  vCumulativeRead = $derived(this.vCumulative.read);

  vCurrentCache = $derived.by(() => {
    const cCached = this.vAssistantTimings.lastTimings?.cache_n ?? 0;

    if (this.vLiveStats && this.vLiveStats.promptTokens > 0) {
      return Math.max(cCached, this.vLiveStats.cacheTokens);
    }

    return cCached;
  });

  vCurrentFresh = $derived.by(() => {
    const cFresh = this.vAssistantTimings.lastTimings?.prompt_n ?? 0;

    return Math.max(cFresh, this.vLiveStats?.freshTokens ?? 0);
  });

  vIsActiveModelLoaded = $derived(
    this.vActiveModelId !== null &&
      (!cServerStore.isRouterMode || cModelsStore.mIsModelLoaded(this.vActiveModelId))
  );

  vIsActiveModelLoading = $derived(
    this.vActiveModelId !== null && cModelsStore.status.mIsOperationInProgress(this.vActiveModelId)
  );

  vKvTotal = $derived(this.vCurrentRead + this.vCurrentOutput);
}

export const cContextStatsStore = new ContextStatsStore();
