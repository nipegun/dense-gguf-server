<script lang="ts">
  import { ChatMessageStatistics } from '$lib/components/app';
  import { ChatMessageStatisticsMode } from '$lib/enums';
  import type { UseProcessingStateReturn } from '$lib/hooks/use-processing-state.svelte';
  import { cAgenticStore } from '$lib/stores';

  interface Props {
    message: DatabaseMessage;
    isLoading: boolean;
    processingState: UseProcessingStateReturn;
    showMessageStats: boolean;
  }

  let { isLoading, message, processingState, showMessageStats }: Props = $props();

  // A running agentic flow stamps per-turn timings on its root message at each
  // turn boundary and the cumulative agentic totals only on exit; while it runs,
  // show the session's live totals on the root message instead.
  const cLiveLlm = $derived(cAgenticStore.mGetLiveLlmTotals(message.convId));
  const cIsLiveFlowRoot = $derived(
    cLiveLlm !== null && cAgenticStore.mGetFlowRootMessageId(message.convId) === message.id
  );
</script>

{#if showMessageStats && cIsLiveFlowRoot && cLiveLlm}
  <ChatMessageStatistics
    isLive
    mode={ChatMessageStatisticsMode.cGeneration}
    predictedMs={cLiveLlm.predicted_ms}
    predictedTokens={cLiveLlm.predicted_n}
    promptMs={cLiveLlm.prompt_ms}
    promptTokens={cLiveLlm.prompt_n}
  />
{:else if showMessageStats && message.timings && message.timings.predicted_n && message.timings.predicted_ms}
  {@const cAgentic = message.timings.agentic}
  <ChatMessageStatistics
    agenticTimings={cAgentic}
    mode={ChatMessageStatisticsMode.cGeneration}
    predictedMs={cAgentic ? cAgentic.llm.predicted_ms : message.timings.predicted_ms}
    predictedTokens={cAgentic ? cAgentic.llm.predicted_n : message.timings.predicted_n}
    promptMs={cAgentic ? cAgentic.llm.prompt_ms : message.timings.prompt_ms}
    promptTokens={cAgentic ? cAgentic.llm.prompt_n : message.timings.prompt_n}
  />
{:else if isLoading && showMessageStats}
  {@const cLiveStats = processingState.mGetLiveProcessingStats()}
  {@const cGenStats = processingState.mGetLiveGenerationStats()}

  {#if cGenStats}
    <ChatMessageStatistics
      isLive
      mode={ChatMessageStatisticsMode.cGeneration}
      predictedMs={cGenStats.timeMs}
      predictedTokens={cGenStats.tokensGenerated}
      promptMs={cLiveStats?.timeMs}
      promptTokens={cLiveStats?.tokensProcessed}
    />
  {/if}
{/if}
