<script lang="ts">
  import { BookOpenText, Clock, Gauge, Layers, Sparkles, WholeWord, Wrench } from '@lucide/svelte';
  import { ChatMessageStatisticsBadge } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cDefaultPerformanceTime, cMsPerSecond } from '$lib/constants';
  import { ChatMessageStatisticsMode, ChatMessageStatsView } from '$lib/enums';
  import type { ChatMessageAgenticTimings } from '$lib/types/chat';
  import { fFormatPerformanceTime } from '$lib/utils';
  import type { Component } from 'svelte';

  interface Props {
    predictedTokens?: number;
    predictedMs?: number;
    promptTokens?: number;
    promptMs?: number;
    isLive?: boolean;
    isProcessingPrompt?: boolean;
    initialView?: ChatMessageStatsView;
    agenticTimings?: ChatMessageAgenticTimings;
    onActiveViewChange?: (pView: ChatMessageStatsView) => void;
    hideSummary?: boolean;
    mode?: ChatMessageStatisticsMode;
  }

  let {
    agenticTimings,
    hideSummary = false,
    initialView = ChatMessageStatsView.cGeneration,
    isLive = false,
    isProcessingPrompt = false,
    mode = ChatMessageStatisticsMode.cSwitchable,
    onActiveViewChange,
    predictedMs,
    predictedTokens,
    promptMs,
    promptTokens
  }: Props = $props();

  let vIsSwitchable = $derived(mode === ChatMessageStatisticsMode.cSwitchable);

  let vActiveView: ChatMessageStatsView = $derived(
    mode === ChatMessageStatisticsMode.cReading
      ? ChatMessageStatsView.cReading
      : mode === ChatMessageStatisticsMode.cGeneration
        ? ChatMessageStatsView.cGeneration
        : initialView
  );
  let vHasAutoSwitchedToGeneration = $state(false);

  $effect(() => {
    if (vIsSwitchable) {
      onActiveViewChange?.(vActiveView);
    }
  });

  // In live mode: auto-switch to GENERATION tab when prompt processing completes
  $effect(() => {
    if (isLive && vIsSwitchable) {
      // Auto-switch to generation tab only when prompt processing is done (once)
      if (
        !vHasAutoSwitchedToGeneration &&
        !isProcessingPrompt &&
        predictedTokens &&
        predictedTokens > 0
      ) {
        vActiveView = ChatMessageStatsView.cGeneration;
        vHasAutoSwitchedToGeneration = true;
      } else if (!vHasAutoSwitchedToGeneration) {
        // Stay on READING while prompt is still being processed
        vActiveView = ChatMessageStatsView.cReading;
      }
    }
  });

  let vHasGenerationStats = $derived(
    predictedTokens !== undefined &&
      predictedTokens > 0 &&
      predictedMs !== undefined &&
      predictedMs > 0
  );

  let vTokensPerSecond = $derived(
    vHasGenerationStats ? (predictedTokens! / predictedMs!) * cMsPerSecond : 0
  );
  let vFormattedTime = $derived(
    predictedMs !== undefined ? fFormatPerformanceTime(predictedMs) : cDefaultPerformanceTime
  );

  let vPromptTokensPerSecond = $derived(
    promptTokens !== undefined && promptMs !== undefined && promptMs > 0
      ? (promptTokens / promptMs) * cMsPerSecond
      : undefined
  );

  let vFormattedPromptTime = $derived(
    promptMs !== undefined ? fFormatPerformanceTime(promptMs) : undefined
  );

  let vHasPromptStats = $derived(
    promptTokens !== undefined &&
      promptMs !== undefined &&
      vPromptTokensPerSecond !== undefined &&
      vFormattedPromptTime !== undefined
  );

  let vIsGenerationDisabled = $derived(isLive && vIsSwitchable && !vHasGenerationStats);

  let vHasAgenticStats = $derived(
    agenticTimings !== undefined && agenticTimings.toolCallsCount > 0
  );

  let vAgenticToolsPerSecond = $derived(
    vHasAgenticStats && agenticTimings!.toolsMs > 0
      ? (agenticTimings!.toolCallsCount / agenticTimings!.toolsMs) * cMsPerSecond
      : 0
  );

  let vFormattedAgenticToolsTime = $derived(
    vHasAgenticStats ? fFormatPerformanceTime(agenticTimings!.toolsMs) : cDefaultPerformanceTime
  );

  let vAgenticTotalTimeMs = $derived(
    vHasAgenticStats
      ? agenticTimings!.toolsMs + agenticTimings!.llm.predicted_ms + agenticTimings!.llm.prompt_ms
      : 0
  );

  let vFormattedAgenticTotalTime = $derived(fFormatPerformanceTime(vAgenticTotalTimeMs));

  const cFixedFormatter = new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2
  });
</script>

{#snippet viewButton(pOpts: {
  view: ChatMessageStatsView;
  icon: Component;
  label: string;
  tooltipText: string;
  disabled?: boolean;
})}
  {@const IconComponent = pOpts.icon}
  <Tooltip.fRoot>
    <Tooltip.Trigger>
      <!-- prevent another nested button element -->
      {#snippet child({ props: dProps })}
        <button
          {...dProps}
          class="inline-flex h-5 w-5 items-center justify-center rounded-sm transition-colors {vActiveView ===
          pOpts.view
            ? 'bg-background text-foreground shadow-sm'
            : pOpts.disabled
              ? 'cursor-not-allowed opacity-40'
              : 'hover:text-foreground'}"
          disabled={pOpts.disabled}
          onclick={() => !pOpts.disabled && (vActiveView = pOpts.view)}
          type="button"
        >
          <IconComponent class="h-3 w-3" />

          <span class="sr-only">{pOpts.label}</span>
        </button>
      {/snippet}
    </Tooltip.Trigger>

    <Tooltip.Content>
      <p>{pOpts.tooltipText}</p>
    </Tooltip.Content>
  </Tooltip.fRoot>
{/snippet}

<div class="inline-flex items-center text-xs text-muted-foreground">
  {#if vIsSwitchable}
    <div class="inline-flex items-center rounded-sm bg-muted-foreground/15 p-0.5">
      {#if vHasPromptStats || isLive}
        {@render viewButton({
          icon: BookOpenText,
          label: 'Reading',
          tooltipText: 'Processing',
          view: ChatMessageStatsView.cReading
        })}
      {/if}

      {@render viewButton({
        disabled: vIsGenerationDisabled,
        icon: Sparkles,
        label: 'Generation',
        tooltipText: vIsGenerationDisabled ? 'Waiting for tokens...' : 'Generation',
        view: ChatMessageStatsView.cGeneration
      })}

      {#if vHasAgenticStats}
        {@render viewButton({
          icon: Wrench,
          label: 'Tools',
          tooltipText: 'Tool calls',
          view: ChatMessageStatsView.cTools
        })}

        {#if !hideSummary}
          {@render viewButton({
            icon: Layers,
            label: 'Summary',
            tooltipText: 'Agentic summary',
            view: ChatMessageStatsView.cSummary
          })}
        {/if}
      {/if}
    </div>
  {/if}

  <div class="flex items-center gap-1 max-md:pt-1 md:px-1">
    {#if vActiveView === ChatMessageStatsView.cGeneration && vHasGenerationStats}
      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={WholeWord}
        tooltipLabel="Generated tokens"
        value="{predictedTokens?.toLocaleString()} tokens"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Clock}
        tooltipLabel="Generation time"
        value={vFormattedTime}
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Gauge}
        tooltipLabel="Generation speed"
        value="{cFixedFormatter.format(vTokensPerSecond)} t/s"
      />
    {:else if vActiveView === ChatMessageStatsView.cTools && vHasAgenticStats}
      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Wrench}
        tooltipLabel="Tool calls executed"
        value="{agenticTimings!.toolCallsCount.toLocaleString()} calls"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Clock}
        tooltipLabel="Tool execution time"
        value={vFormattedAgenticToolsTime}
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Gauge}
        tooltipLabel="Tool execution rate"
        value="{cFixedFormatter.format(vAgenticToolsPerSecond)} calls/s"
      />
    {:else if vActiveView === ChatMessageStatsView.cSummary && vHasAgenticStats}
      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Layers}
        tooltipLabel="Agentic turns (LLM calls)"
        value="{agenticTimings!.turns.toLocaleString()} turns"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={WholeWord}
        tooltipLabel="Total tokens generated"
        value="{agenticTimings!.llm.predicted_n.toLocaleString()} tokens"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Clock}
        tooltipLabel="Total time (LLM + tools)"
        value={vFormattedAgenticTotalTime}
      />
    {:else if vHasPromptStats && (mode === ChatMessageStatisticsMode.cReading || vIsSwitchable)}
      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={WholeWord}
        tooltipLabel="Prompt tokens"
        value="{promptTokens?.toLocaleString()} tokens"
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Clock}
        tooltipLabel="Prompt processing time"
        value={vFormattedPromptTime ?? '0s'}
      />

      <ChatMessageStatisticsBadge
        class="bg-transparent"
        icon={Gauge}
        tooltipLabel="Prompt processing speed"
        value="{cFixedFormatter.format(vPromptTokensPerSecond!)} tokens/s"
      />
    {/if}
  </div>
</div>
