<script lang="ts">
  import { fIsLocalShellTool } from './ChatMessage/ChatMessageToolCall/parsers/local-shell-run';
  import ChatMessageToolCallBlock from './ChatMessage/ChatMessageToolCall/ChatMessageToolCallBlock.svelte';
  import ChatMessageReasoningBlock from './ChatMessageReasoningBlock.svelte';
  import {
    ChatMessageActionCardContinueRequest,
    ChatMessageActionCardPermissionRequest,
    ChatMessageStatistics,
    MarkdownContent
  } from '$lib/components/app';
  import { AgenticSectionType, ChatMessageStatsView, ToolPermissionDecision } from '$lib/enums';
  import { cAgenticStore, cSettingsStore } from '$lib/stores';
  import type {
    AgenticSection,
    ChatMessageAgenticTimings,
    ChatMessageAgenticTurnStats,
    DatabaseMessage
  } from '$lib/types';
  import { fDeriveAgenticSections } from '$lib/utils';

  interface Props {
    message: DatabaseMessage;
    toolMessages?: DatabaseMessage[];
    isStreaming?: boolean;
    isLastAssistantMessage?: boolean;
  }

  let {
    isLastAssistantMessage = false,
    isStreaming = false,
    message,
    toolMessages = []
  }: Props = $props();

  let dExpandedStates: Record<number, boolean> = $state({});

  const cShowThoughtInProgress = $derived(Boolean(cSettingsStore.vConfig.showThoughtInProgress));
  const cAlwaysShowToolCallContent = $derived(
    Boolean(cSettingsStore.vConfig.alwaysShowToolCallContent)
  );
  const cShowMessageStats = $derived(Boolean(cSettingsStore.vConfig.showMessageStats));
  const cShowAgenticTurnStats = $derived(
    cShowMessageStats && Boolean(cSettingsStore.vConfig.showAgenticTurnStats)
  );

  const cHasReasoningError = $derived(
    isLastAssistantMessage ? !!cAgenticStore.mGetLastError(message.convId) : false
  );

  const cPendingPermission = $derived(
    isStreaming && isLastAssistantMessage
      ? cAgenticStore.mGetPendingPermissionRequest(message.convId)
      : null
  );

  // dismissal applies to the request object, so the next request ( new
  // identity ) shows the card again without any reset bookkeeping
  let vDismissedPermission: typeof cPendingPermission = $state(null);

  const cVisiblePermission = $derived(
    cPendingPermission && vDismissedPermission !== cPendingPermission ? cPendingPermission : null
  );

  function fHandlePermission(pDecision: ToolPermissionDecision) {
    vDismissedPermission = cPendingPermission;
    cAgenticStore.mResolvePermission(message.convId, pDecision);
  }

  const cPendingContinue = $derived(
    isStreaming && isLastAssistantMessage
      ? cAgenticStore.mGetPendingContinueRequest(message.convId)
      : false
  );

  let vContinueDismissed = $state(false);

  // the continue request is a plain boolean, so there is no identity to
  // compare against; clear the dismissal whenever no request is pending so
  // the next one starts from a clean state
  $effect(() => {
    if (!cPendingContinue) {
      vContinueDismissed = false;
    }
  });

  const cShowContinue = $derived(Boolean(cPendingContinue) && !vContinueDismissed);

  function fHandleContinue(pShouldContinue: boolean) {
    vContinueDismissed = true;
    cAgenticStore.mResolveContinue(message.convId, pShouldContinue);
  }

  const aSections = $derived(fDeriveAgenticSections(message, toolMessages, [], isStreaming));

  const cCurrentlyExecutingToolCallId = $derived(
    isStreaming ? cAgenticStore.mGetExecutingToolCallId(message.convId) : null
  );

  type TurnGroup = {
    sections: AgenticSection[];
    flatIndices: number[];
  };

  const aTurnGroups: TurnGroup[] = $derived.by(() => {
    const aGroups: TurnGroup[] = [];

    let aCurrentTurn: AgenticSection[] = [];
    let aCurrentIndices: number[] = [];
    let vPrevWasTool = false;

    for (let vI = 0; vI < aSections.length; vI++) {
      const cSection = aSections[vI];
      const cIsTool =
        cSection.type === AgenticSectionType.TOOL_CALL ||
        cSection.type === AgenticSectionType.cToolCallPending ||
        cSection.type === AgenticSectionType.cToolCallStreaming;

      if (!cIsTool && vPrevWasTool && aCurrentTurn.length > 0) {
        aGroups.push({ flatIndices: aCurrentIndices, sections: aCurrentTurn });
        aCurrentTurn = [];
        aCurrentIndices = [];
      }

      aCurrentTurn.push(cSection);
      aCurrentIndices.push(vI);
      vPrevWasTool = cIsTool;
    }

    if (aCurrentTurn.length > 0) {
      aGroups.push({ flatIndices: aCurrentIndices, sections: aCurrentTurn });
    }

    return aGroups;
  });

  function fGetDefaultExpanded(pSection: AgenticSection): boolean {
    if (fIsLocalShellTool(pSection.toolName) && cSettingsStore.vConfig.expandLocalShellCommands)
      return true;
    if (
      pSection.type === AgenticSectionType.TOOL_CALL ||
      pSection.type === AgenticSectionType.cToolCallPending ||
      pSection.type === AgenticSectionType.cToolCallStreaming
    ) {
      return cAlwaysShowToolCallContent;
    }

    if (pSection.type === AgenticSectionType.cReasoningPending) {
      return cShowThoughtInProgress;
    }

    return false;
  }

  function fIsExpanded(pIndex: number, pSection: AgenticSection): boolean {
    if (dExpandedStates[pIndex] !== undefined) {
      return dExpandedStates[pIndex];
    }

    return fGetDefaultExpanded(pSection);
  }

  function fToggleExpanded(pIndex: number, pSection: AgenticSection) {
    const cCurrentState = fIsExpanded(pIndex, pSection);

    dExpandedStates[pIndex] = !cCurrentState;
  }

  function fBuildTurnAgenticTimings(
    pStats: ChatMessageAgenticTurnStats
  ): ChatMessageAgenticTimings {
    return {
      llm: pStats.llm,
      toolCalls: pStats.toolCalls,
      toolCallsCount: pStats.toolCalls.length,
      toolsMs: pStats.toolsMs,
      turns: 1
    };
  }
</script>

{#snippet renderSection(pSection: AgenticSection, pIndex: number)}
  {#if pSection.type === AgenticSectionType.TEXT}
    <div class="agentic-text">
      <MarkdownContent attachments={message?.extra} content={pSection.content} />
    </div>
  {:else if pSection.type === AgenticSectionType.REASONING || pSection.type === AgenticSectionType.cReasoningPending}
    <ChatMessageReasoningBlock
      attachments={message?.extra}
      hasReasoningError={cHasReasoningError}
      {isStreaming}
      onToggle={() => fToggleExpanded(pIndex, pSection)}
      open={fIsExpanded(pIndex, pSection)}
      section={pSection}
    />
  {:else if pSection.type === AgenticSectionType.TOOL_CALL || pSection.type === AgenticSectionType.cToolCallPending || pSection.type === AgenticSectionType.cToolCallStreaming}
    <ChatMessageToolCallBlock
      attachments={pSection.toolResultExtras}
      isExecuting={pSection.toolCallId !== undefined &&
        pSection.toolCallId === cCurrentlyExecutingToolCallId}
      {isStreaming}
      onToggle={() => fToggleExpanded(pIndex, pSection)}
      open={fIsExpanded(pIndex, pSection)}
      section={pSection}
    />
  {/if}
{/snippet}

<div class="agentic-content gap-2">
  {#if aTurnGroups.length > 1}
    {#each aTurnGroups as vTurn, vTurnIndex (vTurnIndex)}
      {@const cTurnStats = message?.timings?.agentic?.perTurn?.[vTurnIndex]}

      <div class="agentic-turn group/turn grid gap-2">
        {#each vTurn.sections as vSection, sIdx (vTurn.flatIndices[sIdx])}
          {@render renderSection(vSection, vTurn.flatIndices[sIdx])}
        {/each}

        {#if cTurnStats && cShowAgenticTurnStats}
          <div class="turn-stats transition-opacity duration-150 mt-1 mb-4">
            <ChatMessageStatistics
              agenticTimings={cTurnStats.toolCalls.length > 0
                ? fBuildTurnAgenticTimings(cTurnStats)
                : undefined}
              hideSummary
              initialView={ChatMessageStatsView.cGeneration}
              predictedMs={cTurnStats.llm.predicted_ms}
              predictedTokens={cTurnStats.llm.predicted_n}
              promptMs={cTurnStats.llm.prompt_ms}
              promptTokens={cTurnStats.llm.prompt_n}
            />
          </div>
        {/if}
      </div>
    {/each}
  {:else}
    {#each aSections as vSection, vIndex (vIndex)}
      {@render renderSection(vSection, vIndex)}
    {/each}
  {/if}

  {#if cVisiblePermission}
    <ChatMessageActionCardPermissionRequest
      onDecision={fHandlePermission}
      serverLabel={cVisiblePermission.serverLabel}
      toolName={cVisiblePermission.toolName}
    />
  {/if}

  {#if cShowContinue}
    <ChatMessageActionCardContinueRequest onDecision={fHandleContinue} />
  {/if}
</div>

<style>
  .agentic-content {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 48rem;
  }

  .agentic-content > :global(*),
  .agentic-turn > :global(*) {
    min-width: 0;
  }

  .agentic-text {
    width: 100%;
  }

  .turn-stats {
    border-top: 1px solid hsl(var(--muted) / 0.5);
  }
</style>
