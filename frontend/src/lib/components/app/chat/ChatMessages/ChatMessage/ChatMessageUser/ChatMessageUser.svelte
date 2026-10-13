<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import {
    ChatMessageActionIcons,
    ChatMessageEditForm,
    ChatMessageStatistics,
    ChatMessageUserBubble
  } from '$lib/components/app/chat';
  import { fGetChatMessageEditContext } from '$lib/contexts';
  import { ChatMessageStatisticsMode, MessageRole } from '$lib/enums';
  import { fUseProcessingState } from '$lib/hooks/use-processing-state.svelte';
  import { cChatStore, cSettingsStore } from '$lib/stores';

  interface Props {
    class?: string;
    message: DatabaseMessage;
    isLastUserMessage?: boolean;
    nextAssistantMessage?: DatabaseMessage | null;
  }

  let {
    class: className = '',
    isLastUserMessage = false,
    message,
    nextAssistantMessage = null
  }: Props = $props();

  // Get contexts
  const cEditCtx = fGetChatMessageEditContext();
  const cProcessingState = fUseProcessingState();

  const cCurrentConfig = $derived(cSettingsStore.vConfig);
  const cIsActivelyProcessing = $derived(isLastUserMessage && cChatStore.isLoading);

  // For agentic turns, prefer the cumulative agentic.llm totals over per-call timings.
  let vStoredReadingStats = $derived.by(() => {
    const cTimings = nextAssistantMessage?.timings;

    if (!cTimings?.prompt_n || !cTimings?.prompt_ms) return null;

    const cAgentic = cTimings.agentic;

    return {
      promptMs: cAgentic ? cAgentic.llm.prompt_ms : cTimings.prompt_ms,
      promptTokens: cAgentic ? cAgentic.llm.prompt_n : cTimings.prompt_n
    };
  });

  let vShowStoredReadingStats = $derived(
    Boolean(cCurrentConfig.showMessageStats) && vStoredReadingStats !== null
  );

  let vShowLiveReadingStats = $derived(
    Boolean(cCurrentConfig.showMessageStats) &&
      cIsActivelyProcessing &&
      vStoredReadingStats === null
  );

  $effect(() => {
    if (vShowLiveReadingStats) {
      cProcessingState.mStartMonitoring();
    }
  });
</script>

<div
  aria-label={fTranslate('text_4c435e0762de')}
  class="chat-message-user group flex flex-col items-end gap-3 md:gap-2 {className}"
  role="group"
>
  {#if cEditCtx.isEditing}
    <ChatMessageEditForm />
  {:else}
    <ChatMessageUserBubble
      attachments={message.extra}
      content={message.content}
      renderMarkdown={true}
    />

    {#if vShowStoredReadingStats}
      <!-- Reading stats sourced from the assistant message that followed this turn -->
      <div class="info my-2 grid w-full justify-items-end gap-4 tabular-nums">
        <div
          class="inline-flex flex-wrap items-start justify-end gap-2 text-xs text-muted-foreground"
        >
          <ChatMessageStatistics
            mode={ChatMessageStatisticsMode.cReading}
            promptMs={vStoredReadingStats!.promptMs}
            promptTokens={vStoredReadingStats!.promptTokens}
          />
        </div>
      </div>
    {:else if vShowLiveReadingStats}
      {@const cLiveStats = cProcessingState.mGetLiveProcessingStats()}
      {#if cLiveStats}
        <div class="info my-2 grid w-full justify-items-end gap-4 tabular-nums">
          <div
            class="inline-flex flex-wrap items-start justify-end gap-2 text-xs text-muted-foreground"
          >
            <ChatMessageStatistics
              isLive
              mode={ChatMessageStatisticsMode.cReading}
              promptMs={cLiveStats.timeMs}
              promptTokens={cLiveStats.tokensProcessed}
            />
          </div>
        </div>
      {/if}
    {/if}

    {#if message.timestamp}
      <div class="max-w-[80%]">
        <ChatMessageActionIcons actionsPosition="right" justify="end" role={MessageRole.cUser} />
      </div>
    {/if}
  {/if}
</div>
