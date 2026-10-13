<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import {
    ChatMessageActionIcons,
    ChatMessageAgenticContent,
    ChatMessageAssistantModel,
    ChatMessageAssistantProcessingInfo,
    ChatMessageAssistantRawOutput,
    ChatMessageAssistantStatistics,
    ChatMessageEditForm
  } from '$lib/components/app';
  import { fGetChatMessageEditContext } from '$lib/contexts';
  import { MessageRole } from '$lib/enums';
  import { fUseProcessingState } from '$lib/hooks/use-processing-state.svelte';
  import { cChatStore, cModelsStore, cServerStore, cSettingsStore } from '$lib/stores';
  import { fModelLoadProgressText } from '$lib/utils';
  import { fHasAgenticContent } from '$lib/utils';

  interface Props {
    class?: string;
    isLastAssistantMessage?: boolean;
    message: DatabaseMessage;
    toolMessages?: DatabaseMessage[];
    onContinue?: () => void;
    onRegenerate: (pModelOverride?: string) => void;
    textareaElement?: HTMLTextAreaElement;
  }

  let {
    class: className = '',
    isLastAssistantMessage = false,
    message,
    onContinue,
    onRegenerate,
    textareaElement = $bindable(),
    toolMessages = []
  }: Props = $props();

  // Get edit context
  const cEditCtx = fGetChatMessageEditContext();

  const cIsAgentic = $derived(fHasAgenticContent(message, toolMessages));
  const cProcessingState = fUseProcessingState();

  let vCurrentConfig = $derived(cSettingsStore.vConfig);
  let vIsRouter = $derived(cServerStore.isRouterMode);

  let vShowRawOutput = $state(false);

  let vDisplayedModel = $derived(message.model ?? null);

  let vIsCurrentlyLoading = $derived(cChatStore.isLoading);
  let vIsStreaming = $derived(cChatStore.mIsStreaming());
  let vHasNoContent = $derived(!message?.content?.trim());
  let vIsActivelyProcessing = $derived(vIsCurrentlyLoading || vIsStreaming);

  // during a router auto-load the message has no model yet: target the model frozen in the
  // persisted stream state (survives a reload), then fall back to the dropdown selection
  let vLoadTargetModel = $derived(
    message.model ?? cChatStore.mGetResumeModel(message.convId) ?? cModelsStore.vSelectedModelName
  );
  let vModelLoadProgress = $derived(
    vIsRouter && vLoadTargetModel ? cModelsStore.status.mGetLoadProgress(vLoadTargetModel) : null
  );
  let vModelLoadingText = $derived(fModelLoadProgressText(vModelLoadProgress));

  let vShowProcessingInfoTop = $derived(
    message?.role === MessageRole.cAssistant &&
      vIsActivelyProcessing &&
      vHasNoContent &&
      !cIsAgentic &&
      isLastAssistantMessage
  );

  let vShowProcessingInfoBottom = $derived(
    message?.role === MessageRole.cAssistant &&
      vIsActivelyProcessing &&
      (!vHasNoContent || cIsAgentic) &&
      isLastAssistantMessage
  );

  let vAssistantEl: HTMLDivElement | undefined = $state();
  let vLastUserMessageHeight = $state(0);
  let vAssistantMarginTop = $state(0);

  // The measured CSS vars feed the :last-child min-height rule only, so only
  // the last assistant message needs them. Reading isLastAssistantMessage
  // here also re-runs the effect when this message stops being the last.
  $effect(() => {
    if (!vAssistantEl || !isLastAssistantMessage) return;

    vAssistantMarginTop = Math.round(parseFloat(getComputedStyle(vAssistantEl).marginTop));

    const cChatMessageEl = vAssistantEl.closest('.chat-message');
    const cPreviousChatMessage = cChatMessageEl?.previousElementSibling;
    const cUserMessageEl = cPreviousChatMessage?.querySelector(
      '.chat-message-user'
    ) as HTMLElement | null;

    if (!cUserMessageEl) {
      vLastUserMessageHeight = 0;

      return;
    }

    const fUpdateHeight = () => {
      const cRect = cUserMessageEl.getBoundingClientRect();
      const cMarginTop = Math.round(parseFloat(getComputedStyle(cUserMessageEl).marginTop));

      vLastUserMessageHeight = Math.round(cRect.height + cMarginTop);
    };

    fUpdateHeight();

    const cResizeObserver = new ResizeObserver(fUpdateHeight);

    cResizeObserver.observe(cUserMessageEl);

    return () => {
      cResizeObserver.disconnect();
    };
  });

  $effect(() => {
    if (vShowProcessingInfoTop || vShowProcessingInfoBottom) {
      cProcessingState.mStartMonitoring();
    }
  });
</script>

<div
  bind:this={vAssistantEl}
  style:--assistant-margin-top={vAssistantMarginTop > 0 ? `${vAssistantMarginTop}px` : undefined}
  style:--last-user-message-height={vLastUserMessageHeight > 0
    ? `${vLastUserMessageHeight}px`
    : undefined}
  aria-label={fTranslate('text_d9495b112c58')}
  class="chat-message-assistant text-md group w-full leading-7.5 {className}"
  role="group"
>
  {#if vShowProcessingInfoTop}
    <ChatMessageAssistantProcessingInfo
      modelLoadingText={vModelLoadingText}
      position="top"
      processingState={cProcessingState}
    />
  {/if}

  {#if cEditCtx.isEditing}
    <ChatMessageEditForm />
  {:else}
    {#if vShowRawOutput}
      <ChatMessageAssistantRawOutput {message} {toolMessages} />
    {:else}
      <ChatMessageAgenticContent
        {isLastAssistantMessage}
        isStreaming={cChatStore.mIsStreaming()}
        {message}
        {toolMessages}
      />
    {/if}
  {/if}

  {#if vShowProcessingInfoBottom}
    <ChatMessageAssistantProcessingInfo
      modelLoadingText={vModelLoadingText}
      position="bottom"
      processingState={cProcessingState}
    />
  {/if}

  {#if vDisplayedModel}
    <div class="info my-6 grid gap-4 tabular-nums">
      <div class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground">
        <ChatMessageAssistantModel
          displayedModel={vDisplayedModel}
          isLoading={cChatStore.isLoading}
          isRouter={vIsRouter}
          {onRegenerate}
        />

        <ChatMessageAssistantStatistics
          isLoading={cChatStore.isLoading}
          {message}
          processingState={cProcessingState}
          showMessageStats={vCurrentConfig.showMessageStats}
        />
      </div>
    </div>
  {/if}

  {#if message.timestamp && !cEditCtx.isEditing}
    <ChatMessageActionIcons
      actionsPosition="left"
      justify="start"
      onContinue={vCurrentConfig.enableContinueGeneration ? onContinue : undefined}
      onRawOutputToggle={(pEnabled) => (vShowRawOutput = pEnabled)}
      {onRegenerate}
      rawOutputEnabled={vShowRawOutput}
      role={MessageRole.cAssistant}
      showRawOutputSwitch={vCurrentConfig.showRawOutputSwitch}
    />
  {/if}
</div>

<style>
  :global(.chat-message):last-child .chat-message-assistant {
    --assistant-min-height-offset: calc(
      var(--last-user-message-height, 19rem) + var(--chat-form-height, 6rem) +
        var(--chat-form-bottom-position, 0.5rem) + var(--chat-form-padding-top, 6rem) +
        var(--assistant-margin-top, 3rem) + var(--chat-tabs-offset, 0px)
    );
    min-height: calc(100dvh - var(--assistant-min-height-offset));

    @media (width > 768px) {
      --assistant-min-height-offset: calc(
        var(--last-user-message-height, 18rem) + var(--chat-form-height, 6rem) +
          var(--chat-form-bottom-position, 1rem) + var(--chat-form-padding-top, 6rem) +
          var(--assistant-margin-top, 3rem) + var(--chat-tabs-offset, 0px)
      );
    }
  }
</style>
