<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { SkipForward, Square } from '@lucide/svelte';
  import { page } from '$app/state';
  import {
    ChatFormActionReasoning,
    ChatFormActionRecord,
    ChatFormActionsAdd,
    ChatFormActionSubmit,
    ChatFormContextGauge,
    ModelsSelector
  } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import { cIconClassDefault } from '$lib/constants';
  import { fSetChatFormActionsContext } from '$lib/contexts';
  import { FileTypeCategory, MessageRole } from '$lib/enums';
  import { fUseChatFormModel } from '$lib/hooks/use-chat-form-model.svelte';
  import { ChatService } from '$lib/services';
  import { cChatStore, cConversationsStore, cSettingsStore } from '$lib/stores';
  import { fGetFileTypeCategory } from '$lib/utils';

  interface Props {
    canSend?: boolean;
    canSubmit?: boolean;
    class?: string;
    disabled?: boolean;
    isLoading?: boolean;
    isReasoning?: boolean;
    isRecording?: boolean;
    showAddButton?: boolean;
    showModelSelector?: boolean;
    uploadedFiles?: ChatUploadedFile[];
    onFileUpload?: () => void;
    onMicClick?: () => void;
    onStop?: () => void;
    onSystemPromptClick?: () => void;
  }

  let {
    canSend = false,
    canSubmit = false,
    class: className = '',
    disabled = false,
    isLoading = false,
    isReasoning = false,
    isRecording = false,
    onFileUpload,
    onMicClick,
    onStop,
    onSystemPromptClick,
    showAddButton = true,
    showModelSelector = true,
    uploadedFiles = []
  }: Props = $props();

  let vCurrentConfig = $derived(cSettingsStore.vConfig);

  const cFormModel = fUseChatFormModel();

  let vHasAudioModality = $derived(cFormModel.hasAudioModality);
  let vHasVideoModality = $derived(cFormModel.hasVideoModality);
  let vHasVisionModality = $derived(cFormModel.hasVisionModality);
  let vHasModelSelected = $derived(cFormModel.hasModelSelected);
  let vIsSelectedModelInCache = $derived(cFormModel.isSelectedModelInCache);
  let vSubmitTooltip = $derived.by(() => {
    if (!vHasModelSelected) return 'Please select a model first';

    if (!vIsSelectedModelInCache) return 'Selected model is not available, please select another';

    return '';
  });

  let vHasAudioAttachments = $derived(
    uploadedFiles.some((pFile) => fGetFileTypeCategory(pFile.type) === FileTypeCategory.AUDIO)
  );
  let vShouldShowRecordButton = $derived(
    vHasAudioModality && !canSubmit && !vHasAudioAttachments && vCurrentConfig.autoMicOnEmpty
  );

  let vSelectorModelRef: ModelsSelector | undefined = $state(undefined);

  export function openModelSelector() {
    vSelectorModelRef?.open();
  }
  // the streaming assistant message carries both the completion id and the model that
  // produced it, targeting reasoning control from the same source keeps them consistent
  let vActiveMessage = $derived(
    cConversationsStore.aActiveMessages[cConversationsStore.aActiveMessages.length - 1]
  );

  let vHasProcessedTokens = $derived.by(() => {
    if (!page.params.id) return false;

    const aMessages = cConversationsStore.aActiveMessages as DatabaseMessage[];

    let vTotalHistoricalTokens = 0;

    for (const cM of aMessages) {
      if (cM.role !== MessageRole.cAssistant) continue;

      const cTimings = cM.timings;

      if (!cTimings) continue;

      const cAgenticLlm = cTimings.agentic?.llm;

      if (cAgenticLlm?.prompt_n != null || cAgenticLlm?.predicted_n != null) {
        vTotalHistoricalTokens += (cAgenticLlm?.prompt_n ?? 0) + (cAgenticLlm?.predicted_n ?? 0);
      } else {
        vTotalHistoricalTokens += (cTimings.prompt_n ?? 0) + (cTimings.predicted_n ?? 0);
      }
    }

    if (vTotalHistoricalTokens > 0) return true;

    if (!cChatStore.isLoading && !cChatStore.mIsStreaming()) return false;

    const cProcessingState = cChatStore.processing.vActiveState;

    if (!cProcessingState) return false;

    const cLivePromptTokens = Math.max(
      cProcessingState.promptTokens ?? 0,
      cProcessingState.promptProgress?.processed ?? 0
    );
    const cLiveOutputTokens = cProcessingState.outputTokensUsed ?? 0;

    return cLivePromptTokens > 0 || cLiveOutputTokens > 0;
  });

  fSetChatFormActionsContext({
    get disabled() {
      return disabled;
    },
    get hasAudioModality() {
      return vHasAudioModality;
    },
    get hasVideoModality() {
      return vHasVideoModality;
    },
    get hasVisionModality() {
      return vHasVisionModality;
    },
    get onFileUpload() {
      return onFileUpload;
    },
    get onSystemPromptClick() {
      return onSystemPromptClick;
    }
  });
</script>

<div
  class="flex w-full items-center gap-3 {className} {showAddButton ? '' : 'justify-end'}"
  style="container-type: inline-size"
>
  {#if showAddButton}
    <div class="mr-auto flex items-center gap-2">
      <ChatFormActionsAdd />
    </div>
  {/if}

  <div class="flex items-center gap-1.5">
    {#if vHasProcessedTokens}
      <ChatFormContextGauge />
    {/if}

    {#if showModelSelector}
      <ChatFormActionReasoning />

      <ModelsSelector
        bind:this={vSelectorModelRef}
        {disabled}
        forceForegroundText
        useGlobalSelection
      />
    {/if}
  </div>

  {#if isReasoning}
    <Button
      class="group h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9"
      onclick={() =>
        ChatService.mStopReasoning(vActiveMessage?.completionId ?? '', vActiveMessage?.model)}
      title={fTranslate('text_43ec555ebd2b')}
      type="button"
      variant="secondary"
    >
      <span class="sr-only">{fTranslate('text_43ec555ebd2b')}</span>

      <SkipForward
        class="{cIconClassDefault} stroke-muted-foreground group-hover:stroke-foreground"
      />
    </Button>
  {/if}

  {#if isLoading && !canSubmit}
    <Button
      class="group h-8 w-8 rounded-full p-0 max-md:h-9 max-md:w-9 hover:bg-destructive/10!"
      onclick={onStop}
      type="button"
      variant="secondary"
    >
      <span class="sr-only">{fTranslate('text_cae7d57bc067')}</span>

      <Square
        class="h-8 w-8 fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive hover:fill-destructive hover:stroke-destructive"
      />
    </Button>
  {:else if vShouldShowRecordButton}
    <ChatFormActionRecord
      {disabled}
      hasAudioModality={vHasAudioModality}
      {isLoading}
      {isRecording}
      {onMicClick}
    />
  {:else}
    <ChatFormActionSubmit
      canSend={canSend && (showModelSelector ? vHasModelSelected && vIsSelectedModelInCache : true)}
      {disabled}
      showErrorState={showModelSelector && vHasModelSelected && !vIsSelectedModelInCache}
      tooltipLabel={vSubmitTooltip}
    />
  {/if}
</div>
