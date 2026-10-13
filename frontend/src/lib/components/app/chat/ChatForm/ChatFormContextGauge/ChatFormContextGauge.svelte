<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ContextGaugeDial from './ContextGaugeDial.svelte';
  import {
    fGaugeTriggerClick,
    fGaugeTriggerEnter,
    fGaugeTriggerKeydown,
    fGaugeTriggerLeave,
    fGaugeTriggerPointerDown
  } from './gauge-popup.svelte';
  import { fUseContextGauge } from '$lib/hooks/use-context-gauge.svelte';
  import { cChatStore, cConversationsStore, cDeviceStore } from '$lib/stores';
  import { untrack } from 'svelte';

  const cGauge = fUseContextGauge();

  $effect(() => {
    const cConv = cConversationsStore.vActiveConversation;

    untrack(() => cChatStore.processing.mSetActiveConversation(cConv?.id ?? null));
  });

  $effect(() => {
    const cConv = cConversationsStore.vActiveConversation;
    const aMessages = cConversationsStore.aActiveMessages as DatabaseMessage[];

    if (!cConv) return;

    if (cChatStore.isLoading || cChatStore.mIsStreaming()) return;

    if (aMessages.length === 0) {
      untrack(() => cChatStore.processing.mSetState(cConv.id, null));

      return;
    }

    untrack(() => cChatStore.processing.mRestoreFromMessages(aMessages, cConv.id));
  });

  $effect(() => {
    cGauge.mStartMonitoring();
  });
</script>

<div
  aria-label={fTranslate('text_36416faf4b8a')}
  class="flex h-5 w-5 cursor-default items-center justify-center max-md:h-6 max-md:w-6"
  data-context-gauge-trigger
  onclick={fGaugeTriggerClick}
  onkeydown={fGaugeTriggerKeydown}
  onpointerdown={fGaugeTriggerPointerDown}
  onpointerenter={fGaugeTriggerEnter}
  onpointerleave={fGaugeTriggerLeave}
  role="button"
  tabindex="0"
>
  <!-- a phone gets the larger dial: the trigger is a touch target there -->
  <ContextGaugeDial
    level={cGauge.colorLevel}
    percent={cGauge.contextPercent}
    size={cDeviceStore.isMobile ? 'md' : 'sm'}
  />
</div>
