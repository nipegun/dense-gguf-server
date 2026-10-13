<script lang="ts">
  import { page } from '$app/state';
  import { ChatScreen, ChatTabs } from '$lib/components/app';
  import { cNewChatTabId } from '$lib/constants';
  import { cSettingsStore, cTabsStore } from '$lib/stores';

  let { children } = $props();

  // the new-chat screen is the bare `#/` route (no conversation id)
  let vShowCenteredEmpty = $derived(!page.params.id);

  let vShowTabs = $derived(
    Boolean(cSettingsStore.vConfig.conversationTabs) &&
      (page.params.id || cTabsStore.aOpenTabs.some((pId) => pId !== cNewChatTabId))
  );

  $effect(() => {
    const cId = page.params.id ?? (page.route.id === '/(chat)' ? cNewChatTabId : undefined);

    if (cId && cSettingsStore.vConfig.conversationTabs) {
      cTabsStore.mSyncWithRoute(cId);
    }
  });
</script>

<div class={vShowTabs ? 'md:[--chat-tabs-offset:1.25rem]' : ''}>
  {#if vShowTabs}
    <ChatTabs />
  {/if}

  <ChatScreen showCenteredEmpty={vShowCenteredEmpty} />
</div>

{@render children?.()}
