<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Eject, Loader2, MessageSquare, Power, SquarePen } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import { cConversationsStore, cDeviceStore } from '$lib/stores';

  interface Props {
    /** A provider that cannot load or unload has no button to offer. */
    canToggleLoad?: boolean;
    isLoaded: boolean;
    isLoading: boolean;
    onToggleLoad: () => void;
    onUseInChat: () => void;
    onUseInNewChat: () => void;
    /** The header gives the actions room to breathe, the phone bar keeps them small. */
    size?: 'sm' | 'default';
  }

  let {
    canToggleLoad = true,
    isLoaded,
    isLoading,
    onToggleLoad,
    onUseInChat,
    onUseInNewChat,
    size = 'sm'
  }: Props = $props();

  // a chat with anything in it takes the model, a new chat is not what its user asked for
  let vHasChat = $derived(
    cConversationsStore.vActiveConversation !== null ||
      (cConversationsStore.aActiveMessages as DatabaseMessage[]).length > 0
  );
  let vIsMobile = $derived(cDeviceStore.isMobile);
</script>

<div class="my-3 flex flex-col gap-2">
  {#if vIsMobile}
    <!-- a phone picks its model in the manager, so pointing this chat at it leads -->
    <Button class="w-full gap-1.5" onclick={vHasChat ? onUseInChat : onUseInNewChat} {size}>
      {#if vHasChat}
        <MessageSquare class="h-3.5 w-3.5" />

        {fTranslate('dgs_d41d9731a66c')}{:else}
        <SquarePen class="h-3.5 w-3.5" />

        {fTranslate('dgs_61025e4f8790')}{/if}
    </Button>
  {/if}

  <div class="flex gap-3 mt-2">
    {#if !vIsMobile}
      <Button class="flex-1 gap-1.5" onclick={onUseInNewChat} {size}>
        <SquarePen class="h-3.5 w-3.5" />

        {fTranslate('dgs_61025e4f8790')}
      </Button>
    {/if}

    {#if canToggleLoad}
      <Button
        class="flex-1 gap-1.5"
        disabled={isLoading}
        onclick={onToggleLoad}
        {size}
        variant="secondary"
      >
        {#if isLoading}
          <Loader2 class="h-3.5 w-3.5 animate-spin" />

          {fTranslate('dgs_47d2a515ef2f')}{:else if isLoaded}
          <Eject class="h-3.5 w-3.5" />

          {fTranslate('text_c2ed103ced3e')}{:else}
          <Power class="h-3.5 w-3.5" />

          {fTranslate('text_889fad74ece5')}{/if}
      </Button>
    {/if}
  </div>
</div>
