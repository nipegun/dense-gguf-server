<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ConversationSelection } from '$lib/components/app';
  import * as Dialog from '$lib/components/ui/dialog';

  interface Props {
    conversations: DatabaseConversation[];
    messageCountMap?: Map<string, number>;
    mode: 'export' | 'import';
    onCancel: () => void;
    onConfirm: (pSelectedConversations: DatabaseConversation[]) => void;
    open?: boolean;
  }

  let {
    conversations,
    messageCountMap = new Map(),
    mode,
    onCancel,
    onConfirm,
    open = $bindable(false)
  }: Props = $props();

  let vConversationSelectionRef: ConversationSelection | undefined = $state();

  let vPreviousOpen = $state(false);

  $effect(() => {
    if (open && !vPreviousOpen && vConversationSelectionRef) {
      vConversationSelectionRef.reset();
    } else if (!open && vPreviousOpen) {
      onCancel();
    }

    vPreviousOpen = open;
  });
</script>

<Dialog.fRoot bind:open>
  <Dialog.fPortal>
    <Dialog.Overlay class="z-1000000" />

    <Dialog.Content class="z-1000001 max-w-2xl">
      <Dialog.Header>
        <Dialog.Title>
          {fTranslate('text_950fdc1e46c2')}
          {mode === 'export' ? 'Export' : 'Import'}
        </Dialog.Title>

        <Dialog.Description>
          {#if mode === 'export'}{fTranslate('dgs_2ae6f6a0cc95')}{:else}{fTranslate(
              'dgs_22d3a722a92b'
            )}{/if}
        </Dialog.Description>
      </Dialog.Header>

      <ConversationSelection
        bind:this={vConversationSelectionRef}
        {conversations}
        isOpen={open}
        {messageCountMap}
        {mode}
        {onCancel}
        {onConfirm}
      />
    </Dialog.Content>
  </Dialog.fPortal>
</Dialog.fRoot>
