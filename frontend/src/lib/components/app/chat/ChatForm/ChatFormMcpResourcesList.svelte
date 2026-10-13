<script lang="ts">
  import { ChatAttachmentsListItemMcpResource, ScrollCarousel } from '$lib/components/app';
  import { ScrollCarouselVariant } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';

  interface Props {
    class?: string;
    onResourceClick?: (pUri: string) => void;
  }

  let { class: className, onResourceClick }: Props = $props();

  const attachments = $derived(cMcpStore.resources.attachments);
  const cHasAttachments = $derived(cMcpStore.resources.hasAttachments);

  function fHandleRemove(pAttachmentId: string) {
    cMcpStore.mRemoveResourceAttachment(pAttachmentId);
  }

  function fHandleResourceClick(pUri: string) {
    onResourceClick?.(pUri);
  }
</script>

{#if cHasAttachments}
  <div class={className}>
    <ScrollCarousel gapSize="2" variant={ScrollCarouselVariant.CENTER}>
      {#each attachments as vAttachment, vI (vAttachment.id)}
        <ChatAttachmentsListItemMcpResource
          attachment={vAttachment}
          class={vI === 0 ? 'ml-3' : ''}
          onRemove={fHandleRemove}
          onclick={() => fHandleResourceClick(vAttachment.resource.uri)}
        />
      {/each}
    </ScrollCarousel>
  </div>
{/if}
