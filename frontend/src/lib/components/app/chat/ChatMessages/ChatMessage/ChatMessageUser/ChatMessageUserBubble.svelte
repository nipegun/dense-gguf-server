<script lang="ts">
  import { ChatAttachmentsList, MarkdownContent, MentionText } from '$lib/components/app';
  import { Card } from '$lib/components/ui/card';
  import { cSettingsStore } from '$lib/stores';
  import type { DatabaseMessageExtra } from '$lib/types/database';

  interface Props {
    content: string;
    attachments?: DatabaseMessageExtra[];
    renderMarkdown?: boolean;
    textColorClass?: string;
    cardBgClass?: string;
    maxHeightStyle?: string;
  }

  let {
    attachments = [],
    cardBgClass = 'dark:bg-primary/15',
    content,
    maxHeightStyle = '',
    renderMarkdown = false,
    textColorClass = 'text-foreground'
  }: Props = $props();

  let vIsMultiline = $state(false);
  let vMessageElement: HTMLElement | undefined = $state();
  const cCurrentConfig = cSettingsStore.vConfig;

  $effect(() => {
    if (!vMessageElement || !content.trim()) return;

    if (content.includes('\n')) {
      vIsMultiline = true;

      return;
    }

    const cResizeObserver = new ResizeObserver((entries) => {
      for (const cEntry of entries) {
        const cElement = cEntry.target as HTMLElement;
        const cEstimatedSingleLineHeight = 24; // Typical line height for text-md

        vIsMultiline = cElement.offsetHeight > cEstimatedSingleLineHeight * 1.5;
      }
    });

    cResizeObserver.observe(vMessageElement);

    return () => {
      cResizeObserver.disconnect();
    };
  });
</script>

{#if attachments && attachments.length > 0}
  <div class="mb-2 max-w-[80%]">
    <ChatAttachmentsList
      {attachments}
      imageHeight="max-h-40"
      imageWidth="w-auto max-w-full"
      readonly
    />
  </div>
{/if}

{#if content.trim()}
  <Card
    class="chat-message-user-bubble max-w-[80%] overflow-y-auto rounded-[1.125rem] border-none bg-primary/5 px-3.75 py-1.5 {textColorClass} backdrop-blur-md data-multiline:py-2.5 {cardBgClass}"
    data-multiline={vIsMultiline ? '' : undefined}
    style="{maxHeightStyle} overflow-wrap: anywhere; word-break: break-word;"
  >
    {#if renderMarkdown && cCurrentConfig.renderUserContentAsMarkdown}
      <div bind:this={vMessageElement}>
        <MarkdownContent class="markdown-user-content" {content} />
      </div>
    {:else}
      <span bind:this={vMessageElement} class="text-md whitespace-pre-wrap"
        ><MentionText {content} /></span
      >
    {/if}
  </Card>
{/if}
