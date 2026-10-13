<script lang="ts">
  import {
    ChatAttachmentsListItemMcpPrompt,
    ChatAttachmentsListItemMcpResource,
    ChatAttachmentsListItemThumbnailFile,
    ChatAttachmentsListItemThumbnailImage
  } from '$lib/components/app';
  import { AttachmentType } from '$lib/enums';
  import type {
    ChatAttachmentDisplayItem,
    DatabaseMessageExtraMcpPrompt,
    DatabaseMessageExtraMcpResource,
    MCPResourceAttachment
  } from '$lib/types';
  import { fIsMcpPrompt, fIsMcpResource, fIsPdfFile } from '$lib/utils';

  interface Props {
    class?: string;
    imageClass?: string;
    imageHeight?: string;
    imageWidth?: string;
    item: ChatAttachmentDisplayItem;
    limitToSingleRow?: boolean;
    onFileRemove?: (pFileId: string) => void;
    onMcpResourcePreview?: (pExtra: DatabaseMessageExtraMcpResource) => void;
    onPreview?: (pItem: ChatAttachmentDisplayItem) => void;
    readonly?: boolean;
  }

  let {
    class: className = '',
    imageClass = '',
    imageHeight = 'h-24',
    imageWidth = 'w-auto',
    item,
    limitToSingleRow = false,
    onFileRemove,
    onMcpResourcePreview,
    onPreview,
    readonly = false
  }: Props = $props();

  const cScrollClasses = $derived(limitToSingleRow ? 'first:ml-4 last:mr-4' : '');
  // Carousel items must keep their width; wrapped attachments (message bubbles)
  // shrink so wide images fit the bubble instead of overflowing it
  const cLayoutClasses = $derived(limitToSingleRow ? 'flex-shrink-0' : 'min-w-0');

  function fToMcpResourceAttachment(
    pExtra: DatabaseMessageExtraMcpResource,
    pId: string
  ): MCPResourceAttachment {
    return {
      id: pId,
      resource: {
        name: pExtra.name,
        serverName: pExtra.serverName,
        title: pExtra.name,
        uri: pExtra.uri
      }
    };
  }
</script>

{#if fIsMcpPrompt(item)}
  {@const mcpPrompt =
    item.attachment?.type === AttachmentType.MCP_PROMPT
      ? (item.attachment as DatabaseMessageExtraMcpPrompt)
      : item.uploadedFile?.mcpPrompt
        ? {
            arguments: item.uploadedFile.mcpPrompt.arguments,
            content: item.textContent ?? '',
            name: item.name,
            promptName: item.uploadedFile.mcpPrompt.promptName,
            serverName: item.uploadedFile.mcpPrompt.serverName,
            type: AttachmentType.MCP_PROMPT as const
          }
        : null}
  {#if mcpPrompt}
    <ChatAttachmentsListItemMcpPrompt
      class="max-w-[300px] min-w-[200px] flex-shrink-0 {className} {cScrollClasses}"
      isLoading={item.isLoading}
      loadError={item.loadError}
      onRemove={onFileRemove ? () => onFileRemove(item.id) : undefined}
      prompt={mcpPrompt}
      {readonly}
    />
  {/if}
{:else if fIsMcpResource(item)}
  {@const mcpResource = item.attachment as DatabaseMessageExtraMcpResource}

  <ChatAttachmentsListItemMcpResource
    attachment={fToMcpResourceAttachment(mcpResource, item.id)}
    class="flex-shrink-0 {className} {cScrollClasses}"
    onclick={() => onMcpResourcePreview?.(mcpResource)}
  />
{:else if item.isImage && item.preview}
  <ChatAttachmentsListItemThumbnailImage
    class="{cLayoutClasses} cursor-pointer {className} {cScrollClasses}"
    height={imageHeight}
    id={item.id}
    {imageClass}
    name={item.name}
    onRemove={onFileRemove}
    onclick={() => onPreview?.(item)}
    preview={item.preview}
    {readonly}
    width={imageWidth}
  />
{:else if fIsPdfFile(item.attachment, item.uploadedFile)}
  <ChatAttachmentsListItemThumbnailFile
    attachment={item.attachment}
    class="flex-shrink-0 cursor-pointer {className} {cScrollClasses}"
    id={item.id}
    name={item.name}
    onRemove={onFileRemove}
    onclick={() => onPreview?.(item)}
    {readonly}
    size={item.size}
    textContent={item.textContent}
    uploadedFile={item.uploadedFile}
  />
{:else}
  <ChatAttachmentsListItemThumbnailFile
    attachment={item.attachment}
    class="flex-shrink-0 cursor-pointer {className} {cScrollClasses}"
    id={item.id}
    name={item.name}
    onRemove={onFileRemove}
    onclick={() => onPreview?.(item)}
    {readonly}
    size={item.size}
    textContent={item.textContent}
    uploadedFile={item.uploadedFile}
  />
{/if}
