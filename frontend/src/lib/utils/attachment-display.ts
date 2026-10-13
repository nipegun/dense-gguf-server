import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import type {
  AttachmentDisplayItemsOptions,
  ChatAttachmentDisplayItem,
  ChatUploadedFile
} from '$lib/types';
import { fGetFileTypeCategory, fGetFileTypeCategoryByExtension, fIsImageFile } from '$lib/utils';

/**
 * Check if a display item represents an MCP prompt
 * (either from attachment type or uploaded file with mcpPrompt metadata)
 */
export function fIsMcpPrompt(pItem: ChatAttachmentDisplayItem): boolean {
  if (pItem.attachment?.type === AttachmentType.MCP_PROMPT) {
    return true;
  }

  if (pItem.uploadedFile?.type === SpecialFileType.MCP_PROMPT && pItem.uploadedFile.mcpPrompt) {
    return true;
  }

  return false;
}

/**
 * Check if a display item represents an MCP resource
 */
export function fIsMcpResource(pItem: ChatAttachmentDisplayItem): boolean {
  return pItem.attachment?.type === AttachmentType.MCP_RESOURCE;
}

/**
 * Gets the file type category from an uploaded file, checking both MIME type and extension
 */
function fGetUploadedFileCategory(pFile: ChatUploadedFile): FileTypeCategory | null {
  const cCategoryByMime = fGetFileTypeCategory(pFile.type);

  if (cCategoryByMime) {
    return cCategoryByMime;
  }

  return fGetFileTypeCategoryByExtension(pFile.name);
}

/**
 * Creates a unified list of display items from uploaded files and stored attachments.
 * Items are returned in reverse order (newest first).
 */
export function fGetAttachmentDisplayItems(
  pOptions: AttachmentDisplayItemsOptions
): ChatAttachmentDisplayItem[] {
  const { attachments = [], uploadedFiles: aUploadedFiles = [] } = pOptions;
  const aItems: ChatAttachmentDisplayItem[] = [];

  // Add uploaded files (ChatForm)
  for (const file of aUploadedFiles) {
    aItems.push({
      id: file.id,
      isImage: fGetUploadedFileCategory(file) === FileTypeCategory.IMAGE,
      isLoading: file.isLoading,
      loadError: file.loadError,
      name: file.name,
      preview: file.preview,
      size: file.size,
      textContent: file.textContent,
      uploadedFile: file
    });
  }

  // Add stored attachments (ChatMessage)
  for (const [vIndex, vAttachment] of attachments.entries()) {
    const cIsImage = fIsImageFile(vAttachment);

    aItems.push({
      attachment: vAttachment,
      attachmentIndex: vIndex,
      id: `attachment-${vIndex}`,
      isImage: cIsImage,
      name: vAttachment.name,
      preview: cIsImage && 'base64Url' in vAttachment ? vAttachment.base64Url : undefined,
      size: 'size' in vAttachment ? vAttachment.size : undefined,
      textContent: 'content' in vAttachment ? vAttachment.content : undefined
    });
  }

  return aItems.reverse();
}
