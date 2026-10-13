import { AttachmentType, UrlProtocol } from '$lib/enums';
import type { DatabaseMessageExtra, DatabaseMessageExtraImageFile } from '$lib/types/database';
import type { Root as HastRoot } from 'hast';
import { visit } from 'unist-util-visit';

/**
 * Rehype plugin to resolve attachment image sources.
 * Converts attachment names (e.g., "mcp-attachment-xxx.png") to base64 data URLs.
 */
export function fRehypeResolveAttachmentImages(pOptions: { attachments?: DatabaseMessageExtra[] }) {
  return (pTree: HastRoot) => {
    visit(pTree, 'element', (node) => {
      if (node.tagName === 'img' && node.properties?.src) {
        const cSrc = String(node.properties.src);

        // Skip data URLs and external URLs
        if (cSrc.startsWith(UrlProtocol.DATA) || cSrc.startsWith(UrlProtocol.cHttp)) {
          return;
        }

        // Find matching attachment
        const attachment = pOptions.attachments?.find(
          (pA): pA is DatabaseMessageExtraImageFile =>
            pA.type === AttachmentType.IMAGE && pA.name === cSrc
        );

        // Replace with base64 URL if found
        if (attachment?.base64Url) {
          node.properties.src = attachment.base64Url;
        }
      }
    });
  };
}
