import { cFilePathSeparatorRegex, cNewline } from '$lib/constants';
import {
  cPrefixFile,
  cPrefixMime,
  cPrefixSize,
  cReadMediaSizeRegex
} from '$lib/constants/read-media';
import type { AgenticSection } from '$lib/types';

export interface ReadMediaMeta {
  fileName: string;
  path: string;
  sizeBytes?: number;
  mimeType?: string;
}

/**
 * Parse read_media tool result to extract metadata.
 * Expected format (after extractBase64Attachments processing):
 *   File: /path/to/file.png
 *   Size: 12345 bytes
 *   MIME: image/png
 *   [Attachment saved: mcp-attachment-xxx.png]
 *
 * The data URI line is replaced by the attachment marker by
 * agenticStore.extractBase64Attachments before storage.
 */
export function fParseReadMediaMeta(pSection: AgenticSection): ReadMediaMeta | null {
  if (!pSection.toolResult) return null;

  const aLines = pSection.toolResult.split(cNewline);

  let vFileName = '';
  let vPath = '';
  let vSizeBytes: number | undefined;
  let vMimeType: string | undefined;

  for (const cLine of aLines) {
    const cTrimmed = cLine.trim();

    if (cTrimmed.startsWith(cPrefixFile)) {
      vPath = cTrimmed.slice(cPrefixFile.length).trim();
      vFileName = vPath.split(cFilePathSeparatorRegex).pop() ?? vPath;
    } else if (cTrimmed.startsWith(cPrefixSize)) {
      const cMatch = cTrimmed.match(cReadMediaSizeRegex);

      if (cMatch) vSizeBytes = Number(cMatch[1]);
    } else if (cTrimmed.startsWith(cPrefixMime)) {
      vMimeType = cTrimmed.slice(cPrefixMime.length).trim();
    }
  }

  if (!vPath) return null;

  return { fileName: vFileName, mimeType: vMimeType, path: vPath, sizeBytes: vSizeBytes };
}
