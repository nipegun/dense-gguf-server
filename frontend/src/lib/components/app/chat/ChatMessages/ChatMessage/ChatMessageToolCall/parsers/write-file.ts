// Meta parser for `write_file` tool calls. Reads the path/content from
// the streamed args (partial JSON so we can render before the call
// finishes) and surfaces `bytes`, `result`, and `error` from the
// result blob.

import { fExtractToolArgString, fParseToolArgs } from './_shared';
import { cCodeBlock, cFilePathSeparatorRegex, aToolArgPathKeys } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection, WriteFileMeta, WriteFileTitleMeta } from '$lib/types';
import { fGetFileTypeByExtension, fTryParseToolResultObject } from '$lib/utils';

export function fParseWriteFileMeta(pSection: AgenticSection): WriteFileMeta | null {
  const dArgs = fParseToolArgs(BuiltInTool.cServerWriteFile, pSection, { partial: true });

  if (!dArgs) return null;

  // Tool contracts drifted over time: some models emit `path`,
  // others `file_path` / `filePath`. Accept all three.
  const cRawPath = dArgs.path ?? dArgs.file_path ?? dArgs.filePath;

  if (typeof cRawPath !== 'string' || !cRawPath) return null;

  const cFileName = cRawPath.split(cFilePathSeparatorRegex).pop() || cRawPath;
  const cContent = typeof dArgs.content === 'string' ? dArgs.content : '';
  const cLanguage =
    fGetFileTypeByExtension(cRawPath)?.replace(cCodeBlock.TEXT_LANGUAGE_PREFIX_REGEX, '') ??
    cCodeBlock.DEFAULT_LANGUAGE;
  const dResultObj = fTryParseToolResultObject(pSection.toolResult);
  const cBytesWritten =
    dResultObj && Number.isFinite(Number(dResultObj.bytes)) ? Number(dResultObj.bytes) : undefined;
  const cResultMessage = typeof dResultObj?.result === 'string' ? dResultObj.result : undefined;
  const cErrorMessage = typeof dResultObj?.error === 'string' ? dResultObj.error : undefined;

  return {
    bytesWritten: cBytesWritten,
    content: cContent,
    errorMessage: cErrorMessage,
    fileName: cFileName,
    filePath: cRawPath,
    language: cLanguage,
    resultMessage: cResultMessage
  };
}

/**
 * Title-tier meta for write_file blocks: everything the header and status
 * pill render, obtained without parsing the embedded file content. The path
 * comes from a targeted key extraction; the full parse runs only as a
 * fallback for arg shapes the extraction can't see.
 */
export function fParseWriteFileTitleMeta(pSection: AgenticSection): WriteFileTitleMeta | null {
  if (pSection.toolName !== BuiltInTool.cServerWriteFile || !pSection.toolArgs) return null;

  let vRawPath: string | undefined = fExtractToolArgString(pSection.toolArgs, aToolArgPathKeys);

  if (!vRawPath) {
    const dArgs = fParseToolArgs(BuiltInTool.cServerWriteFile, pSection, { partial: true });
    const cFallbackPath = dArgs?.path ?? dArgs?.file_path ?? dArgs?.filePath;

    if (typeof cFallbackPath === 'string' && cFallbackPath) vRawPath = cFallbackPath;
  }

  if (!vRawPath) return null;

  const cFileName = vRawPath.split(cFilePathSeparatorRegex).pop() || vRawPath;
  const cLanguage =
    fGetFileTypeByExtension(vRawPath)?.replace(cCodeBlock.TEXT_LANGUAGE_PREFIX_REGEX, '') ??
    cCodeBlock.DEFAULT_LANGUAGE;
  const dResultObj = fTryParseToolResultObject(pSection.toolResult);
  const cBytesWritten =
    dResultObj && Number.isFinite(Number(dResultObj.bytes)) ? Number(dResultObj.bytes) : undefined;
  const cResultMessage = typeof dResultObj?.result === 'string' ? dResultObj.result : undefined;
  const cErrorMessage = typeof dResultObj?.error === 'string' ? dResultObj.error : undefined;

  return {
    bytesWritten: cBytesWritten,
    errorMessage: cErrorMessage,
    fileName: cFileName,
    filePath: vRawPath,
    language: cLanguage,
    resultMessage: cResultMessage
  };
}
