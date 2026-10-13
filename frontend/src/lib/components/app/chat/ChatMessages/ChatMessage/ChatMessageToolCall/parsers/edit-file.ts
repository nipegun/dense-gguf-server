// Meta parser for `edit_file` tool calls. Reads the file path and the
// array of edits from the streamed args (partial JSON for incremental
// rendering), plus the result blob for `result` / `edits_applied` /
// `error` fields.

import { fExtractToolArgString, fParseToolArgs } from './_shared';
import { cFilePathSeparatorRegex, aToolArgPathKeys } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection, EditFileEdit, EditFileMeta, EditFileTitleMeta } from '$lib/types';
import { fTryParseToolResultObject } from '$lib/utils';

export function fParseEditFileMeta(pSection: AgenticSection): EditFileMeta | null {
  const dArgs = fParseToolArgs(BuiltInTool.cServerEditFile, pSection, { partial: true });

  if (!dArgs) return null;

  const cRawPath = dArgs.path ?? dArgs.file_path ?? dArgs.filePath;

  if (typeof cRawPath !== 'string' || !cRawPath) return null;

  const cFileName = cRawPath.split(cFilePathSeparatorRegex).pop() || cRawPath;
  // Filter the streamed edits array strictly: each entry must be an
  // object with a non-empty `old_text`. Edits without an old_text
  // would diff against empty and render as a full re-write.
  const aRawEdits = Array.isArray(dArgs.edits) ? dArgs.edits : [];
  const aEdits: EditFileEdit[] = [];

  for (const cE of aRawEdits) {
    if (!cE || typeof cE !== 'object' || Array.isArray(cE)) continue;

    const dObj = cE as Record<string, unknown>;
    const cOldText = typeof dObj.old_text === 'string' ? dObj.old_text : '';

    if (!cOldText) continue;

    const cNewText = typeof dObj.new_text === 'string' ? dObj.new_text : '';

    aEdits.push({ newText: cNewText, oldText: cOldText });
  }

  const dResultObj = fTryParseToolResultObject(pSection.toolResult);

  let vResultMessage: string | undefined;
  let vEditsApplied: number | undefined;
  let vErrorMessage: string | undefined;

  if (typeof dResultObj?.error === 'string') {
    vErrorMessage = dResultObj.error;
  } else if (dResultObj) {
    if (typeof dResultObj.result === 'string') {
      vResultMessage = dResultObj.result;
    }

    if (Number.isFinite(Number(dResultObj.edits_applied))) {
      vEditsApplied = Number(dResultObj.edits_applied);
    }
  }

  return {
    edits: aEdits,
    editsApplied: vEditsApplied,
    errorMessage: vErrorMessage,
    fileName: cFileName,
    filePath: cRawPath,
    resultMessage: vResultMessage
  };
}

/**
 * Title-tier meta for edit_file blocks: everything the header and status
 * pill render, obtained without parsing the embedded edit strings. The path
 * comes from a targeted key extraction; the full parse runs only as a
 * fallback for arg shapes the extraction can't see.
 */
export function fParseEditFileTitleMeta(pSection: AgenticSection): EditFileTitleMeta | null {
  if (pSection.toolName !== BuiltInTool.cServerEditFile || !pSection.toolArgs) return null;

  let vRawPath: string | undefined = fExtractToolArgString(pSection.toolArgs, aToolArgPathKeys);

  if (!vRawPath) {
    const dArgs = fParseToolArgs(BuiltInTool.cServerEditFile, pSection, { partial: true });
    const cFallbackPath = dArgs?.path ?? dArgs?.file_path ?? dArgs?.filePath;

    if (typeof cFallbackPath === 'string' && cFallbackPath) vRawPath = cFallbackPath;
  }

  if (!vRawPath) return null;

  const cFileName = vRawPath.split(cFilePathSeparatorRegex).pop() || vRawPath;
  const dResultObj = fTryParseToolResultObject(pSection.toolResult);

  let vResultMessage: string | undefined;
  let vEditsApplied: number | undefined;
  let vErrorMessage: string | undefined;

  if (typeof dResultObj?.error === 'string') {
    vErrorMessage = dResultObj.error;
  } else if (dResultObj) {
    if (typeof dResultObj.result === 'string') {
      vResultMessage = dResultObj.result;
    }

    if (Number.isFinite(Number(dResultObj.edits_applied))) {
      vEditsApplied = Number(dResultObj.edits_applied);
    }
  }

  return {
    editsApplied: vEditsApplied,
    errorMessage: vErrorMessage,
    fileName: cFileName,
    filePath: vRawPath,
    resultMessage: vResultMessage
  };
}
