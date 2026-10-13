// Meta parser for `read_file` tool calls. Reads the file path and an
// optional line range (either `start_line`+`end_line` or
// `start_line`+`line_count`). Args are parsed partially so a header
// can render incrementally as the file path streams in.

import { fParseToolArgs } from './_shared';
import { cCodeBlock, cFilePathSeparatorRegex } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types';
import { fGetFileTypeByExtension } from '$lib/utils';

export type ReadFileMeta = {
  fileName: string;
  lineRange: { start: number; end: number } | null;
  language: string;
};

export function fParseReadFileMeta(pSection: AgenticSection): ReadFileMeta | null {
  const dArgs = fParseToolArgs(BuiltInTool.cServerReadFile, pSection, { partial: true });

  if (!dArgs) return null;

  const cRawPath = dArgs.path ?? dArgs.file_path ?? dArgs.filePath;

  if (typeof cRawPath !== 'string' || !cRawPath) return null;

  const cFileName = cRawPath.split(cFilePathSeparatorRegex).pop() || cRawPath;
  // Models emit range arguments under several aliases. Accept all to
  // stay forgiving across prompt variations.
  const cStartRaw = dArgs.start_line ?? dArgs.line_start ?? dArgs.startLine ?? dArgs.from_line;
  const cEndRaw = dArgs.end_line ?? dArgs.line_end ?? dArgs.endLine ?? dArgs.to_line;
  const cCountRaw = dArgs.line_count ?? dArgs.count ?? dArgs.num_lines;

  let vLineRange: { start: number; end: number } | null = null;

  const sNum = Number(cStartRaw);
  const cENum = Number(cEndRaw);

  if (cStartRaw != null && cEndRaw != null && Number.isFinite(sNum) && Number.isFinite(cENum)) {
    vLineRange = { end: cENum, start: sNum };
  } else if (cStartRaw != null && cCountRaw != null) {
    const cNum = Number(cCountRaw);

    if (Number.isFinite(sNum) && Number.isFinite(cNum)) {
      vLineRange = { end: sNum + cNum - 1, start: sNum };
    }
  }

  const cFileType = fGetFileTypeByExtension(cFileName);
  const cLanguage = cFileType
    ? cFileType.replace(cCodeBlock.TEXT_LANGUAGE_PREFIX_REGEX, '')
    : cCodeBlock.DEFAULT_LANGUAGE;

  return { fileName: cFileName, language: cLanguage, lineRange: vLineRange };
}
