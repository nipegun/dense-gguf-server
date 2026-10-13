// Meta parser for `file_glob_search` tool calls. Reads the path,
// include pattern, and optional exclude from the args (strict parsing)
// and the matches from the result blob. Like grep_search, the result
// parser keeps the original raw-text fallback for MCP servers that
// emit unparseable output.

import { fParseToolArgs } from './_shared';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types';
import { fSplitSearchSummaryList } from '$lib/utils';

export type FileGlobSearchMeta = {
  path: string;
  include: string;
  exclude?: string;
  matches: string[];
  totalMatches?: number;
  errorMessage?: string;
};

export function fParseFileGlobSearchMeta(pSection: AgenticSection): FileGlobSearchMeta | null {
  const dArgs = fParseToolArgs(BuiltInTool.cServerFileGlobSearch, pSection);

  if (!dArgs) return null;

  const cPath = typeof dArgs.path === 'string' ? dArgs.path : '';
  const cInclude = typeof dArgs.include === 'string' && dArgs.include ? dArgs.include : '**';
  const cExclude = typeof dArgs.exclude === 'string' && dArgs.exclude ? dArgs.exclude : undefined;

  if (!cPath) return null;

  let aMatches: string[] = [];
  let vTotalMatches: number | undefined;
  let vErrorMessage: string | undefined;

  const cToolResultString = pSection.toolResult;

  if (cToolResultString) {
    try {
      const cParsed: unknown = JSON.parse(cToolResultString);

      if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
        const dObj = cParsed as Record<string, unknown>;

        if (typeof dObj.error === 'string') {
          vErrorMessage = dObj.error;
        } else if (typeof dObj.plain_text_response === 'string') {
          const cSplit = fSplitSearchSummaryList(dObj.plain_text_response, (pTotal) => {
            vTotalMatches = pTotal;
          });

          aMatches = cSplit.lines;
        }
      }
    } catch {
      // See grep-search.ts: same fallback used there.
      const cSplit = fSplitSearchSummaryList(cToolResultString, (pTotal) => {
        vTotalMatches = pTotal;
      });

      aMatches = cSplit.lines;
    }
  }

  return {
    errorMessage: vErrorMessage,
    exclude: cExclude,
    include: cInclude,
    matches: aMatches,
    path: cPath,
    totalMatches: vTotalMatches
  };
}
