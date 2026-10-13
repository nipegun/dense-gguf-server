// Meta parser for `grep_search` tool calls. Reads the path/pattern
// triplet from args (strict parsing - we wait for the args to
// complete) and the matches from the result blob. The result parser
// keeps the original "scan result as raw text on JSON.parse failure"
// fallback so MCP servers that return unparseable output still get
// surfaced.

import { fParseToolArgs } from './_shared';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types';
import { fSplitSearchSummaryList } from '$lib/utils';

export type GrepSearchMatch = {
  file: string;
  line?: number;
  content: string;
};

export type GrepSearchMeta = {
  path: string;
  pattern: string;
  include: string;
  exclude?: string;
  showLineNumbers: boolean;
  matches: GrepSearchMatch[];
  totalMatches?: number;
  errorMessage?: string;
};

export function fParseGrepSearchMeta(pSection: AgenticSection): GrepSearchMeta | null {
  const dArgs = fParseToolArgs(BuiltInTool.cServerGrepSearch, pSection);

  if (!dArgs) return null;

  const cPath = typeof dArgs.path === 'string' ? dArgs.path : '';
  const cPattern = typeof dArgs.pattern === 'string' ? dArgs.pattern : '';

  if (!cPath || !cPattern) return null;

  const cInclude = typeof dArgs.include === 'string' && dArgs.include ? dArgs.include : '**';
  const cExclude = typeof dArgs.exclude === 'string' && dArgs.exclude ? dArgs.exclude : undefined;
  const cShowLineNumbers = dArgs.return_line_numbers === true;

  let aMatches: GrepSearchMatch[] = [];
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

          aMatches = cSplit.lines.map((pLine) => fParseGrepLine(pLine, cShowLineNumbers));
        }
      }
    } catch {
      // Result wasn't JSON: keep behaviour for MCP servers that
      // emit raw text and treat each line as a `<file>:<content>`
      // (or `<file>:<line>:<content>`) match.
      const cSplit = fSplitSearchSummaryList(cToolResultString, (pTotal) => {
        vTotalMatches = pTotal;
      });

      aMatches = cSplit.lines.map((pLine) => fParseGrepLine(pLine, cShowLineNumbers));
    }
  }

  return {
    errorMessage: vErrorMessage,
    exclude: cExclude,
    include: cInclude,
    matches: aMatches,
    path: cPath,
    pattern: cPattern,
    showLineNumbers: cShowLineNumbers,
    totalMatches: vTotalMatches
  };
}

function fParseGrepLine(pLine: string, pShowLineNumbers: boolean): GrepSearchMatch {
  // Server output:
  //   <file>:<content>          when return_line_numbers=false
  //   <file>:<lineno>:<content> when return_line_numbers=true
  const cFirstColon = pLine.indexOf(':');

  if (cFirstColon === -1) {
    return { content: '', file: pLine };
  }

  const file = pLine.slice(0, cFirstColon);
  const cTail = pLine.slice(cFirstColon + 1);

  if (!pShowLineNumbers) {
    return { content: cTail, file };
  }

  const cSecondColon = cTail.indexOf(':');

  if (cSecondColon === -1) {
    return { content: cTail, file };
  }

  const cLineNum = parseInt(cTail.slice(0, cSecondColon), 10);

  return {
    content: cTail.slice(cSecondColon + 1),
    file,
    line: Number.isFinite(cLineNum) ? cLineNum : undefined
  };
}
