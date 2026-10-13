// Meta parser for `run_javascript` tool calls. Reads the JS code and
// optional timeout from args (strict parsing) and surfaces any error
// from the result blob. SandboxService.formatReply emits a JSON object
// containing an `error` field on failure, but a partial/non-JSON
// failure renders as a flat line beginning with `Error:`. Both shapes
// are handled.

import { fParseToolArgs } from './_shared';
import { cJsonArrayOpen, cJsonObjectOpen } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types';

export type RunJavascriptMeta = {
  code: string;
  timeoutMs?: number;
  errorMessage?: string;
};

export function fParseRunJavascriptMeta(pSection: AgenticSection): RunJavascriptMeta | null {
  const dArgs = fParseToolArgs(BuiltInTool.cBrowserRunJavascript, pSection);

  if (!dArgs) return null;

  const cCode = typeof dArgs.code === 'string' ? dArgs.code : '';

  if (!cCode) return null;

  const cTimeoutRaw = Number(dArgs.timeout_ms);
  const cTimeoutMs = Number.isFinite(cTimeoutRaw) && cTimeoutRaw > 0 ? cTimeoutRaw : undefined;

  let vErrorMessage: string | undefined;

  const cToolResultString = pSection.toolResult;

  if (cToolResultString) {
    // Branches matter here: a JSON object can carry `error`, but a
    // JSON array always represents successful output (sandbox returns
    // the array of values). Only when the result isn't a JSON object
    // do we scan raw lines for the `Error:` prefix.
    let dParsedObject: Record<string, unknown> | null = null;

    // Successful sandbox output is a JSON array, errors are objects; plain
    // text (huge console logs) fails the parse below anyway, so only try
    // when the blob starts with a JSON container
    const cTrimmedResult = cToolResultString.trimStart();

    if (cTrimmedResult[0] === cJsonObjectOpen || cTrimmedResult[0] === cJsonArrayOpen) {
      try {
        const cParsed: unknown = JSON.parse(cTrimmedResult);

        if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
          dParsedObject = cParsed as Record<string, unknown>;
        }
      } catch {
        dParsedObject = null;
      }
    }

    if (typeof dParsedObject?.error === 'string') {
      vErrorMessage = dParsedObject.error;
    } else if (!dParsedObject) {
      const cErrorLine = cToolResultString
        .split('\n')
        .map((pLine) => pLine.trim())
        .find((pLine) => pLine.startsWith('Error:'));

      if (cErrorLine) vErrorMessage = cErrorLine.slice('Error:'.length).trim();
    }
  }

  return { code: cCode, errorMessage: vErrorMessage, timeoutMs: cTimeoutMs };
}
