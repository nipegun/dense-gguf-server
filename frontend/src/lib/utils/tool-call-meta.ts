// Generic helper for parsing tool-result blobs (the "out" side of a
// tool call). Used by the per-tool meta parsers under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// Each tool needs to surface fields like `error`, `result`, `bytes`,
// `edits_applied` without repeating the try/JSON.parse/object guard inline.

import { cJsonObjectOpen } from '$lib/constants';

/**
 * Parse a tool-result blob into a JSON object, or `null` if it isn't
 * one. Returns null for:
 *   - missing / empty input,
 *   - a JSON object that turns out to be an array or primitive,
 *   - any parse failure (always returns null rather than throwing).
 */
export function fTryParseToolResultObject(
  pToolResultString: string | undefined
): Record<string, unknown> | null {
  if (!pToolResultString) return null;

  // Tool results are usually large plain text (file contents, stdout); only
  // a JSON object root can carry fields, so skip the parse otherwise
  const cTrimmed = pToolResultString.trimStart();

  if (cTrimmed[0] !== cJsonObjectOpen) return null;

  try {
    const cParsed: unknown = JSON.parse(cTrimmed);

    if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
      return cParsed as Record<string, unknown>;
    }

    return null;
  } catch {
    return null;
  }
}
