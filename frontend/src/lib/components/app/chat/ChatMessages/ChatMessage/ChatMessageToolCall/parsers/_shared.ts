// Helpers shared by the per-tool meta parsers under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// Each tool needs the same first three steps (tool-name check,
// args-present check, JSON parse) - keeping them here lets each parser
// stay focused on its own format quirks.

import { cToolArgStringFieldPatternTemplate } from '$lib/constants';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types/agentic';
import { fParsePartialJsonArgs } from '$lib/utils/parse-partial-json-args';

/**
 * Strict (final-state) JSON parser for a tool-args blob. Mirrors the
 * behaviour the per-tool components used before extraction: an
 * invalid JSON blob, a JSON array, or a JSON primitive all map to
 * `null` so callers don't have to guard against surprise shapes.
 */
function fParseFinalToolArgs(pBlob: string): Record<string, unknown> | null {
  try {
    const cParsed: unknown = JSON.parse(pBlob);

    if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
      return cParsed as Record<string, unknown>;
    }

    return null;
  } catch {
    return null;
  }
}

// Compiled per key on first use; the key set is tiny and fixed.
const dToolArgStringRegexes = new Map<string, RegExp>();

/**
 * Extract a string field from a JSON tool-args blob without parsing the
 * whole document. write_file and edit_file args embed full file contents,
 * yet the block title needs only the path; a targeted key match plus a
 * JSON.parse of the captured string literal alone keeps title rendering
 * O(path) instead of O(blob). Returns undefined when the key is missing
 * or its value is not a string; callers fall back to the full parse.
 */
export function fExtractToolArgString(
  pToolArgs: string,
  pKeys: readonly string[]
): string | undefined {
  for (const cKey of pKeys) {
    let vPattern = dToolArgStringRegexes.get(cKey);

    if (!vPattern) {
      vPattern = new RegExp(cToolArgStringFieldPatternTemplate.replace('{key}', cKey));
      dToolArgStringRegexes.set(cKey, vPattern);
    }

    const cMatch = vPattern.exec(pToolArgs);

    if (!cMatch) continue;

    try {
      const cValue: unknown = JSON.parse(`"${cMatch[1]}"`);

      if (typeof cValue === 'string') return cValue;
    } catch {
      // fall through to the next key; the full parse is the fallback
    }
  }

  return undefined;
}

/**
 * Parse a section's toolArgs against an expected tool name. Returns
 * `null` when:
 *   - the section's toolName doesn't match (component isn't for this
 *     tool);
 *   - the section has no args yet (call hasn't started streaming);
 *   - or the args blob can't be parsed.
 *
 * Pass `{ partial: true }` for tools that need to render incrementally
 * as each token lands (read_file, edit_file, write_file).
 */
export function fParseToolArgs(
  pExpected: BuiltInTool,
  pSection: AgenticSection,
  pOptions: { partial?: boolean } = {}
): Record<string, unknown> | null {
  if (pSection.toolName !== pExpected || !pSection.toolArgs) return null;

  return pOptions.partial
    ? fParsePartialJsonArgs(pSection.toolArgs)
    : fParseFinalToolArgs(pSection.toolArgs);
}
