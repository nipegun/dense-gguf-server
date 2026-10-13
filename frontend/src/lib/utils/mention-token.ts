// An `@` starts a mention only when preceded by start-of-string or one of
// these; identifier chars are not delimiters, so a mid-word `@` does not.
const sTokenBoundaryChars = new Set([
  ' ',
  '\t',
  '\n',
  '\r',
  '(',
  ')',
  '[',
  ']',
  ',',
  ';',
  ':',
  '"',
  "'"
]);

/**
 * Find the most-recent `@`-mention token whose extent includes `cursor`;
 * the query covers the whole `@...` token regardless of caret position.
 */
export function fFindMentionToken(
  pValue: string,
  pCursor: number
): { start: number; end: number; query: string } | null {
  if (pCursor <= 0 || pCursor > pValue.length) return null;

  let vAtIndex = -1;

  for (let vI = pCursor - 1; vI >= 0; vI--) {
    const cCh = pValue[vI];

    if (cCh === '@') {
      const cPrev = vI > 0 ? pValue[vI - 1] : '';

      if (vI === 0 || sTokenBoundaryChars.has(cPrev)) {
        vAtIndex = vI;
      }

      break;
    }

    if (sTokenBoundaryChars.has(cCh)) break;
  }

  if (vAtIndex === -1) return null;

  let vEnd = vAtIndex + 1;

  while (vEnd < pValue.length && !sTokenBoundaryChars.has(pValue[vEnd])) {
    vEnd++;
  }

  return {
    end: vEnd,
    query: pValue.slice(vAtIndex + 1, vEnd),
    start: vAtIndex
  };
}

/**
 * Stable signature of a mention token for use as a "dismissed" marker:
 * while the picker is closed and this exact token is still intact, the
 * picker does not silently re-open on in-token edits.
 */
export interface MentionDismissSnapshot {
  start: number;
  query: string;
}

export function fTakeMentionDismissSnapshot(
  pValue: string,
  pCursor: number
): MentionDismissSnapshot | null {
  const cToken = fFindMentionToken(pValue, pCursor);

  if (!cToken) return null;

  return { query: cToken.query, start: cToken.start };
}
