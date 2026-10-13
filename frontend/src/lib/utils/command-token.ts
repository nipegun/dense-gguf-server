/**
 * Slash-command token detection for the chat form. Valid only at offset 0.
 */
export function fFindCommandToken(
  pValue: string
): { name: string; args: string; end: number } | null {
  if (!pValue.startsWith('/')) return null;

  const cRest = pValue.slice(1);
  const cSpaceIdx = cRest.search(/\s/);
  const name = cSpaceIdx === -1 ? cRest : cRest.slice(0, cSpaceIdx);
  const cArgs = cSpaceIdx === -1 ? '' : cRest.slice(cSpaceIdx + 1);

  return { args: cArgs, end: pValue.length, name };
}

/**
 * Stable signature of a slash-command token for use as a "dismissed"
 * marker: while the picker is closed and this exact token is still intact,
 * the picker does not re-open on in-token edits.
 */
export interface CommandDismissSnapshot {
  name: string;
  args: string;
}

export function fTakeCommandDismissSnapshot(pValue: string): CommandDismissSnapshot | null {
  const cToken = fFindCommandToken(pValue);

  if (!cToken) return null;

  return { args: cToken.args, name: cToken.name };
}
