export function fParseExecShellCommandError(
  pToolResultString: string | undefined
): string | undefined {
  if (!pToolResultString) return undefined;

  // Exec results are usually large plain-text stdout; only a JSON object
  // root can carry an error field, so skip the parse otherwise
  const cTrimmed = pToolResultString.trimStart();

  if (cTrimmed[0] !== '{') return undefined;

  try {
    const cParsed: unknown = JSON.parse(cTrimmed);

    if (
      cParsed &&
      typeof cParsed === 'object' &&
      !Array.isArray(cParsed) &&
      typeof (cParsed as Record<string, unknown>).error === 'string'
    ) {
      return (cParsed as { error: string }).error;
    }
  } catch {
    // Plain-text result = stdout/stderr, no structured error to surface.
  }

  return undefined;
}
