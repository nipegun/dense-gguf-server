import {
  cCwdChangedPrefix,
  cCwdClearedText,
  cCwdLinkRegex,
  cFileUriPrefix,
  cHomeTilde,
  cHomeTildePrefix,
  cPathSeparator,
  cTrailingSlashesRegex
} from '$lib/constants';

export function fLastPathSegment(pP: string): string {
  const cTrimmed = pP.replace(cTrailingSlashesRegex, '');
  const cIdx = cTrimmed.lastIndexOf(cPathSeparator);

  return cIdx === -1 ? cTrimmed : cTrimmed.slice(cIdx + 1);
}

// `~/...` under `home`; falls back to the basename when home is unknown
// or the path is outside it.
export function fAbbreviateWorkingDir(
  pPath: string | null | undefined,
  pHome: string | null | undefined
): string {
  if (!pPath) return '';

  if (!pHome) return fLastPathSegment(pPath);

  if (pPath === pHome) return cHomeTilde;

  if (pPath.startsWith(pHome + cPathSeparator))
    return cHomeTildePrefix + pPath.slice(pHome.length + 1);

  return fLastPathSegment(pPath);
}

// Unlike abbreviateWorkingDir, paths outside `home` are returned
// unchanged - used where the full path matters.
export function fAbbreviateHome(pPath: string, pHome: string | null | undefined): string {
  if (!pHome) return pPath;

  if (pPath === pHome) return cHomeTilde;

  if (pPath.startsWith(pHome + cPathSeparator))
    return cHomeTildePrefix + pPath.slice(pHome.length + 1);

  return pPath;
}

export { cCwdChangedPrefix, cCwdClearedText } from '$lib/constants';

export interface CwdMessageInfo {
  // absolute server-side path, null when the cwd was cleared
  path: string | null;
  // display form shown in the UI (e.g. ~/Documents)
  display: string;
}

/**
 * Format a synthetic cwd-change message. The path travels as
 * `[file:///abs/path](display)` so both the absolute and short form are
 * visible to the model and parseable back by the UI.
 */
export function fFormatCwdMessage(pCwd: string, pHome: string | null): string {
  const cDisplay = fAbbreviateWorkingDir(pCwd, pHome);

  return `${cCwdChangedPrefix}[${cFileUriPrefix}${pCwd}](${cDisplay}).`;
}

/**
 * Parse a synthetic cwd message back into its parts. The caller must
 * already know the message is synthetic (via the persisted `isSynthetic`
 * flag); this only extracts the path.
 */
export function fParseCwdMessage(pContent: string): CwdMessageInfo | null {
  const cTrimmed = pContent.trim();

  if (cTrimmed === cCwdClearedText) {
    return { display: '', path: null };
  }

  if (cTrimmed.startsWith(cCwdChangedPrefix)) {
    const cRest = cTrimmed.slice(cCwdChangedPrefix.length);
    // not anchored to the end: guidance may follow the link
    const cLink = cRest.match(cCwdLinkRegex);

    if (cLink) return { display: cLink[2], path: cLink[1] };

    return { display: cRest, path: cRest };
  }

  return null;
}
