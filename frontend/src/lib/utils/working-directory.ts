/**
 * Pure helpers for the working-directory picker search, backed by the
 * server's `file_glob_search` tool. Queries starting from a root (`/`,
 * `C:\`, `\\host\share`) or `~` navigate the tree (search the parent for
 * the last segment); anything else glob-matches home-relative entries.
 */

import { fLastPathSegment } from './path-display';
import {
  cGlob,
  cHomeTilde,
  cLeadingSlashesRegex,
  cPathSeparator,
  SEARCH,
  cTrailingSlashesRegex
} from '$lib/constants';
import type { GlobEntry, GlobSearchArgs } from '$lib/types/glob';

export interface PathQuery {
  parent: string;
  last: string;
}

/**
 * Rewrite `\` into `/` when the query carries a Windows root. Elsewhere the
 * backslash is left alone: it is a legal filename character on POSIX.
 */
function fToPosixSeparators(pQuery: string): string {
  if (!cGlob.DRIVE_PREFIX_REGEX.test(pQuery) && !pQuery.startsWith(cGlob.WINDOWS_SEPARATOR))
    return pQuery;

  return pQuery.split(cGlob.WINDOWS_SEPARATOR).join(cPathSeparator);
}

export function fRootPrefixLength(pPath: string): number {
  const cUnc = pPath.match(cGlob.UNC_ROOT_REGEX);

  if (cUnc) return cUnc[0].length;

  const cDrive = pPath.match(cGlob.DRIVE_ROOT_REGEX);

  if (cDrive) return cDrive[0].length;

  return pPath.startsWith(cPathSeparator) ? cPathSeparator.length : 0;
}

/** A query starting from a root or from `~` is path navigation, not a home-relative glob. */
export function fSplitPathQuery(pQuery: string): PathQuery | null {
  const cNormalized = fToPosixSeparators(pQuery);
  const cRootLength = fRootPrefixLength(cNormalized);

  if (cRootLength === 0 && !cNormalized.startsWith(cHomeTilde)) return null;

  // a root keeps its trailing separator so it stays absolute on its own
  const root =
    cRootLength > 0
      ? cNormalized.slice(0, cRootLength).replace(cTrailingSlashesRegex, '') + cPathSeparator
      : cHomeTilde;
  const cRest = cNormalized
    .slice(cRootLength > 0 ? cRootLength : cHomeTilde.length)
    .replace(cLeadingSlashesRegex, '')
    .replace(cTrailingSlashesRegex, '');
  const fParentOf = (pDirs: string) =>
    cRootLength > 0 ? root + pDirs : cHomeTilde + cPathSeparator + pDirs;

  if (!cRest) return { last: '', parent: root };

  const cIdx = cRest.lastIndexOf(cPathSeparator);

  if (cIdx === -1) return { last: cRest, parent: root };

  return { last: cRest.slice(cIdx + 1), parent: fParentOf(cRest.slice(0, cIdx)) };
}

export function fBuildCaseInsensitiveGlob(pQuery: string): string {
  let vOut = cGlob.WILDCARD;

  for (const c of pQuery) {
    const cLo = c.toLowerCase();
    const cUp = c.toUpperCase();

    if (cLo !== cUp) vOut += cGlob.RANGE_OPEN + cLo + cUp + cGlob.RANGE_CLOSE;
    // glob metacharacters are escaped into a literal character class so a
    // query like "a*b" matches a literal '*' instead of becoming "ab"
    else if (cGlob.SPECIAL_CHARS.includes(c)) vOut += cGlob.RANGE_OPEN + c + cGlob.RANGE_CLOSE;
    else vOut += c;
  }

  return vOut + cGlob.WILDCARD;
}

export function fBuildGlobSearchArgs(
  pQuery: string,
  pScopePath: string,
  pSearchDepth: number
): GlobSearchArgs {
  const cPathQuery = fSplitPathQuery(pQuery);
  const cPath = cPathQuery ? cPathQuery.parent : pScopePath;
  const cInclude = cPathQuery
    ? cPathQuery.last
      ? fBuildCaseInsensitiveGlob(cPathQuery.last)
      : cGlob.WILDCARD
    : fBuildCaseInsensitiveGlob(pQuery);
  const cMaxDepth = cPathQuery ? SEARCH.PATH_NAV_MAX_DEPTH : pSearchDepth;

  return {
    include: cInclude,
    last: cPathQuery?.last,
    maxDepth: cMaxDepth,
    path: cPath,
    rankQuery: cPathQuery?.last ?? pQuery
  };
}

const cRankExact = 0;
const cRankPrefix = 1;
const cRankSubstring = 2;
const cRankOther = 3;

function fRankScore(pPath: string, pQuery: string): number {
  const name = fLastPathSegment(pPath).toLowerCase();
  const cQ = pQuery.toLowerCase();

  if (name === cQ) return cRankExact;

  if (name.startsWith(cQ)) return cRankPrefix;

  if (name.includes(cQ)) return cRankSubstring;

  return cRankOther;
}

export function fRankEntries(entries: GlobEntry[], pQuery: string): GlobEntry[] {
  return [...entries].sort(
    (pA, pB) =>
      fRankScore(pA.path, pQuery) - fRankScore(pB.path, pQuery) ||
      pA.path.length - pB.path.length ||
      pA.path.localeCompare(pB.path)
  );
}

export function fJoinPath(pBase: string, pRel: string): string {
  if (!pBase) return pRel;

  return pBase.replace(cTrailingSlashesRegex, '') + cPathSeparator + pRel;
}

export function fHighlightMatch(pText: string, pQuery: string): { text: string; match: boolean }[] {
  if (!pQuery) return [{ match: false, text: pText }];

  const aSegments: { text: string; match: boolean }[] = [];
  const cLowerText = pText.toLowerCase();
  const cLowerQuery = pQuery.toLowerCase();

  let vI = 0;

  while (vI < pText.length) {
    const cIdx = cLowerText.indexOf(cLowerQuery, vI);

    if (cIdx < 0) {
      aSegments.push({ match: false, text: pText.slice(vI) });

      break;
    }

    if (cIdx > vI) aSegments.push({ match: false, text: pText.slice(vI, cIdx) });

    aSegments.push({ match: true, text: pText.slice(cIdx, cIdx + pQuery.length) });
    vI = cIdx + pQuery.length;
  }

  return aSegments;
}
