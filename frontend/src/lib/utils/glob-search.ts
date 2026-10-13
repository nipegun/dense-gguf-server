/**
 * Shared `file_glob_search` runners with a short-lived result cache, so a
 * repeated query for the same (type, path, glob, depth) reuses the last
 * result instead of re-walking the tree.
 */

import { fLastPathSegment } from './path-display';
import { fBuildGlobSearchArgs, fJoinPath, fRankEntries } from './working-directory';
import { cGlob, cPathSeparator, SEARCH } from '$lib/constants';
import { BuiltInTool, GlobSearchType } from '$lib/enums';
import { ToolsService } from '$lib/services/tools.service';
import type {
  GlobEntry,
  GlobEntryResult,
  GlobSearchArgs,
  GlobSearchChildOptions,
  GlobSearchChildResult,
  GlobSearchResult
} from '$lib/types/glob';

const cSearchCacheTtlMs = 2000;

interface CacheEntry {
  results: GlobEntry[];
  base: string;
  at: number;
}

const dSearchCache = new Map<string, CacheEntry>();

export async function fRunGlobSearch(
  pArgs: GlobSearchArgs,
  pType: GlobSearchType,
  pLimit: number,
  pSignal: AbortSignal
): Promise<GlobSearchResult> {
  const cKey = `${pType}\u0000${pArgs.path}\u0000${pArgs.include}\u0000${pArgs.maxDepth}\u0000${pLimit}`;
  const cCached = dSearchCache.get(cKey);

  if (cCached && Date.now() - cCached.at < cSearchCacheTtlMs) {
    return { base: cCached.base, entries: cCached.results };
  }

  const dRes = await ToolsService.mExecuteToolRaw(
    BuiltInTool.cServerFileGlobSearch,
    {
      include: pArgs.include,
      limit: pLimit,
      max_depth: pArgs.maxDepth,
      path: pArgs.path,
      type: pType
    },
    pSignal
  );

  if (typeof dRes.error === 'string') return { base: '', entries: [], error: dRes.error };

  const cBase = typeof dRes.base === 'string' ? dRes.base : '';
  const entries = Array.isArray(dRes.entries) ? (dRes.entries as GlobEntry[]) : [];
  const cNow = Date.now();

  // prune stale entries so the short-lived cache cannot grow unbounded
  for (const [vK, vV] of dSearchCache) {
    if (cNow - vV.at >= cSearchCacheTtlMs) dSearchCache.delete(vK);
  }
  dSearchCache.set(cKey, { at: cNow, base: cBase, results: entries });

  return { base: cBase, entries };
}

function fToEntryResult(pE: GlobEntry, pBase: string): GlobEntryResult {
  return { name: fLastPathSegment(pE.path), path: fJoinPath(pBase, pE.path), type: pE.type };
}

/**
 * One ranked glob search that may also list the matched directory's
 * children, shared by the WD picker (descend on exact match) and the
 * mention picker (descend on a trailing `/` or `\`).
 */
export async function fRunGlobSearchWithChildren(
  pQuery: string,
  pScopePath: string,
  pSearchDepth: number,
  pLimit: number,
  pSignal: AbortSignal,
  pOptions: GlobSearchChildOptions = {}
): Promise<GlobSearchChildResult> {
  const {
    childMaxDepth: vChildMaxDepth = SEARCH.PATH_NAV_MAX_DEPTH,
    descendOnTrailingSeparator: vDescendOnTrailingSeparator = false,
    type: vType = GlobSearchType.cAll
  } = pOptions;
  const cArgs = fBuildGlobSearchArgs(pQuery, pScopePath, pSearchDepth);
  const cRes = await fRunGlobSearch(cArgs, vType, pLimit, pSignal);

  if (cRes.error) return { args: cArgs, base: cRes.base, entries: [], error: cRes.error };

  const aRanked = fRankEntries(cRes.entries, cArgs.rankQuery);
  const entries = aRanked.map((pE) => fToEntryResult(pE, cRes.base));
  const cLast = cArgs.last;

  if (cLast) {
    const cWantsDescend = vDescendOnTrailingSeparator
      ? pQuery.endsWith(cPathSeparator) || pQuery.endsWith(cGlob.WINDOWS_SEPARATOR)
      : true;
    const cExact = aRanked.find(
      (pE) => pE.type === 'dir' && fLastPathSegment(pE.path).toLowerCase() === cLast.toLowerCase()
    );

    if (cWantsDescend && cExact) {
      const cExactDir = fJoinPath(cRes.base, cExact.path);
      const cChildRes = await fRunGlobSearch(
        { include: cGlob.WILDCARD, maxDepth: vChildMaxDepth, path: cExactDir, rankQuery: '' },
        vType,
        pLimit,
        pSignal
      );

      if (!cChildRes.error) {
        const aChildren = cChildRes.entries
          .map((pE) => fToEntryResult(pE, cChildRes.base))
          .sort((pA, pB) => pA.path.localeCompare(pB.path));

        return {
          args: cArgs,
          base: cRes.base,
          entries: [...entries, ...aChildren],
          exactDir: cExactDir
        };
      }
    }
  }

  return { args: cArgs, base: cRes.base, entries };
}
