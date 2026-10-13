import { fAbbreviateHome, fLastPathSegment } from './path-display';
import {
  cDirectoryPathSuffix,
  cFileUriPrefix,
  aMentionBadgeFileIconPaths,
  aMentionBadgeFolderIconPaths,
  cMentionLinkScanFlags
} from '$lib/constants';
import { FileMentionEntryType } from '$lib/enums';
import type { FileMentionEntry } from '$lib/types';

export {
  cMentionBadgeClassname,
  cMentionBadgeIconClassname,
  cMentionBadgeSvgAttributes,
  aMentionBadgeFileIconPaths,
  aMentionBadgeFolderIconPaths
} from '$lib/constants';

// `)` is allowed in a path only when not followed by whitespace or `[`,
// so macOS paths parse while adjacent badges still terminate the match.
const cFileMentionLinkSource = String.raw`\[([^\]\n]+?)\]\(file:\/\/((?:[^)\n]|\)(?![\s[]))+)\)`;

export function fFileMentionLinkRe(pFlags = ''): RegExp {
  return new RegExp(cFileMentionLinkSource, pFlags);
}

export function fContainsFileMentionLink(pValue: string): boolean {
  return fFileMentionLinkRe().test(pValue);
}

// Escape each path segment for a markdown link destination (spaces/parens
// break CommonMark); keeps the trailing slash that marks a directory.
export function fEncodeFileLinkPath(pPath: string): string {
  return pPath
    .split('/')
    .map((pSegment) => encodeURIComponent(pSegment))
    .join('/');
}

// Malformed escape sequences fall back to the input unchanged.
export function fDecodeFileLinkPath(pPath: string): string {
  try {
    return pPath
      .split('/')
      .map((pSegment) => decodeURIComponent(pSegment))
      .join('/');
  } catch {
    return pPath;
  }
}

export interface MentionTextSegment {
  text: string;
  mention: { name: string; path: string } | null;
}

/**
 * Split raw text into plain runs and `[name](file://path)` mentions.
 * The raw-text renderers walk these segments to draw badges without
 * handing the message to the markdown parser, so a `#` stays a `#`.
 */
export function fSplitMentionSegments(pValue: string): MentionTextSegment[] {
  const cLinkRe = fFileMentionLinkRe(cMentionLinkScanFlags);
  const aSegments: MentionTextSegment[] = [];

  let vCursor = 0;
  let vMatch: RegExpExecArray | null;

  while ((vMatch = cLinkRe.exec(pValue)) !== null) {
    if (vMatch.index > vCursor) {
      aSegments.push({ mention: null, text: pValue.slice(vCursor, vMatch.index) });
    }

    aSegments.push({
      mention: { name: vMatch[1], path: fDecodeFileLinkPath(vMatch[2]) },
      text: vMatch[0]
    });

    vCursor = vMatch.index + vMatch[0].length;
  }

  if (vCursor < pValue.length) aSegments.push({ mention: null, text: pValue.slice(vCursor) });

  return aSegments;
}

export function fGetMentionBadgeIconPaths(pPath: string): readonly string[] {
  return pPath.endsWith(cDirectoryPathSuffix)
    ? aMentionBadgeFolderIconPaths
    : aMentionBadgeFileIconPaths;
}

export function fGetMentionBadgeLabel(
  pName: string,
  pPath: string,
  pShowFullPath: boolean,
  pHome?: string | null
): string {
  if (!pShowFullPath) return pName;

  const cDecoded = fDecodeFileLinkPath(pPath.replace(/\/+$/, ''));

  if (!cDecoded) return pName;

  return fAbbreviateHome(cDecoded, pHome);
}

/**
 * Build the markdown link that replaces a mention token. Entry `path` is
 * already rooted, so `file://` + `/abs` yields the canonical `file:///`.
 * Null when the token is invalid.
 */
export function fBuildMentionInsertion(
  pEntry: FileMentionEntry,
  pValue: string,
  pToken: { start: number; end: number }
): { newValue: string; caretOffset: number } | null {
  if (pToken.start < 0 || pToken.end > pValue.length || pToken.start > pToken.end) return null;

  // Strip the entry's directory marker so it is not doubled below.
  const cCleanedPath = pEntry.path.replace(/\/+$/, '');
  const cPathWithSeparator =
    pEntry.type === FileMentionEntryType.cDirectory ? `${cCleanedPath}/` : cCleanedPath;
  const cBasename = fLastPathSegment(cCleanedPath) || pEntry.name;
  const cInsertion = `[${cBasename}](${cFileUriPrefix}${fEncodeFileLinkPath(cPathWithSeparator)}) `;
  const cNewValue = pValue.slice(0, pToken.start) + cInsertion + pValue.slice(pToken.end);

  return { caretOffset: pToken.start + cInsertion.length, newValue: cNewValue };
}
