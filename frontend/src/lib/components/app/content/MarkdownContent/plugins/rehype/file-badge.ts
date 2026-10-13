/**
 * Rehype plugin that rewrites `file://` markdown anchors into the inline
 * mention chip, sharing the class string with the ChatFormInputRich
 * tokenizer via `$lib/constants`.
 *
 * The chip is presentational: `file://` navigation is blocked from
 * http(s) pages, so the anchor becomes a plain `<span>` (no link role,
 * no tab stop); the full path stays available on `title`.
 */

import {
  cFileUriPrefix,
  cMentionBadgeClassname,
  cMentionBadgeIconClassname,
  cMentionBadgeSvgAttributes,
  cPathSeparator,
  cSettingsKeys
} from '$lib/constants';
import { cSettingsStore, cToolsStore } from '$lib/stores';
import { fDecodeFileLinkPath, fGetMentionBadgeIconPaths, fGetMentionBadgeLabel } from '$lib/utils';
import type { Element, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

// Trailing path separators mark a directory and are kept out of the label.
const cTrailingSeparatorRegex = /\/+$/;

function fDecodeHrefPath(pHref: string): string {
  const cStripped = pHref.startsWith(cFileUriPrefix) ? pHref.slice(cFileUriPrefix.length) : pHref;

  return fDecodeFileLinkPath(cStripped);
}

function fLabelFromFileUrl(pHref: string): string {
  const cDecoded = fDecodeHrefPath(pHref);
  const cTrimmed = cDecoded.replace(cTrailingSeparatorRegex, '');
  const cSlash = cTrimmed.lastIndexOf(cPathSeparator);

  return cSlash === -1 ? cTrimmed : cTrimmed.slice(cSlash + 1);
}

// A trailing `/` in the target marks a directory and selects the folder
// icon, matching the convention the mention picker inserts with.
function fIconElement(pHref: string): Element {
  return {
    children: fGetMentionBadgeIconPaths(pHref).map((pD) => ({
      children: [],
      properties: { d: pD },
      tagName: 'path',
      type: 'element'
    })),
    properties: {
      ...cMentionBadgeSvgAttributes,
      className: cMentionBadgeIconClassname.split(' ').filter(Boolean)
    },
    tagName: 'svg',
    type: 'element'
  };
}

export const fRehypeFileBadge: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (node: Element) => {
      if (node.tagName !== 'a') return;

      const cProps = node.properties ?? {};
      const cHref = typeof cProps.href === 'string' ? cProps.href : null;

      if (!cHref || !cHref.startsWith(cFileUriPrefix)) return;

      const cLabel = fLabelFromFileUrl(cHref);
      const cTitleAttr = typeof cProps.title === 'string' ? cProps.title : cHref;
      const cDecodedPath = fDecodeHrefPath(cHref);

      node.tagName = 'span';
      node.properties = {
        className: cMentionBadgeClassname.split(' ').filter(Boolean),
        title: cTitleAttr.startsWith(cFileUriPrefix) ? cDecodedPath : cTitleAttr
      };
      node.children = [
        fIconElement(cHref),
        {
          children: [
            {
              type: 'text',
              value: fGetMentionBadgeLabel(
                cLabel,
                cDecodedPath,
                cSettingsStore.mGetConfig(cSettingsKeys.SHOW_FULL_PATH_IN_MENTIONS),
                cToolsStore.serverHome
              )
            }
          ],
          properties: { className: ['shrink-0', 'truncate'] },
          tagName: 'span',
          type: 'element'
        }
      ];
    });
  };
};
