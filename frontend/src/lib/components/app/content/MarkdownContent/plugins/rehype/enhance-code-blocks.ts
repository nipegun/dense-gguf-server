/**
 * Rehype plugin to enhance code blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre><code> elements with a container that includes:
 * - Language label
 * - Copy button
 * - Preview button (for HTML code blocks)
 *
 * This operates directly on the HAST tree for better performance,
 * avoiding the need to stringify and re-parse HTML.
 */

import {
  fCreateBlockHeader,
  fCreateCopyButton,
  fCreatePreviewButton,
  fCreateWrapper,
  fGenerateBlockId
} from './code-block-utils';
import { cCodeBlockClass, cMarkdownDataAttrs } from '$lib/constants';
import type { Element, ElementContent, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

declare global {
  interface Window {
    idxCodeBlock?: number;
  }
}

function fExtractLanguage(pCodeElement: Element): string {
  const cClassName = pCodeElement.properties?.className;

  if (!Array.isArray(cClassName)) return 'text';

  for (const cCls of cClassName) {
    if (typeof cCls === 'string' && cCls.startsWith('language-')) {
      return cCls.replace('language-', '');
    }
  }

  return 'text';
}

/**
 * Rehype plugin to enhance code blocks with wrapper, header, and action buttons.
 * This plugin wraps <pre><code> elements with a container that includes:
 * - Language label
 * - Copy button
 * - Preview button (for HTML code blocks)
 */
export const fRehypeEnhanceCodeBlocks: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (node: Element, pIndex, pParent) => {
      if (node.tagName !== 'pre' || !pParent || pIndex === undefined) return;

      const cCodeElement = node.children.find(
        (pChild): pChild is Element => pChild.type === 'element' && pChild.tagName === 'code'
      );

      if (!cCodeElement) return;

      const cLanguage = fExtractLanguage(cCodeElement);
      const cCodeId = fGenerateBlockId('code', 'idxCodeBlock');

      cCodeElement.properties = {
        ...cCodeElement.properties,
        [cMarkdownDataAttrs.CODE_ID]: cCodeId
      };

      const actions: Element[] = [
        fCreateCopyButton(cCodeId, cMarkdownDataAttrs.CODE_ID, 'Copy code')
      ];

      if (cLanguage.toLowerCase() === 'html') {
        actions.push(fCreatePreviewButton(cCodeId, cMarkdownDataAttrs.CODE_ID, 'Preview code'));
      }

      const cHeader = fCreateBlockHeader(cLanguage, cCodeId, cMarkdownDataAttrs.CODE_ID, actions);
      const cWrapper = fCreateWrapper(
        cHeader,
        node,
        cCodeBlockClass.WRAPPER,
        cCodeBlockClass.SCROLL_CONTAINER
      );

      // Replace pre with wrapper in parent
      (pParent.children as ElementContent[])[pIndex] = cWrapper;
    });
  };
};
