/**
 * Rehype plugin to enhance svg blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre class="svg-block"> elements with a container that includes:
 * - Language label ("svg")
 * - Copy button (copies svg source to clipboard)
 * - Preview button (opens fullscreen preview dialog)
 *
 * Operates directly on the HAST tree and reuses the shared code-block builders.
 */

import {
  fCreateBlockHeader,
  fCreateCopyButton,
  fCreatePreviewButton,
  fCreateSourceView,
  fCreateToggleSourceButton,
  fCreateWrapper,
  fGenerateBlockId
} from './code-block-utils';
import type { DiagramPreData } from './pre-transform';
import { cDiagramViewModeAttr, cDiagramViewRendered, SVG } from '$lib/constants';
import type { Element, ElementContent, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

declare global {
  interface Window {
    idxSvgBlock?: number;
  }
}

export const fRehypeEnhanceSvgBlocks: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (node: Element, pIndex, pParent) => {
      if (node.tagName !== 'pre' || !pParent || pIndex === undefined) return;

      const cClassName = node.properties?.className;

      if (!Array.isArray(cClassName)) return;

      const cIsSvg = cClassName.some(
        (pCls) => typeof pCls === 'string' && pCls === SVG.BLOCK_CLASS
      );

      if (!cIsSvg) return;

      const cSvgId = fGenerateBlockId(SVG.LANGUAGE, 'idxSvgBlock');
      // Extract the svg source (text content of the pre element)
      const cSvgSource = node.children
        .map((pChild) => {
          if (pChild.type === 'text') return pChild.value;

          return '';
        })
        .join('');

      // Store the svg source in data attribute for copy and render
      node.properties = {
        ...node.properties,
        [SVG.ID_ATTR]: cSvgId,
        [SVG.SOURCE_ATTR]: cSvgSource
      };

      const actions = [
        fCreateCopyButton(cSvgId, SVG.ID_ATTR, 'Copy svg source'),
        fCreateToggleSourceButton(cSvgId, SVG.ID_ATTR, 'Toggle svg source'),
        fCreatePreviewButton(cSvgId, SVG.ID_ATTR, 'Preview svg')
      ];
      const cHeader = fCreateBlockHeader(SVG.LANGUAGE, cSvgId, SVG.ID_ATTR, actions);
      const cPreservedCode = (node.data as DiagramPreData | undefined)?.sourceCode;
      const cSourceView = fCreateSourceView(cPreservedCode, cSvgSource, SVG.LANGUAGE);
      const cWrapper = fCreateWrapper(
        cHeader,
        node,
        SVG.WRAPPER_CLASS,
        SVG.SCROLL_CONTAINER_CLASS,
        {
          [cDiagramViewModeAttr]: cDiagramViewRendered,
          [SVG.ID_ATTR]: cSvgId
        },
        [cSourceView]
      );

      // Replace pre with wrapper in parent
      (pParent.children as ElementContent[])[pIndex] = cWrapper;
    });
  };
};
