/**
 * Rehype plugin to enhance mermaid diagram blocks with wrapper, header, and action buttons.
 *
 * Wraps <pre class="mermaid"> elements with a container that includes:
 * - Language label ("mermaid")
 * - Copy button (copies mermaid syntax to clipboard)
 * - Preview button (opens fullscreen preview dialog)
 *
 * This operates directly on the HAST tree for better performance,
 * avoiding the need to stringify and re-parse HTML.
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
import {
  cDiagramViewModeAttr,
  cDiagramViewRendered,
  cMermaidBlockClass,
  cMermaidIdAttr,
  cMermaidLanguage,
  cMermaidScrollContainerClass,
  cMermaidSyntaxAttr,
  cMermaidWrapperClass
} from '$lib/constants';
import type { Element, ElementContent, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

declare global {
  interface Window {
    idxMermaidBlock?: number;
  }
}

/**
 * Rehype plugin to enhance mermaid diagram blocks with wrapper, header, and action buttons.
 * This plugin wraps <pre class="mermaid"> elements with a container that includes:
 * - Language label ("mermaid")
 * - Copy button
 * - Preview button
 */
export const fRehypeEnhanceMermaidBlocks: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (node: Element, pIndex, pParent) => {
      if (node.tagName !== 'pre' || !pParent || pIndex === undefined) return;

      const cClassName = node.properties?.className;

      if (!Array.isArray(cClassName)) return;

      const cIsMermaid = cClassName.some(
        (pCls) => typeof pCls === 'string' && pCls === cMermaidBlockClass
      );

      if (!cIsMermaid) return;

      const cMermaidId = fGenerateBlockId(cMermaidLanguage, 'idxMermaidBlock');
      // Extract the mermaid syntax (text content of the pre element)
      const cDiagramText = node.children
        .map((pChild) => {
          if (pChild.type === 'text') return pChild.value;

          return '';
        })
        .join('');

      // Store the mermaid syntax in data attribute for copy functionality
      node.properties = {
        ...node.properties,
        [cMermaidIdAttr]: cMermaidId,
        [cMermaidSyntaxAttr]: cDiagramText
      };

      const actions = [
        fCreateCopyButton(cMermaidId, cMermaidIdAttr, 'Copy mermaid syntax'),
        fCreateToggleSourceButton(cMermaidId, cMermaidIdAttr, 'Toggle mermaid source'),
        fCreatePreviewButton(cMermaidId, cMermaidIdAttr, 'Preview diagram')
      ];
      const cHeader = fCreateBlockHeader(cMermaidLanguage, cMermaidId, cMermaidIdAttr, actions);
      const cPreservedCode = (node.data as DiagramPreData | undefined)?.sourceCode;
      const cSourceView = fCreateSourceView(cPreservedCode, cDiagramText, cMermaidLanguage);
      const cWrapper = fCreateWrapper(
        cHeader,
        node,
        cMermaidWrapperClass,
        cMermaidScrollContainerClass,
        {
          [cDiagramViewModeAttr]: cDiagramViewRendered,
          [cMermaidIdAttr]: cMermaidId
        },
        [cSourceView]
      );

      // Replace pre with wrapper in parent
      (pParent.children as ElementContent[])[pIndex] = cWrapper;
    });
  };
};
