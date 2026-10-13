/**
 * Rehype plugin to wrap tables in a horizontal scroll container.
 *
 * A bare <table> keeps its content-driven minimum width, which propagates up
 * the layout and can stretch the chat column past the window. Wrapping in
 * div.table-wrapper makes the wrapper the scroll container (styled in
 * markdown-content.css), so wide tables scroll in place instead.
 */

import type { Element, ElementContent, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

export const fRehypeEnhanceTables: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (node: Element, pIndex, pParent) => {
      if (node.tagName !== 'table' || !pParent || pIndex === undefined) return;

      // already wrapped (e.g. nested tables in raw HTML input)
      const cParentClass = pParent.type === 'element' ? pParent.properties?.className : undefined;

      if (Array.isArray(cParentClass) && cParentClass.includes('table-wrapper')) return;

      const cWrapper: Element = {
        children: [node as ElementContent],
        properties: { className: ['table-wrapper'] },
        tagName: 'div',
        type: 'element'
      };

      pParent.children[pIndex] = cWrapper;
    });
  };
};
