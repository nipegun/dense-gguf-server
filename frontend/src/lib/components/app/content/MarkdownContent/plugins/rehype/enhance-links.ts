/**
 * Rehype plugin to enhance links with security attributes.
 *
 * Adds target="_blank" and rel="noopener noreferrer" to all anchor elements,
 * ensuring external links open in new tabs safely.
 */

import type { Element, Root } from 'hast';
import type { Plugin } from 'unified';
import { visit } from 'unist-util-visit';

/**
 * Rehype plugin that adds security attributes to all links.
 * This plugin ensures external links open in new tabs safely by adding:
 * - target="_blank"
 * - rel="noopener noreferrer"
 */
export const fRehypeEnhanceLinks: Plugin<[], Root> = () => {
  return (pTree: Root) => {
    visit(pTree, 'element', (node: Element) => {
      if (node.tagName !== 'a') return;

      const cProps = node.properties ?? {};

      // Only modify if href exists
      if (!cProps.href) return;

      cProps.target = '_blank';
      cProps.rel = 'noopener noreferrer';
      node.properties = cProps;
    });
  };
};
