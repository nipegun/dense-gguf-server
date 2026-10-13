/**
 * Shared utilities for enhanced code blocks and mermaid diagram blocks.
 * Contains common HAST element creation functions to avoid code duplication.
 */

import {
  cCodeBlockClass,
  cCodeIconSvg,
  cCopyIconSvg,
  cDiagramSourceClass,
  cPreviewIconSvg,
  cToggleSourceBtnClass
} from '$lib/constants';
import type { Element, ElementContent } from 'hast';

export interface BlockIdGenerator {
  (pId: number): string;
}

/**
 * Creates an icon element with the given SVG content.
 */
export function fCreateIconElement(pSvg: string): Element {
  return {
    children: [{ type: 'raw', value: pSvg } as unknown as ElementContent],
    properties: {},
    tagName: 'span',
    type: 'element'
  };
}

/**
 * Creates a button element with icon. Extra properties merge onto the button,
 * which lets a stateful button carry attributes like aria-pressed.
 */
export function fCreateButton(
  pClassName: string,
  pTitle: string,
  pIconSvg: string,
  pId: string,
  pIdAttribute: string,
  pExtraProperties: Record<string, string> = {}
): Element {
  return {
    children: [fCreateIconElement(pIconSvg)],
    properties: {
      className: [pClassName],
      [pIdAttribute]: pId,
      title: pTitle,
      type: 'button',
      ...pExtraProperties
    },
    tagName: 'button',
    type: 'element'
  };
}

/**
 * Creates a copy button element.
 */
export function fCreateCopyButton(
  pId: string,
  pIdAttribute: string,
  pTitle: string = 'Copy'
): Element {
  return fCreateButton(cCodeBlockClass.COPY_BTN, pTitle, cCopyIconSvg, pId, pIdAttribute);
}

/**
 * Creates a preview button element.
 */
export function fCreatePreviewButton(
  pId: string,
  pIdAttribute: string,
  pTitle: string = 'Preview'
): Element {
  return fCreateButton(cCodeBlockClass.PREVIEW_BTN, pTitle, cPreviewIconSvg, pId, pIdAttribute);
}

/**
 * Creates a button that toggles a diagram block between its rendered view and
 * its source view. aria-pressed starts false, the rendered view is the default.
 */
export function fCreateToggleSourceButton(
  pId: string,
  pIdAttribute: string,
  pTitle: string = 'Toggle source'
): Element {
  return fCreateButton(cToggleSourceBtnClass, pTitle, cCodeIconSvg, pId, pIdAttribute, {
    'aria-pressed': 'false'
  });
}

/**
 * Creates a source view for a diagram block. It reuses the code block scroll
 * container so it matches the app code blocks, and wraps the highlighted code
 * element captured at transform time. A missing code element falls back to a
 * plain code node built from the raw source.
 */
export function fCreateSourceView(
  pCodeElement: Element | undefined,
  pSource: string,
  pLanguage: string
): Element {
  const cCode: Element = pCodeElement ?? {
    children: [{ type: 'text', value: pSource }],
    properties: { className: ['hljs', `language-${pLanguage}`] },
    tagName: 'code',
    type: 'element'
  };

  return {
    children: [
      {
        children: [cCode],
        properties: {},
        tagName: 'pre',
        type: 'element'
      }
    ],
    properties: { className: [cDiagramSourceClass, cCodeBlockClass.SCROLL_CONTAINER] },
    tagName: 'div',
    type: 'element'
  };
}

/**
 * Creates a block header with language label and action buttons.
 */
export function fCreateBlockHeader(
  pLanguage: string,
  pId: string,
  pIdAttribute: string,
  actions: Element[],
  pLanguageClassName: string = cCodeBlockClass.LANGUAGE
): Element {
  return {
    children: [
      {
        children: [{ type: 'text', value: pLanguage }],
        properties: { className: [pLanguageClassName] },
        tagName: 'span',
        type: 'element'
      },
      {
        children: actions,
        properties: { className: [cCodeBlockClass.ACTIONS] },
        tagName: 'div',
        type: 'element'
      }
    ],
    properties: { className: [cCodeBlockClass.HEADER] },
    tagName: 'div',
    type: 'element'
  };
}

/**
 * Creates a scroll container element.
 */
export function fCreateScrollContainer(
  pPreElement: Element,
  pScrollContainerClass: string
): Element {
  return {
    children: [pPreElement],
    properties: { className: [pScrollContainerClass] },
    tagName: 'div',
    type: 'element'
  };
}

/**
 * Creates a wrapper element with header and scroll container. Extra children
 * append after the scroll container, which lets a block carry a source view
 * alongside its rendered output.
 */
export function fCreateWrapper(
  pHeader: Element,
  pPreElement: Element,
  pWrapperClass: string,
  pScrollContainerClass: string,
  pAdditionalAttributes?: Record<string, string>,
  pExtraChildren: Element[] = []
): Element {
  return {
    children: [
      pHeader,
      fCreateScrollContainer(pPreElement, pScrollContainerClass),
      ...pExtraChildren
    ],
    properties: {
      className: [pWrapperClass, cCodeBlockClass.RELATIVE],
      ...pAdditionalAttributes
    } as Element['properties'],
    tagName: 'div',
    type: 'element'
  };
}

/**
 * Generates a unique block ID using a global counter.
 */
export function fGenerateBlockId(pPrefix: string, pWindowKey: keyof Window): string {
  if (typeof window !== 'undefined') {
    const cIdx = window[pWindowKey] as number | undefined;
    const cNext = (cIdx ?? 0) + 1;

    (window as unknown as Record<string, number>)[pWindowKey] = cNext;

    return `${pPrefix}-${cNext}`;
  }

  // Fallback for SSR - use timestamp + random
  return `${pPrefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
}
