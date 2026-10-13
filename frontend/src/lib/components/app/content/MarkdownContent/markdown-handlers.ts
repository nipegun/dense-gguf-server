/**
 * Event handler factories for markdown content components.
 * Uses dependency injection pattern to avoid direct component state access.
 */

import {
  cCodeBlockClass,
  cMarkdownDataAttrs,
  cMermaidBlockClass,
  cMermaidSyntaxAttr,
  cMermaidWrapperClass
} from '$lib/constants';
import { BooleanString } from '$lib/enums';
import { fCopyCodeToClipboard, fCopyToClipboard } from '$lib/utils';

export interface PreviewState {
  previewDialogOpen: boolean;
  previewCode: string;
  previewLanguage: string;
  setPreviewDialogOpen: (pOpen: boolean) => void;
  setPreviewCode: (pCode: string) => void;
  setPreviewLanguage: (pLang: string) => void;
}

export interface MermaidPreviewState {
  mermaidPreviewOpen: boolean;
  mermaidPreviewSvgHtml: string;
  setMermaidPreviewOpen: (pOpen: boolean) => void;
  setMermaidPreviewSvgHtml: (pHtml: string) => void;
}

export interface RenderedBlocksState {
  renderedBlocks: Array<{ id: string; html: string; contentHash?: string }>;
  setRenderedBlocks: (pBlocks: Array<{ id: string; html: string; contentHash?: string }>) => void;
}

/**
 * Creates a click handler for copy buttons in code blocks.
 * Copies the code content to clipboard.
 */
export function fCreateHandleCopyClick() {
  return async function handleCopyClick(pEvent: Event) {
    pEvent.preventDefault();
    pEvent.stopPropagation();

    const cTarget = pEvent.currentTarget as HTMLButtonElement | null;

    if (!cTarget) return;

    const cWrapper = cTarget.closest(`.${cCodeBlockClass.WRAPPER}`);

    if (!cWrapper) return;

    const cCodeElement = cWrapper.querySelector<HTMLElement>(`code[${cMarkdownDataAttrs.CODE_ID}]`);

    if (!cCodeElement) return;

    const cRawCode = cCodeElement.textContent ?? '';

    try {
      await fCopyCodeToClipboard(cRawCode);
    } catch (error) {
      console.error('Failed to copy code:', error);
    }
  };
}

/**
 * Creates a handler for preview dialog open state changes.
 * Clears preview content when dialog is closed.
 */
export function fCreateHandlePreviewDialogOpenChange(pPreviewState: PreviewState) {
  return function handlePreviewDialogOpenChange(pOpen: boolean) {
    pPreviewState.setPreviewDialogOpen(pOpen);

    if (!pOpen) {
      pPreviewState.setPreviewCode('');
      pPreviewState.setPreviewLanguage('text');
    }
  };
}

/**
 * Creates a click handler for preview buttons within HTML code blocks.
 * Opens a preview dialog with the rendered HTML content.
 */
export function fCreateHandlePreviewClick(pPreviewState: PreviewState) {
  return async function handlePreviewClick(pEvent: Event) {
    pEvent.preventDefault();
    pEvent.stopPropagation();

    const cTarget = pEvent.currentTarget as HTMLButtonElement | null;

    if (!cTarget) return;

    const cWrapper = cTarget.closest(`.${cCodeBlockClass.WRAPPER}`);

    if (!cWrapper) return;

    const cCodeElement = cWrapper.querySelector<HTMLElement>(`code[${cMarkdownDataAttrs.CODE_ID}]`);

    if (!cCodeElement) return;

    const cRawCode = cCodeElement.textContent ?? '';
    const cLanguageLabel = cWrapper.querySelector<HTMLElement>(`.${cCodeBlockClass.LANGUAGE}`);
    const cLanguage = cLanguageLabel?.textContent?.trim() || 'text';

    pPreviewState.setPreviewCode(cRawCode);
    pPreviewState.setPreviewLanguage(cLanguage);
    pPreviewState.setPreviewDialogOpen(true);
  };
}

/**
 * Creates a click handler for mermaid block interactions.
 * Handles copy, preview, and diagram click events via event delegation.
 */
export function fCreateHandleMermaidClick(pMermaidState: MermaidPreviewState) {
  return async function handleMermaidClick(pEvent: MouseEvent) {
    const cTarget = pEvent.target as HTMLElement;
    // Check if clicking on copy or preview button in mermaid block
    const cCopyBtn = cTarget.closest(`.${cMermaidWrapperClass} .${cCodeBlockClass.COPY_BTN}`);
    const cPreviewBtn = cTarget.closest(`.${cMermaidWrapperClass} .${cCodeBlockClass.PREVIEW_BTN}`);

    if (cCopyBtn || cPreviewBtn) {
      const cWrapper = cTarget.closest(`.${cMermaidWrapperClass}`);

      if (!cWrapper) return;

      const cPreElement = cWrapper.querySelector<HTMLElement>(
        `pre.${cMermaidBlockClass}[${cMermaidSyntaxAttr}]`
      );

      if (!cPreElement) return;

      const cMermaidSyntax = cPreElement.getAttribute(cMermaidSyntaxAttr) ?? '';

      if (cCopyBtn) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
        try {
          await fCopyToClipboard(cMermaidSyntax);
        } catch (error) {
          console.error('Failed to copy mermaid syntax:', error);
        }

        return;
      }

      if (cPreviewBtn) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
        const svg = cPreElement.querySelector('svg');

        if (!svg) return;

        pMermaidState.setMermaidPreviewSvgHtml(svg.outerHTML);
        pMermaidState.setMermaidPreviewOpen(true);

        return;
      }
    }

    // Otherwise, open preview when clicking on the mermaid diagram itself
    const cMermaidEl = cTarget.closest(`.${cMermaidBlockClass}`);

    if (!cMermaidEl) return;

    const svg = cMermaidEl.querySelector('svg');

    if (!svg) return;

    pMermaidState.setMermaidPreviewSvgHtml(svg.outerHTML);
    pMermaidState.setMermaidPreviewOpen(true);
  };
}

/**
 * Creates a handler for mermaid preview dialog open state changes.
 * Cleans up SVG content when dialog is closed.
 */
export function fCreateHandleMermaidPreviewOpenChange(pMermaidState: MermaidPreviewState) {
  return function handleMermaidPreviewOpenChange(pOpen: boolean) {
    pMermaidState.setMermaidPreviewOpen(pOpen);

    if (!pOpen) {
      pMermaidState.setMermaidPreviewSvgHtml('');
    }
  };
}

/**
 * Creates an error handler for images that fail to load (e.g., CORS issues).
 * Shows fallback UI for broken images.
 */
export function fCreateHandleImageError(
  pRenderedBlocksState: RenderedBlocksState,
  pImageNotErrorBoundSelector: string,
  pErrorBoundAttr: string,
  pBooleanString: BooleanString
) {
  return async function handleImageError(pEvent: Event) {
    const cImg = pEvent.target as HTMLImageElement;

    if (!cImg) return;

    const cBlockId = cImg
      .closest(`[${cMarkdownDataAttrs.BLOCK_ID}]`)
      ?.getAttribute(cMarkdownDataAttrs.BLOCK_ID);

    if (!cBlockId) return;

    const cBlock = pRenderedBlocksState.renderedBlocks.find((pB) => pB.id === cBlockId);

    if (!cBlock) return;

    // Skip if already handled
    if (cImg.getAttribute(pErrorBoundAttr) === pBooleanString) return;

    cImg.setAttribute(pErrorBoundAttr, pBooleanString);

    // Get the fallback HTML and replace the image
    const cFallbackHtml = `<div class="image-error-placeholder" ${cMarkdownDataAttrs.ORIGINAL_SRC}="${cImg.src}">
			<span class="image-error-icon">⚠️</span>
			<span class="image-error-text">Failed to load image</span>
		</div>`;
    // Replace the img element with fallback in the block's HTML
    const cNewHtml = cBlock.html.replace(/img[^>]*src=["']([^"']*)[^>]*>/g, (match, pSrc) => {
      if (pSrc === cImg.src) {
        return cFallbackHtml.replace(
          `${cMarkdownDataAttrs.ORIGINAL_SRC}=""`,
          `${cMarkdownDataAttrs.ORIGINAL_SRC}="${pSrc}"`
        );
      }

      return match;
    });
    // Update the block
    const aNewBlocks = pRenderedBlocksState.renderedBlocks.map((pB) =>
      pB.id === cBlockId ? { ...pB, html: cNewHtml } : pB
    );

    pRenderedBlocksState.setRenderedBlocks(aNewBlocks);
  };
}

/**
 * Creates a function to set up code block action event listeners.
 * Binds click handlers to copy and preview buttons within code blocks.
 */
export function fCreateSetupCodeBlockActions(
  pHandleCopyClick: (pEvent: Event) => void,
  pHandlePreviewClick: (pEvent: Event) => void
) {
  return function setupCodeBlockActions(pContainerRef: HTMLElement | null) {
    if (!pContainerRef) return;

    const cWrappers = pContainerRef.querySelectorAll<HTMLElement>(`.${cCodeBlockClass.WRAPPER}`);

    for (const cWrapper of cWrappers) {
      const cCopyButton = cWrapper.querySelector<HTMLButtonElement>(`.${cCodeBlockClass.COPY_BTN}`);
      const cPreviewButton = cWrapper.querySelector<HTMLButtonElement>(
        `.${cCodeBlockClass.PREVIEW_BTN}`
      );

      if (
        cCopyButton &&
        cCopyButton.getAttribute(cMarkdownDataAttrs.LISTENER_BOUND) !== BooleanString.cTrue
      ) {
        cCopyButton.setAttribute(cMarkdownDataAttrs.LISTENER_BOUND, BooleanString.cTrue);
        cCopyButton.addEventListener('click', pHandleCopyClick);
      }

      if (
        cPreviewButton &&
        cPreviewButton.getAttribute(cMarkdownDataAttrs.LISTENER_BOUND) !== BooleanString.cTrue
      ) {
        cPreviewButton.setAttribute(cMarkdownDataAttrs.LISTENER_BOUND, BooleanString.cTrue);
        cPreviewButton.addEventListener('click', pHandlePreviewClick);
      }
    }
  };
}

/**
 * Creates a function to set up image error handlers.
 * Attaches error handlers to images to show fallback UI when loading fails.
 */
export function fCreateSetupImageErrorHandlers(
  pHandleImageError: (pEvent: Event) => void,
  pImageNotErrorBoundSelector: string,
  pErrorBoundAttr: string,
  pBooleanString: BooleanString
) {
  return function setupImageErrorHandlers(pContainerRef: HTMLElement | null) {
    if (!pContainerRef) return;

    const images = pContainerRef.querySelectorAll<HTMLImageElement>(pImageNotErrorBoundSelector);

    for (const cImg of images) {
      cImg.setAttribute(pErrorBoundAttr, pBooleanString);
      cImg.addEventListener('error', pHandleImageError);
    }
  };
}
