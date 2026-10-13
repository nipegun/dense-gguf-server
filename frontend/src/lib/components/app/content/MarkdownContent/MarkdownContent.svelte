<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import '$lib/styles/katex-custom.scss';
  import { fGetMarkdownProcessor, type MarkdownProcessor } from './markdown-processor';
  import {
    fGetCodeInfoFromTarget,
    fGetHastNodeId,
    fGetMdastNodeHash,
    fIsAppendMode
  } from './markdown-utils';
  import {
    ActionIconCopyToClipboard,
    CodeBlockActions,
    DialogCodePreview,
    DialogMermaidPreview
  } from '$lib/components/app';
  import {
    cCodeBlockClass,
    cDiagramViewModeAttr,
    cDiagramViewRendered,
    cDiagramViewSource,
    cImageNotErrorBoundSelector,
    cMarkdownDataAttrs,
    cMermaidBlockClass,
    cMermaidLanguage,
    cMermaidRenderedAttr,
    cMermaidSyntaxAttr,
    cMermaidWrapperClass,
    cSettingsKeys,
    SVG,
    cToggleSourceBtnClass,
    cUiDataAttrs
  } from '$lib/constants';
  import { BooleanString, ColorMode, UrlProtocol } from '$lib/enums';
  import { fCreateAutoScrollController } from '$lib/hooks/use-auto-scroll.svelte';
  import { cSettingsStore } from '$lib/stores';
  import type { DatabaseMessageExtra } from '$lib/types/database';
  import {
    fCopyCodeToClipboard,
    fCopyToClipboard,
    fGetImageErrorFallbackHtml,
    fPreprocessLaTeX,
    fSplitGluedClosingCodeFences
  } from '$lib/utils';
  import { fDetectIncompleteCodeBlock, fHighlightCode, type IncompleteCodeBlock } from '$lib/utils';
  import { fSanitizeSvg } from '$lib/utils/sanitize-svg';
  import { fMountSvgShadow } from '$lib/utils/svg-shadow';
  import type { Root as HastRoot, RootContent as HastRootContent } from 'hast';
  import githubLightCss from 'highlight.js/styles/github.css?inline';
  import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
  import type { Root as MdastRoot } from 'mdast';
  import { mode } from 'mode-watcher';
  import { onDestroy, tick } from 'svelte';
  import { SvelteMap } from 'svelte/reactivity';

  interface Props {
    attachments?: DatabaseMessageExtra[];
    content: string;
    class?: string;
    disableMath?: boolean;
  }

  interface MarkdownBlock {
    id: string;
    html: string;
    contentHash?: string;
  }

  let { attachments, class: className = '', content, disableMath = false }: Props = $props();

  let vContainerRef = $state<HTMLDivElement>();
  let aRenderedBlocks = $state<MarkdownBlock[]>([]);
  let vUnstableBlockHtml = $state('');
  let vIncompleteCodeBlock = $state<IncompleteCodeBlock | null>(null);
  const cStreamingSvgCode = $derived.by(() => {
    const cBlock = vIncompleteCodeBlock;

    if (!cBlock) return null;

    if (cBlock.language === SVG.LANGUAGE) return cBlock.code;

    if (cBlock.language === SVG.XML_LANGUAGE && cBlock.code.trimStart().startsWith(SVG.TAG_PREFIX))
      return cBlock.code;

    return null;
  });
  const cLiveSvgHtml = $derived(cStreamingSvgCode !== null ? fSanitizeSvg(cStreamingSvgCode) : '');

  // Derived rather than called inline in the template so it only recomputes when
  // the block actually changes. Auto-detection is disabled while streaming: it
  // costs ~38ms a call and re-guesses the language on every chunk.
  const cStreamingCodeHtml = $derived(
    vIncompleteCodeBlock
      ? fHighlightCode(vIncompleteCodeBlock.code, vIncompleteCodeBlock.language || 'text', false)
      : ''
  );
  let vPreviewDialogOpen = $state(false);
  let vPreviewCode = $state('');
  let vPreviewLanguage = $state('text');
  let vMermaidPreviewOpen = $state(false);
  let vMermaidPreviewSvgHtml = $state('');
  let vSvgPreviewLive = $state(false);
  let vStreamingSvgHost = $state<HTMLDivElement | null>(null);

  // While the zoom dialog is open on a streaming svg, mirror the live render into it
  $effect(() => {
    if (vSvgPreviewLive && cLiveSvgHtml) vMermaidPreviewSvgHtml = cLiveSvgHtml;
  });

  // Mount the streaming svg into its shadow host on every chunk so it renders live
  $effect(() => {
    if (vStreamingSvgHost)
      fMountSvgShadow(vStreamingSvgHost, cLiveSvgHtml, SVG.INLINE_SHADOW_STYLE);
  });

  let vStreamingCodeScrollContainer = $state<HTMLDivElement>();

  // Auto-scroll controller for streaming code block content
  const cStreamingAutoScroll = fCreateAutoScrollController();

  let vPendingMarkdown: string | null = null;
  let vIsProcessing = false;

  // Per-instance transform cache, avoids re-transforming stable blocks during streaming
  // Garbage collected when component is destroyed (on conversation change)
  const cTransformCache = new SvelteMap<string, string>();
  let vPreviousContent = '';

  /**
   * Removes click event listeners from copy and preview buttons.
   * Called on component destroy.
   */
  function fCleanupEventListeners() {
    if (!vContainerRef) return;

    const cCopyButtons = vContainerRef.querySelectorAll<HTMLButtonElement>('.copy-code-btn');
    const cPreviewButtons = vContainerRef.querySelectorAll<HTMLButtonElement>('.preview-code-btn');

    for (const cButton of cCopyButtons) {
      cButton.removeEventListener('click', fHandleCopyClick);
    }

    for (const cButton of cPreviewButtons) {
      cButton.removeEventListener('click', fHandlePreviewClick);
    }
  }

  /**
   * Loads the appropriate highlight.js theme based on dark/light mode.
   * One shared style element for every markdown block, mirroring
   * SyntaxHighlightedCode.svelte. The old per-instance copies duplicated the
   * full theme CSS once per rendered message, which grows without bound in
   * long conversations.
   * @param isDark - Whether to load the dark theme (true) or light theme (false)
   */
  function fLoadHighlightTheme(pIsDark: boolean) {
    document
      .querySelectorAll(`style[${cUiDataAttrs.HIGHLIGHT_THEME_PREVIEW}]`)
      .forEach((pStyle) => pStyle.remove());

    const cStyle = document.createElement('style');

    cStyle.setAttribute(cUiDataAttrs.HIGHLIGHT_THEME_PREVIEW, BooleanString.cTrue);
    cStyle.textContent = pIsDark ? githubDarkCss : githubLightCss;

    document.head.appendChild(cStyle);
  }

  /**
	 * Transforms a single MDAST node to HTML string with caching.

	/**
	 * Transforms a single MDAST node to HTML string with caching.
	 * Runs the full remark/rehype plugin pipeline (GFM, math, syntax highlighting, etc.)
	 * on an isolated single-node tree, then stringifies the resulting HAST to HTML.
	 * Results are cached by node position hash for streaming performance.
	 * @param processorInstance - The remark/rehype processor instance
	 * @param node - The MDAST node to transform
	 * @param index - Node index for hash fallback
	 * @returns Object containing the HTML string and cache hash
	 */
  async function fTransformMdastNode(
    pProcessorInstance: MarkdownProcessor,
    node: unknown,
    pIndex: number
  ): Promise<{ html: string; hash: string }> {
    const cHash = fGetMdastNodeHash(node, pIndex);
    const cCached = cTransformCache.get(cHash);

    if (cCached) {
      return { hash: cHash, html: cCached };
    }

    const cSingleNodeRoot = { children: [node], type: 'root' };
    const cTransformedRoot = (await pProcessorInstance.mRun(
      cSingleNodeRoot as MdastRoot
    )) as HastRoot;
    const html = pProcessorInstance.mStringify(cTransformedRoot);

    cTransformCache.set(cHash, html);

    return { hash: cHash, html };
  }

  /**
   * Handles click events on copy buttons within code blocks.
   * Copies the raw code content to the clipboard.
   * @param event - The click event from the copy button
   */
  async function fHandleCopyClick(pEvent: Event) {
    pEvent.preventDefault();
    pEvent.stopPropagation();

    const cTarget = pEvent.currentTarget as HTMLButtonElement | null;

    if (!cTarget) {
      return;
    }

    const info = fGetCodeInfoFromTarget(cTarget);

    if (!info) {
      return;
    }

    try {
      await fCopyCodeToClipboard(info.rawCode);
    } catch (error) {
      console.error('Failed to copy code:', error);
    }
  }

  /**
   * Handles preview dialog open state changes.
   * Clears preview content when dialog is closed.
   * @param open - Whether the dialog is being opened or closed
   */
  function fHandlePreviewDialogOpenChange(pOpen: boolean) {
    vPreviewDialogOpen = pOpen;

    if (!pOpen) {
      vPreviewCode = '';
      vPreviewLanguage = 'text';
    }
  }

  /**
   * Handles click events on preview buttons within HTML code blocks.
   * Opens a preview dialog with the rendered HTML content.
   * @param event - The click event from the preview button
   */
  function fHandlePreviewClick(pEvent: Event) {
    pEvent.preventDefault();
    pEvent.stopPropagation();

    const cTarget = pEvent.currentTarget as HTMLButtonElement | null;

    if (!cTarget) {
      return;
    }

    const info = fGetCodeInfoFromTarget(cTarget);

    if (!info) {
      return;
    }

    vPreviewCode = info.rawCode;
    vPreviewLanguage = info.language;
    vPreviewDialogOpen = true;
  }

  /**
   * Processes markdown content into stable and unstable HTML blocks.
   * Uses incremental rendering: stable blocks are cached, unstable block is re-rendered.
   * Incomplete code blocks are rendered using SyntaxHighlightedCode to maintain interactivity.
   * @param markdown - The raw markdown string to process
   */
  async function fProcessMarkdown(pRawMarkdown: string) {
    // Text glued to a closing code fence is not a fence to the parser -
    // the block would swallow it. Split it onto its own line first.
    const markdown = fSplitGluedClosingCodeFences(pRawMarkdown);

    // Early exit if content unchanged (can happen with rapid coalescing)
    if (markdown === vPreviousContent) {
      return;
    }

    if (!markdown) {
      aRenderedBlocks = [];
      vUnstableBlockHtml = '';
      vIncompleteCodeBlock = null;
      vPreviousContent = '';

      return;
    }

    // Check for incomplete code block at the end of content
    const cIncompleteBlock = fDetectIncompleteCodeBlock(markdown);

    if (cIncompleteBlock) {
      // Process only the prefix (content before the incomplete code block)
      const cPrefixMarkdown = markdown.slice(0, cIncompleteBlock.openingIndex);

      if (cPrefixMarkdown.trim()) {
        const cNormalizedPrefix = fPreprocessLaTeX(cPrefixMarkdown);
        const cProcessorInstance = fGetMarkdownProcessor({ attachments, disableMath });
        const cAst = cProcessorInstance.mParse(cNormalizedPrefix) as MdastRoot;
        const aMdastChildren = (cAst as { children?: unknown[] }).children ?? [];
        const aNextBlocks: MarkdownBlock[] = [];
        // Check if we're in append mode for cache reuse
        const cAppendMode = fIsAppendMode(cPrefixMarkdown, vPreviousContent);
        const cPreviousBlockCount = cAppendMode ? aRenderedBlocks.length : 0;

        // All prefix blocks are now stable since code block is separate
        for (let vIndex = 0; vIndex < aMdastChildren.length; vIndex++) {
          const cChild = aMdastChildren[vIndex];

          // In append mode, reuse previous blocks if unchanged
          if (cAppendMode && vIndex < cPreviousBlockCount) {
            const cPrevBlock = aRenderedBlocks[vIndex];
            const cCurrentHash = fGetMdastNodeHash(cChild, vIndex);

            if (cPrevBlock?.contentHash === cCurrentHash) {
              aNextBlocks.push(cPrevBlock);

              continue;
            }
          }

          // Transform this block (with caching)
          const { hash: vHash, html: vHtml } = await fTransformMdastNode(
            cProcessorInstance,
            cChild,
            vIndex
          );
          const cId = fGetHastNodeId(
            { position: (cChild as { position?: unknown }).position } as HastRootContent,
            vIndex
          );

          aNextBlocks.push({ contentHash: vHash, html: vHtml, id: cId });
        }

        aRenderedBlocks = aNextBlocks;
      } else {
        aRenderedBlocks = [];
      }

      vPreviousContent = cPrefixMarkdown;
      vUnstableBlockHtml = '';
      vIncompleteCodeBlock = cIncompleteBlock;

      return;
    }

    // No incomplete code block - use standard processing
    vIncompleteCodeBlock = null;

    const cNormalized = fPreprocessLaTeX(markdown);
    const cProcessorInstance = fGetMarkdownProcessor({ attachments, disableMath });
    const cAst = cProcessorInstance.mParse(cNormalized) as MdastRoot;
    const aMdastChildren = (cAst as { children?: unknown[] }).children ?? [];
    const cStableCount = Math.max(aMdastChildren.length - 1, 0);
    const aNextBlocks: MarkdownBlock[] = [];
    // Check if we're in append mode for cache reuse
    const cAppendMode = fIsAppendMode(markdown, vPreviousContent);
    const cPreviousBlockCount = cAppendMode ? aRenderedBlocks.length : 0;

    for (let vIndex = 0; vIndex < cStableCount; vIndex++) {
      const cChild = aMdastChildren[vIndex];

      // In append mode, reuse previous blocks if unchanged
      if (cAppendMode && vIndex < cPreviousBlockCount) {
        const cPrevBlock = aRenderedBlocks[vIndex];
        const cCurrentHash = fGetMdastNodeHash(cChild, vIndex);

        if (cPrevBlock?.contentHash === cCurrentHash) {
          aNextBlocks.push(cPrevBlock);

          continue;
        }
      }

      // Transform this block (with caching)
      const { hash: vHash, html: vHtml } = await fTransformMdastNode(
        cProcessorInstance,
        cChild,
        vIndex
      );
      const cId = fGetHastNodeId(
        { position: (cChild as { position?: unknown }).position } as HastRootContent,
        vIndex
      );

      aNextBlocks.push({ contentHash: vHash, html: vHtml, id: cId });
    }

    let vUnstableHtml = '';

    if (aMdastChildren.length > cStableCount) {
      const cUnstableChild = aMdastChildren[cStableCount];
      const cSingleNodeRoot = { children: [cUnstableChild], type: 'root' };
      const cTransformedRoot = (await cProcessorInstance.mRun(
        cSingleNodeRoot as MdastRoot
      )) as HastRoot;

      vUnstableHtml = cProcessorInstance.mStringify(cTransformedRoot);
    }

    aRenderedBlocks = aNextBlocks;
    vPreviousContent = markdown;
    await tick(); // Force DOM sync before updating unstable HTML block
    vUnstableBlockHtml = vUnstableHtml;
  }

  /**
   * Attaches click event listeners to copy and preview buttons in code blocks.
   * Uses data-listener-bound attribute to prevent duplicate bindings.
   */
  function fSetupCodeBlockActions() {
    if (!vContainerRef) return;

    const cWrappers = vContainerRef.querySelectorAll<HTMLElement>('.code-block-wrapper');

    for (const cWrapper of cWrappers) {
      const cCopyButton = cWrapper.querySelector<HTMLButtonElement>('.copy-code-btn');
      const cPreviewButton = cWrapper.querySelector<HTMLButtonElement>('.preview-code-btn');

      if (
        cCopyButton &&
        cCopyButton.getAttribute(cMarkdownDataAttrs.LISTENER_BOUND) !== BooleanString.cTrue
      ) {
        cCopyButton.setAttribute(cMarkdownDataAttrs.LISTENER_BOUND, BooleanString.cTrue);
        cCopyButton.addEventListener('click', fHandleCopyClick);
      }

      if (
        cPreviewButton &&
        cPreviewButton.getAttribute(cMarkdownDataAttrs.LISTENER_BOUND) !== BooleanString.cTrue
      ) {
        cPreviewButton.setAttribute(cMarkdownDataAttrs.LISTENER_BOUND, BooleanString.cTrue);
        cPreviewButton.addEventListener('click', fHandlePreviewClick);
      }
    }
  }

  /**
   * Attaches error handlers to images to show fallback UI when loading fails (e.g., CORS).
   * Uses data-error-bound attribute to prevent duplicate bindings.
   */
  function fSetupImageErrorHandlers() {
    if (!vContainerRef) return;

    const images = vContainerRef.querySelectorAll<HTMLImageElement>(cImageNotErrorBoundSelector);

    for (const cImg of images) {
      cImg.setAttribute(cMarkdownDataAttrs.ERROR_BOUND, BooleanString.cTrue);
      cImg.addEventListener('error', fHandleImageError);
    }
  }

  /**
   * Opens the mermaid diagram in a full-screen preview dialog with zoom/pan support.
   * Also handles copy and preview button clicks for mermaid blocks.
   * Uses event delegation: a single handler on the container.
   */
  async function fHandleMermaidClick(pEvent: MouseEvent) {
    const cTarget = pEvent.target as HTMLElement;
    // Toggle a diagram block between its rendered view and its source view.
    // Shared by mermaid and svg, css drives the visibility from the wrapper mode.
    const cToggleBtn = cTarget.closest(`.${cToggleSourceBtnClass}`);

    if (cToggleBtn) {
      pEvent.preventDefault();
      pEvent.stopPropagation();

      const cWrapper = cToggleBtn.closest(`.${cMermaidWrapperClass}, .${SVG.WRAPPER_CLASS}`);

      if (!cWrapper) return;

      const cIsSource = cWrapper.getAttribute(cDiagramViewModeAttr) === cDiagramViewSource;
      const cNext = cIsSource ? cDiagramViewRendered : cDiagramViewSource;

      cWrapper.setAttribute(cDiagramViewModeAttr, cNext);
      cToggleBtn.setAttribute('aria-pressed', String(!cIsSource));

      return;
    }

    // Check if clicking on copy or preview button in mermaid block
    const cCopyBtn = cTarget.closest(`.${cMermaidWrapperClass} .copy-code-btn`);
    const cPreviewBtn = cTarget.closest(`.${cMermaidWrapperClass} .preview-code-btn`);

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

        vMermaidPreviewSvgHtml = svg.outerHTML;
        vSvgPreviewLive = false;
        vMermaidPreviewOpen = true;

        return;
      }
    }

    // Check if clicking on copy or preview button in svg block
    const cSvgCopyBtn = cTarget.closest(`.${SVG.WRAPPER_CLASS} .copy-code-btn`);
    const cSvgPreviewBtn = cTarget.closest(`.${SVG.WRAPPER_CLASS} .preview-code-btn`);

    if (cSvgCopyBtn || cSvgPreviewBtn) {
      const cWrapper = cTarget.closest(`.${SVG.WRAPPER_CLASS}`);

      if (!cWrapper) return;

      const cPreElement = cWrapper.querySelector<HTMLElement>(
        `pre.${SVG.BLOCK_CLASS}[${SVG.SOURCE_ATTR}]`
      );

      if (!cPreElement) return;

      if (cSvgCopyBtn) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
        try {
          await fCopyToClipboard(cPreElement.getAttribute(SVG.SOURCE_ATTR) ?? '');
        } catch (error) {
          console.error('Failed to copy svg source:', error);
        }

        return;
      }

      if (cSvgPreviewBtn) {
        pEvent.preventDefault();
        pEvent.stopPropagation();
        vMermaidPreviewSvgHtml = fSanitizeSvg(cPreElement.getAttribute(SVG.SOURCE_ATTR) ?? '');
        vSvgPreviewLive = false;
        vMermaidPreviewOpen = true;

        return;
      }
    }

    // A click on the header chrome targets the action buttons, never the
    // diagram. Guard so a header click can not fall through to the click to
    // zoom branches below, whatever the scroll position or stacking.
    if (cTarget.closest(`.${cCodeBlockClass.HEADER}`)) return;

    // Open preview when clicking the svg block itself. A final block carries its
    // source, a streaming block does not and is mirrored live into the dialog.
    const cSvgEl = cTarget.closest(`.${SVG.BLOCK_CLASS}`);

    if (cSvgEl) {
      const cSource = cSvgEl.getAttribute(SVG.SOURCE_ATTR);

      if (cSource !== null) {
        vMermaidPreviewSvgHtml = fSanitizeSvg(cSource);
        vSvgPreviewLive = false;
      } else {
        vSvgPreviewLive = true;
      }

      vMermaidPreviewOpen = true;

      return;
    }

    // Otherwise, open preview when clicking on the mermaid diagram itself
    const cMermaidEl = cTarget.closest(`.${cMermaidBlockClass}`);

    if (!cMermaidEl) return;

    const svg = cMermaidEl.querySelector('svg');

    if (!svg) return;

    vMermaidPreviewSvgHtml = svg.outerHTML;
    vSvgPreviewLive = false;
    vMermaidPreviewOpen = true;
  }

  /**
   * Handles mermaid preview dialog open state changes.
   * Cleans up SVG content when dialog is closed.
   */
  function fHandleMermaidPreviewOpenChange(pOpen: boolean) {
    vMermaidPreviewOpen = pOpen;

    if (!pOpen) {
      vMermaidPreviewSvgHtml = '';
      vSvgPreviewLive = false;
    }
  }

  /**
   * Renders mermaid diagrams that haven't been rendered yet.
   * Called after each markdown content update.
   * Marks nodes immediately to prevent duplicate renders during streaming.
   * Reads mode.current before await to ensure reactive tracking.
   */
  async function fRenderMermaidDiagrams() {
    if (!vContainerRef) return;

    const cNodes = vContainerRef.querySelectorAll(
      `pre.${cMermaidBlockClass}:not([${cMermaidRenderedAttr}])`
    );

    if (cNodes.length === 0) return;

    // Mark nodes immediately to prevent duplicate renders if called again during streaming.
    // This avoids needing a guard that would block node discovery.
    cNodes.forEach((node) => node.setAttribute(cMermaidRenderedAttr, BooleanString.cTrue));

    // Read mode before await so Svelte tracks it reactively.
    const cIsDark = mode.current === ColorMode.DARK;
    // lazy load the mermaid dependecy only when needed to reduce bundle size.
    const { default: vMermaid } = await import('mermaid');

    vMermaid.initialize({
      flowchart: {
        htmlLabels: true,
        useMaxWidth: false
      },
      gantt: {
        useMaxWidth: false
      },
      securityLevel: 'strict',
      sequence: {
        useMaxWidth: false
      },
      startOnLoad: false,
      theme: cIsDark ? 'dark' : 'default'
    });

    try {
      await vMermaid.run({
        nodes: Array.from(cNodes) as unknown as NodeListOf<HTMLElement>
      });
    } catch (error) {
      console.error('Failed to render mermaid diagram:', error);
    }
  }

  /**
   * Renders svg diagrams that haven't been rendered yet.
   * Sanitizes the source before injecting and marks each node so it renders once.
   * An empty sanitize result keeps the raw source as escaped text.
   */
  function fRenderSvgDiagrams() {
    if (!vContainerRef) return;

    const cNodes = vContainerRef.querySelectorAll<HTMLElement>(
      `pre.${SVG.BLOCK_CLASS}:not([${SVG.RENDERED_ATTR}])`
    );

    if (cNodes.length === 0) return;

    cNodes.forEach((node) => {
      node.setAttribute(SVG.RENDERED_ATTR, BooleanString.cTrue);

      const cSource = node.getAttribute(SVG.SOURCE_ATTR) ?? node.textContent ?? '';
      const cClean = fSanitizeSvg(cSource);

      if (cClean) {
        node.textContent = '';
        const cHost = document.createElement('div');

        node.appendChild(cHost);
        fMountSvgShadow(cHost, cClean, SVG.INLINE_SHADOW_STYLE);
      }
    });
  }

  /**
   * Handles image load errors by replacing the image with a fallback UI.
   * Shows a placeholder with a link to open the image in a new tab.
   */
  function fHandleImageError(pEvent: Event) {
    const cImg = pEvent.target as HTMLImageElement;

    if (!cImg || !cImg.src) return;

    // Don't handle data URLs or already-handled images
    if (
      cImg.src.startsWith(UrlProtocol.DATA) ||
      cImg.getAttribute(cMarkdownDataAttrs.ERROR_HANDLED) === BooleanString.cTrue
    )
      return;

    cImg.setAttribute(cMarkdownDataAttrs.ERROR_HANDLED, BooleanString.cTrue);

    const cSrc = cImg.src;
    // Create fallback element
    const cFallback = document.createElement('div');

    cFallback.className = 'image-load-error';
    cFallback.innerHTML = fGetImageErrorFallbackHtml(cSrc);

    // Replace image with fallback
    cImg.parentNode?.replaceChild(cFallback, cImg);
  }

  /**
   * Queues markdown for processing with coalescing support.
   * Only processes the latest markdown when multiple updates arrive quickly.
   * Uses requestAnimationFrame to yield to browser paint between batches.
   * @param markdown - The markdown content to render
   */
  async function fUpdateRenderedBlocks(pMarkdown: string) {
    vPendingMarkdown = pMarkdown;

    if (vIsProcessing) {
      return;
    }

    vIsProcessing = true;

    try {
      while (vPendingMarkdown !== null) {
        const cNextMarkdown = vPendingMarkdown;

        vPendingMarkdown = null;

        await fProcessMarkdown(cNextMarkdown);

        // Yield to browser for paint. During this, new chunks coalesce
        // into pendingMarkdown, so we always render the latest state.
        if (vPendingMarkdown !== null) {
          await new Promise((pResolve) => requestAnimationFrame(pResolve));
        }
      }
    } catch (error) {
      console.error('Failed to process markdown:', error);
      aRenderedBlocks = [];
      vUnstableBlockHtml = pMarkdown.replace(/\n/g, '<br>');
    } finally {
      vIsProcessing = false;
    }
  }

  $effect(() => {
    const cCurrentMode = mode.current;
    const cIsDark = cCurrentMode === ColorMode.DARK;

    fLoadHighlightTheme(cIsDark);
  });

  $effect(() => {
    fUpdateRenderedBlocks(content);
  });

  $effect(() => {
    const cHasRenderedBlocks = aRenderedBlocks.length > 0;
    const cHasUnstableBlock = Boolean(vUnstableBlockHtml);

    if ((cHasRenderedBlocks || cHasUnstableBlock) && vContainerRef) {
      fSetupCodeBlockActions();
      fSetupImageErrorHandlers();
      fRenderMermaidDiagrams();
      fRenderSvgDiagrams();
    }
  });

  // Auto-scroll for streaming code block
  $effect(() => {
    cStreamingAutoScroll.mSetContainer(vStreamingCodeScrollContainer);
  });

  $effect(() => {
    cStreamingAutoScroll.mUpdateInterval(vIncompleteCodeBlock !== null);
  });

  onDestroy(() => {
    fCleanupEventListeners();
    cStreamingAutoScroll.mDestroy();
  });
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  bind:this={vContainerRef}
  class="markdown-content {className}{cSettingsStore.vConfig[cSettingsKeys.FULL_HEIGHT_CODE_BLOCKS]
    ? ' full-height-code-blocks'
    : ''}"
  onclick={fHandleMermaidClick}
>
  {#each aRenderedBlocks as vBlock (vBlock.id)}
    <div class="markdown-block" {...{ [cMarkdownDataAttrs.BLOCK_ID]: vBlock.id }}>
      {@html vBlock.html}
    </div>
  {/each}

  {#if vUnstableBlockHtml}
    <div
      class="markdown-block markdown-block--unstable"
      {...{ [cMarkdownDataAttrs.BLOCK_ID]: 'unstable' }}
    >
      <!-- eslint-disable-next-line no-at-html-tags -->
      {@html vUnstableBlockHtml}
    </div>
  {/if}

  {#if vIncompleteCodeBlock}
    {#if vIncompleteCodeBlock.language === cMermaidLanguage}
      <div class="mermaid-block-wrapper streaming-mermaid-block">
        <div class="code-block-header">
          <span class="code-language">{fTranslate('text_0fbccedd6152')}</span>

          <div class="code-block-actions">
            <ActionIconCopyToClipboard
              ariaLabel="Diagram incomplete"
              canCopy={false}
              text={vIncompleteCodeBlock.code}
            />
          </div>
        </div>

        <div class="mermaid-loading-placeholder">
          <span class="mermaid-loading-text">{fTranslate('text_f7f714a03395')}</span>
        </div>
      </div>
    {:else if cStreamingSvgCode !== null}
      <div class="svg-block-wrapper streaming-svg-block">
        <div class="code-block-header">
          <span class="code-language">{fTranslate('text_acdb1373d176')}</span>

          <div class="code-block-actions">
            <ActionIconCopyToClipboard
              ariaLabel="Diagram incomplete"
              canCopy={false}
              text={vIncompleteCodeBlock.code}
            />
          </div>
        </div>

        {#if cLiveSvgHtml}
          <div class="svg-scroll-container">
            <div class={SVG.BLOCK_CLASS}>
              <div bind:this={vStreamingSvgHost}></div>
            </div>
          </div>
        {:else}
          <div class="mermaid-loading-placeholder">
            <span class="mermaid-loading-text">{fTranslate('text_6702e91a7860')}</span>
          </div>
        {/if}
      </div>
    {:else}
      <div class="code-block-wrapper streaming-code-block relative">
        <div class="code-block-header">
          <span class="code-language">{vIncompleteCodeBlock.language || 'text'}</span>

          <CodeBlockActions
            code={vIncompleteCodeBlock.code}
            disabled
            language={vIncompleteCodeBlock.language || 'text'}
            onPreview={(pCode, pLang) => {
              vPreviewCode = pCode;
              vPreviewLanguage = pLang;
              vPreviewDialogOpen = true;
            }}
          />
        </div>

        <div
          bind:this={vStreamingCodeScrollContainer}
          class="streaming-code-scroll-container"
          onscroll={() => cStreamingAutoScroll.mHandleScroll()}
        >
          <pre class="streaming-code-pre"><code
              class="hljs language-{vIncompleteCodeBlock.language || 'text'}"
              >{@html cStreamingCodeHtml}</code
            ></pre>
        </div>
      </div>
    {/if}
  {/if}
</div>

<DialogCodePreview
  code={vPreviewCode}
  language={vPreviewLanguage}
  onOpenChange={fHandlePreviewDialogOpenChange}
  open={vPreviewDialogOpen}
/>

<DialogMermaidPreview
  onOpenChange={fHandleMermaidPreviewOpenChange}
  open={vMermaidPreviewOpen}
  svgHtml={vMermaidPreviewSvgHtml}
/>

<style>
  @import './markdown-content.css';
</style>
