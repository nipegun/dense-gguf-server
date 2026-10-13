<script lang="ts">
  import { browser } from '$app/environment';
  import { cSyntaxCodeScrollAtBottomThresholdPx, cUiDataAttrs } from '$lib/constants';
  import { BooleanString, ColorMode } from '$lib/enums';
  import { fHighlightCode } from '$lib/utils';
  import githubLightCss from 'highlight.js/styles/github.css?inline';
  import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
  import { mode } from 'mode-watcher';

  interface Props {
    borderColor?: string;
    code: string;
    language?: string;
    class?: string;
    maxHeight?: string;
    maxWidth?: string;
    /** Auto-scrolls to the bottom of new chunks; pauses on user scroll-up
     *  until the user returns to the bottom. */
    streaming?: boolean;
  }

  let {
    class: className = '',
    code,
    borderColor,
    language = 'text',
    maxHeight = '60vh',
    maxWidth = '',
    streaming = false
  }: Props = $props();

  const cHighlightedHtml = $derived(fHighlightCode(code, language));

  let vScrollEl = $state<HTMLDivElement>();
  let userScrolledUp = $state(false);
  let lastScrollTop = 0;
  const cScrollBottomThresholdPx = cSyntaxCodeScrollAtBottomThresholdPx;
  let vPendingFrame: number | null = null;

  function fLoadHighlightTheme(pIsDark: boolean) {
    if (!browser) return;

    const cExistingThemes = document.querySelectorAll(
      `style[${cUiDataAttrs.HIGHLIGHT_THEME_PREVIEW}]`
    );

    cExistingThemes.forEach((pStyle) => pStyle.remove());

    const cStyle = document.createElement('style');

    cStyle.setAttribute(cUiDataAttrs.HIGHLIGHT_THEME_PREVIEW, BooleanString.cTrue);
    cStyle.textContent = pIsDark ? githubDarkCss : githubLightCss;

    document.head.appendChild(cStyle);
  }

  function fIsAtBottom(): boolean {
    if (!vScrollEl) return false;

    return (
      vScrollEl.scrollHeight - vScrollEl.clientHeight - vScrollEl.scrollTop <=
      cScrollBottomThresholdPx
    );
  }

  function fScrollToBottomOnFrame() {
    if (vPendingFrame !== null || !vScrollEl || userScrolledUp) return;

    vPendingFrame = requestAnimationFrame(() => {
      vPendingFrame = null;

      // User may scroll between scheduling and paint.
      if (vScrollEl && !userScrolledUp) {
        vScrollEl.scrollTop = vScrollEl.scrollHeight;
      }
    });
  }

  function fHandleScrollEvent() {
    if (!vScrollEl) return;

    const cIsScrollingUp = vScrollEl.scrollTop < lastScrollTop;

    if (cIsScrollingUp && !fIsAtBottom()) {
      userScrolledUp = true;
    } else if (fIsAtBottom()) {
      userScrolledUp = false;
    }

    lastScrollTop = vScrollEl.scrollTop;
  }

  $effect(() => {
    const cCurrentMode = mode.current;
    const cIsDark = cCurrentMode === ColorMode.DARK;

    fLoadHighlightTheme(cIsDark);
  });

  // Pin to bottom at the start of each streaming episode.
  $effect(() => {
    if (streaming) {
      userScrolledUp = false;
      lastScrollTop = 0;
    }
  });

  $effect(() => {
    void code;

    if (!streaming || userScrolledUp) return;

    fScrollToBottomOnFrame();
  });

  // Layout shifts that don't change `code` (highlight.js re-tokenize, line-wrap reflow).
  $effect(() => {
    if (!streaming || !vScrollEl) return;

    const cObserver = new MutationObserver(() => fScrollToBottomOnFrame());

    cObserver.observe(vScrollEl, {
      characterData: true,
      childList: true,
      subtree: true
    });

    return () => cObserver.disconnect();
  });
</script>

<div
  bind:this={vScrollEl}
  class="code-preview-wrapper min-w-0 max-w-full overflow-auto rounded-xl border shadow-[0_1px_2px_0_rgb(0_0_0_/_0.05)] {className}"
  onscroll={fHandleScrollEvent}
  style="border-color: {borderColor ??
    'color-mix(in oklch, var(--border) 30%, transparent)'}; background: var(--code-background); max-height: {maxHeight}; {maxWidth
    ? `max-width: ${maxWidth};`
    : ''}"
>
  <!-- Single line: hljs injection depends on a contiguous source string. -->
  <pre class="m-0"><code class="hljs text-sm leading-relaxed">{@html cHighlightedHtml}</code></pre>
</div>

<style>
  .code-preview-wrapper {
    overscroll-behavior: contain;
  }

  .code-preview-wrapper pre {
    background: transparent;
    padding: 0;
  }

  .code-preview-wrapper code {
    background: transparent;
    display: block;
    padding: 0.5rem;
  }

  :global(.dark) .code-preview-wrapper {
    border-color: color-mix(in oklch, var(--border) 20%, transparent);
  }
</style>
