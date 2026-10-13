<script lang="ts">
  import { cCodeBlock, cCodeTokenAttr, cUiDataAttrs } from '$lib/constants';
  import { BooleanString, ChatFormInputRichTokenKind, ColorMode } from '$lib/enums';
  import { cDeviceStore } from '$lib/stores';
  import type { ChatFormInputRichToken } from '$lib/types';
  import type { SourceHistoryEntry } from '$lib/utils';
  import {
    fBadgeAwareWordJump,
    fBuildFragment,
    fDomMatchesTokens,
    fHighlightCode,
    fIsIMEComposing,
    fIsOffsetInCodeBlock,
    fLeadingBadgeEdgeOffset,
    fRangeToTextOffset,
    fSerializeContent,
    SourceHistory,
    fStripBlockBoundaryLineBreaks,
    fSyncCodeBlockHatches,
    fTextOffsetToRange,
    fTokenizeContent
  } from '$lib/utils';
  import githubLightCss from 'highlight.js/styles/github.css?inline';
  import githubDarkCss from 'highlight.js/styles/github-dark.css?inline';
  import { mode } from 'mode-watcher';
  import { onDestroy, onMount, untrack } from 'svelte';

  interface Props {
    class?: string;
    disabled?: boolean;
    onInput?: () => void;
    onKeydown?: (pEvent: KeyboardEvent) => void;
    onPaste?: (pEvent: ClipboardEvent) => void;
    placeholder?: string;
    value?: string;
  }

  let {
    class: className = '',
    disabled = false,
    onInput,
    onKeydown,
    onPaste,
    placeholder = 'Ask anything...',
    value = $bindable('')
  }: Props = $props();

  let vRootElement: HTMLDivElement | undefined = $state();
  let vLastEmittedValue = '';
  let vIsComposing = $state(false);

  // Undo/redo in source space: the imperative token rebuilds destroy the
  // browser's native undo stack.
  const cHistory = new SourceHistory();

  // Browsers disagree on what an empty rich chat form input contains (`<br>`,
  // `<div><br></div>`, or nothing), so emptiness is decided by the
  // serialized source, not the DOM shape.
  function fSyncEmptyState(pSerialized?: string) {
    if (!vRootElement) return;

    const cSource = pSerialized ?? fSerializeContent(vRootElement);

    vRootElement.dataset.empty = cSource.length === 0 ? BooleanString.cTrue : BooleanString.cFalse;
  }

  function fRenderTokens(pTokens: ChatFormInputRichToken[]) {
    if (!vRootElement) return;

    const cCaret = fRangeToTextOffset(vRootElement, fSafeRange());

    // eslint-disable-next-line svelte/no-dom-manipulating -- the token layer is owned imperatively; Svelte renders only the rich chat form input host, never its children
    vRootElement.replaceChildren(fBuildFragment(pTokens));

    fSyncCodeBlockHatches(vRootElement);
    fHighlightCodeBlocks(vRootElement);

    fRestoreCaret(cCaret);
    fResizeHeight();
    fSyncEmptyState();
  }

  // Last highlighted source segment per block element - typing inside
  // a block re-highlights only when the segment actually changed.
  const cHighlightedSegments = new WeakMap<HTMLElement, string>();

  const cCodeBlockOpenRe = /^```([^\n`]*)\n/;

  /**
   * Apply syntax highlighting to a code block element's CONTENT. The
   * fence lines stay plain text, and the blank padding that
   * `highlightCode` trims is re-added as plain text, so the element's
   * textContent stays byte-exact with the source segment. Replaces
   * the element's children - callers restore the caret afterwards.
   * Returns false when nothing changed.
   */
  function fHighlightCodeBlockElement(pEl: HTMLElement): boolean {
    const cSegment = pEl.textContent ?? '';

    if (cHighlightedSegments.get(pEl) === cSegment) return false;

    const cOpen = cCodeBlockOpenRe.exec(cSegment);

    if (!cOpen) return false;

    const prefix = cOpen[0];
    const cLanguage = cOpen[1].trim().split(/\s+/)[0] ?? '';
    const cContent = cSegment.slice(prefix.length, -3);
    const cLeading = cContent.match(cCodeBlock.TRIM_LEADING_PADDING_REGEX)?.[0] ?? '';
    const cTrailing = cContent.match(cCodeBlock.TRIM_TRAILING_PADDING_REGEX)?.[0] ?? '';
    const cCore = cContent.slice(cLeading.length, cContent.length - cTrailing.length);
    // autoDetect off: re-guessing the language on every keystroke
    // costs ~38ms a call and flickers while typing
    const html = cCore ? fHighlightCode(cCore, cLanguage || 'text', false) : '';
    const cTpl = document.createElement('template');

    cTpl.innerHTML = html;

    pEl.replaceChildren(
      document.createTextNode(prefix + cLeading),
      cTpl.content.cloneNode(true),
      document.createTextNode(cTrailing + '```')
    );
    cHighlightedSegments.set(pEl, cSegment);

    return true;
  }

  function fHighlightCodeBlocks(pRoot: HTMLElement) {
    for (const cEl of pRoot.querySelectorAll<HTMLElement>(
      `code[${cCodeTokenAttr}="${ChatFormInputRichTokenKind.cCodeBlock}"]`
    )) {
      fHighlightCodeBlockElement(cEl);
    }
  }

  /**
   * Re-highlight the code block the caret sits in after an edit.
   * Skipped when the block's segment is unchanged since its last
   * highlight, so edits outside blocks cost nothing.
   */
  function fRehighlightCaretCodeBlock() {
    if (!vRootElement) return;

    const cRange = fSafeRange();

    if (!cRange) return;

    let vNode: Node | null = cRange.startContainer;

    if (vNode === vRootElement) {
      vNode = vRootElement.childNodes[cRange.startOffset - 1] ?? null;
    }

    while (vNode && vNode !== vRootElement) {
      if (
        vNode instanceof HTMLElement &&
        vNode.getAttribute(cCodeTokenAttr) === ChatFormInputRichTokenKind.cCodeBlock
      ) {
        const cCaret = fRangeToTextOffset(vRootElement, cRange);

        if (fHighlightCodeBlockElement(vNode)) {
          fRestoreCaret(cCaret);
        }

        return;
      }

      vNode = vNode.parentNode;
    }
  }

  /**
   * Is the caret inside a fenced code block region? Source-level
   * (not DOM-level) so the still-OPEN fence counts too: while the
   * user is typing a block, no closing ``` exists yet and the
   * buffer is plain text with no block element to find. Root-level
   * caret positions right at a closed block's edge (escape
   * hatches, element boundaries restored by `textOffsetToRange`)
   * resolve past the closing fence, so they count as OUTSIDE.
   */
  function fCaretInCodeBlock(): boolean {
    if (!vRootElement) return false;

    return fIsOffsetInCodeBlock(
      fSerializeContent(vRootElement),
      fRangeToTextOffset(vRootElement, fSafeRange())
    );
  }

  /**
   * hljs theme for the highlighted code blocks. Mirrors
   * SyntaxHighlightedCode.svelte: one shared style element
   * (deduped via the data attribute) swapped on mode change.
   */
  function fLoadHighlightTheme(pIsDark: boolean) {
    document
      .querySelectorAll(`style[${cUiDataAttrs.HIGHLIGHT_THEME_PREVIEW}]`)
      .forEach((pS) => pS.remove());

    const cStyle = document.createElement('style');

    cStyle.setAttribute(cUiDataAttrs.HIGHLIGHT_THEME_PREVIEW, BooleanString.cTrue);
    cStyle.textContent = pIsDark ? githubDarkCss : githubLightCss;

    document.head.appendChild(cStyle);
  }

  $effect(() => {
    fLoadHighlightTheme(mode.current === ColorMode.DARK);
  });

  function fSafeRange(): Range | null {
    if (!vRootElement) return null;

    const cSelection = window.getSelection();

    if (!cSelection || cSelection.rangeCount === 0) return null;

    const cRange = cSelection.getRangeAt(0);

    if (
      !vRootElement.contains(cRange.startContainer) ||
      !vRootElement.contains(cRange.endContainer)
    ) {
      return null;
    }

    return cRange;
  }

  function fRestoreCaret(pOffset: number, pExtend = false) {
    if (!vRootElement) return;

    const cTarget = fTextOffsetToRange(vRootElement, pOffset);
    const cSelection = window.getSelection();

    if (!cSelection) return;

    if (pExtend && cSelection.anchorNode) {
      cSelection.setBaseAndExtent(
        cSelection.anchorNode,
        cSelection.anchorOffset,
        cTarget.startContainer,
        cTarget.startOffset
      );

      return;
    }

    cSelection.removeAllRanges();
    cSelection.addRange(cTarget);
  }

  function fResizeHeight() {
    if (!vRootElement) return;

    vRootElement.style.height = 'auto';
    vRootElement.style.height = `${vRootElement.scrollHeight}px`;
  }

  function fRecordHistory(pNewGroup: boolean) {
    if (!vRootElement) return;

    cHistory.mPush(
      { caret: fRangeToTextOffset(vRootElement, fSafeRange()), value: vLastEmittedValue },
      Date.now(),
      pNewGroup
    );
  }

  /**
   * Re-emit the current markdown source value to the parent, then
   * reconcile the DOM against the token stream: when a code span
   * was just completed or broken, the token boundaries no longer
   * match the element structure and the DOM is rebuilt (caret
   * preserved through the source-offset mapping).
   */
  function fProcessInput(pInputType?: string) {
    if (vIsComposing || !vRootElement) return;

    fSyncEmptyState();
    fResizeHeight();

    // Shift+Enter right after a code block leaves an all-newline
    // text node (the fence's separator line plus Chromium's
    // artificial end-of-buffer line break). Strip both so the caret
    // lands on the line directly below the block.
    if (pInputType === 'insertLineBreak' || pInputType === 'insertParagraph') {
      const cCaret = fRangeToTextOffset(vRootElement, fSafeRange());

      if (fStripBlockBoundaryLineBreaks(vRootElement)) {
        fRestoreCaret(cCaret);
      } else {
        const cSource = fSerializeContent(vRootElement);

        let vEnd = cCaret;

        // the caret must end up after the inserted \n; some browsers
        // leave it before (stuck at the end of the old line). A
        // preceding \n means it already sits past the break
        // (Chromium's artificial trailing newline) - leave it.
        if (cSource[vEnd] === '\n' && cSource[vEnd - 1] !== '\n') {
          vEnd += 1;
          fRestoreCaret(vEnd);
        }

        // a line break at the buffer end renders only with a second,
        // artificial trailing \n: a lone trailing \n is collapsed, so
        // the new line is invisible and the next typed character
        // consumes it. Append it when missing - unless the trailing
        // \n doubles as a block's separator line (source ends with
        // \n\n) or sits inside a block element.
        let vLast = vRootElement.lastChild;

        while (vLast && vLast.nodeName === 'BR') vLast = vLast.previousSibling;

        if (
          vEnd === cSource.length &&
          cSource.endsWith('\n') &&
          cSource[cSource.length - 2] !== '\n' &&
          vLast?.nodeType === Node.TEXT_NODE
        ) {
          // eslint-disable-next-line svelte/no-dom-manipulating -- the token layer is owned imperatively; Svelte renders only the rich chat form input host, never its children
          vRootElement.appendChild(document.createTextNode('\n'));
          fRestoreCaret(cSource.length);
          fResizeHeight();
        }
      }
    }

    fSyncCodeBlockHatches(vRootElement);

    const cSerialized = fSerializeContent(vRootElement);

    fSyncEmptyState(cSerialized);

    if (cSerialized === vLastEmittedValue) return;

    // Plain typing/deletes coalesce per time window; structural edits
    // (paste, newline, cut, autocorrect) start a new undo group.
    fRecordHistory(pInputType !== 'insertText' && !pInputType?.startsWith('deleteContent'));

    vLastEmittedValue = cSerialized;
    value = cSerialized;

    // Rebuild when token boundaries shifted (a code span was just
    // completed or broken) - the browser-owned text nodes cannot
    // restyle themselves across element boundaries.
    const aTokens = fTokenizeContent(cSerialized);

    if (!fDomMatchesTokens(vRootElement, aTokens)) {
      fRenderTokens(aTokens);

      // The rebuild can re-shape the DOM in a way that changes the
      // serialization (e.g. Chromium merged trailing text into the
      // block element and the rebuild splits it back out, which
      // synthesizes the separator newline) - keep value in sync.
      const cReserialized = fSerializeContent(vRootElement);

      if (cReserialized !== cSerialized) {
        vLastEmittedValue = cReserialized;
        value = cReserialized;
      }
    } else {
      fRehighlightCaretCodeBlock();
    }

    onInput?.();
  }

  function fHandleInput(pEvent: Event) {
    fProcessInput((pEvent as InputEvent).inputType);
  }

  function fHandleCompositionStart() {
    vIsComposing = true;
  }

  function fHandleCompositionEnd() {
    vIsComposing = false;
    fProcessInput();
  }

  /**
   * Insert a line break at the caret MANUALLY. Native Shift+Enter at
   * the buffer end varies across browsers (a lone trailing \n that the
   * renderer collapses, or a <br> that the hatch sync strips), which
   * can leave the caret stuck on the old line; splitting the text node
   * ourselves keeps the DOM shape - and the caret - deterministic.
   * `processInput` then appends the artificial trailing \n when the
   * break lands at the buffer end.
   */
  function fInsertLineBreak() {
    if (!vRootElement) return;

    const cRange = fSafeRange();

    if (!cRange) return;

    if (!cRange.collapsed) {
      cRange.deleteContents();
    }

    const cContainer = cRange.startContainer;
    const cOffset = cRange.startOffset;
    const cNl = document.createTextNode('\n');

    // a break at the very end of a code block exits the block (the
    // new line belongs below it, not inside)
    let vExitBlock: HTMLElement | null = null;

    if (cContainer.nodeType === Node.TEXT_NODE) {
      let vNode: Node | null = cContainer.parentNode;

      while (vNode && vNode !== vRootElement) {
        if (
          vNode instanceof HTMLElement &&
          vNode.getAttribute(cCodeTokenAttr) === ChatFormInputRichTokenKind.cCodeBlock
        ) {
          const cTail = document.createRange();

          cTail.setStart(cContainer, cOffset);
          cTail.setEnd(vNode, vNode.childNodes.length);

          if (cTail.toString().length === 0) vExitBlock = vNode;

          break;
        }

        vNode = vNode.parentNode;
      }
    }

    if (vExitBlock) {
      vExitBlock.after(cNl);
    } else if (cContainer.nodeType === Node.TEXT_NODE) {
      const text = cContainer as Text;

      if (cOffset === 0) {
        text.before(cNl);
      } else if (cOffset === text.length) {
        text.after(cNl);
      } else {
        text.splitText(cOffset).before(cNl);
      }
    } else {
      cContainer.insertBefore(cNl, cContainer.childNodes[cOffset] ?? null);
    }

    const cSelection = window.getSelection();
    const cAfter = document.createRange();

    cAfter.setStartAfter(cNl);
    cAfter.collapse(true);
    cSelection?.removeAllRanges();
    cSelection?.addRange(cAfter);

    fProcessInput('insertLineBreak');
  }

  /**
   * Arrow escape to the line BEFORE a leading code block. Native
   * caret movement has no position above a buffer-starting block,
   * so a transient `<br>` hatch is created on demand: it gives the
   * caret a visible line, is consumed by the first character typed
   * on it, and is removed again when the caret leaves (see
   * handleSelectionChange). Returns true when the caret was moved.
   */
  function fMoveCaretBeforeLeadingCodeBlock(pKey: string, pExtend: boolean): boolean {
    if (!vRootElement) return false;

    // a hatch already exists - native movement handles it
    if (vRootElement.firstChild?.nodeName === 'BR') return false;

    const cFirst = vRootElement.firstChild;

    if (
      !(cFirst instanceof HTMLElement) ||
      cFirst.getAttribute(cCodeTokenAttr) !== ChatFormInputRichTokenKind.cCodeBlock
    )
      return false;

    const cRange = fSafeRange();

    if (!cRange || !cRange.collapsed) return false;

    // the caret must sit inside the block: on its very first
    // character for ArrowLeft, anywhere on its first line for
    // ArrowUp
    if (!cFirst.contains(cRange.startContainer)) return false;

    const cCaret = fRangeToTextOffset(vRootElement, cRange);

    if (pKey === 'ArrowLeft') {
      if (cCaret !== 0) return false;
    } else {
      const cFirstLineEnd = (cFirst.textContent ?? '').indexOf('\n');

      if (cFirstLineEnd !== -1 && cCaret > cFirstLineEnd) return false;
    }

    // eslint-disable-next-line svelte/no-dom-manipulating -- the token layer is owned imperatively; Svelte renders only the rich chat form input host, never its children
    vRootElement.prepend(document.createElement('br'));
    fRestoreCaret(0, pExtend);

    return true;
  }

  /**
   * Remove the transient leading hatch once the caret leaves it.
   * The hatch only exists to give the caret a line above a leading
   * code block; with the caret anywhere else the empty line would
   * just be visual noise. Typing on the hatch line consumes it via
   * the stale-hatch removal in `syncCodeBlockHatches` instead (the
   * new text node takes its place before the block).
   */
  function fHandleSelectionChange() {
    if (!vRootElement) return;

    const cFirst = vRootElement.firstChild;

    if (cFirst?.nodeName !== 'BR') return;

    const cSecond = cFirst.nextSibling;

    if (
      !(cSecond instanceof HTMLElement) ||
      cSecond.getAttribute(cCodeTokenAttr) !== ChatFormInputRichTokenKind.cCodeBlock
    )
      return;

    const cRange = fSafeRange();
    const cOnHatch =
      cRange !== null && cRange.startContainer === vRootElement && cRange.startOffset === 0;

    if (!cOnHatch) {
      cFirst.remove();
    }
  }

  /**
   * Undo/redo is replayed from source snapshots (the token rebuilds
   * destroy the native undo stack). Arrow keys around badges are
   * repaired locally: a badge is a non-editable island, so plain
   * ArrowLeft after a leading badge has no native previous position
   * and word jumps overshoot it by a word.
   *
   * Plain Enter inside a fenced code block (closed, or still open
   * while being typed) acts as Shift+Enter and adds a line instead of
   * submitting. ArrowLeft/ArrowUp at the edge of a leading code block
   * create the transient before-block hatch.
   */
  function fHandleKeydown(pEvent: KeyboardEvent) {
    const cMod = pEvent.ctrlKey || pEvent.metaKey;

    if (cMod && !pEvent.altKey && !vIsComposing && vRootElement) {
      const cKey = pEvent.key.toLowerCase();
      const cIsUndo = cKey === 'z' && !pEvent.shiftKey;
      const cIsRedo = cKey === 'y' || (cKey === 'z' && pEvent.shiftKey);

      if (cIsUndo || cIsRedo) {
        pEvent.preventDefault();
        const cCurrent = {
          caret: fRangeToTextOffset(vRootElement, fSafeRange()),
          value: vLastEmittedValue
        };
        const cEntry = cIsUndo ? cHistory.mUndo(cCurrent) : cHistory.mRedo(cCurrent);

        if (cEntry) fApplyHistoryEntry(cEntry);

        return;
      }
    }

    if (
      pEvent.key === 'Enter' &&
      pEvent.shiftKey &&
      !pEvent.ctrlKey &&
      !pEvent.metaKey &&
      !pEvent.altKey &&
      !fIsIMEComposing(pEvent) &&
      !disabled &&
      !fCaretInCodeBlock() &&
      fSafeRange()
    ) {
      // Own the break outside code blocks: native end-of-buffer
      // behavior varies across browsers and can leave the caret
      // stuck on the old line (see insertLineBreak).
      pEvent.preventDefault();
      fInsertLineBreak();

      return;
    }

    if (
      pEvent.key === 'Enter' &&
      !pEvent.shiftKey &&
      !pEvent.ctrlKey &&
      !pEvent.metaKey &&
      !pEvent.altKey &&
      !fIsIMEComposing(pEvent) &&
      fCaretInCodeBlock()
    ) {
      // The native plain-Enter path must never run: it splits the
      // buffer into `<div>` wrappers that `serializeContent` cannot
      // see. `insertLineBreak` reproduces the Shift+Enter DOM (a `\n`
      // text node) and fires `input` synchronously, so the usual
      // re-tokenize/re-highlight follows.
      pEvent.preventDefault();
      document.execCommand('insertLineBreak');

      return;
    }

    if (
      vRootElement &&
      (pEvent.key === 'ArrowLeft' || pEvent.key === 'ArrowUp') &&
      !pEvent.altKey &&
      !pEvent.ctrlKey &&
      !pEvent.metaKey
    ) {
      if (fMoveCaretBeforeLeadingCodeBlock(pEvent.key, pEvent.shiftKey)) {
        pEvent.preventDefault();

        return;
      }
    }

    if (vRootElement && (pEvent.key === 'ArrowLeft' || pEvent.key === 'ArrowRight')) {
      const cIsWordJump = (pEvent.altKey || pEvent.ctrlKey) && !pEvent.metaKey && !pEvent.shiftKey;
      const cIsPlainLeft =
        pEvent.key === 'ArrowLeft' && !pEvent.altKey && !pEvent.ctrlKey && !pEvent.metaKey;

      if (cIsWordJump || cIsPlainLeft) {
        const cSource = fSerializeContent(vRootElement);
        const cCaret = fRangeToTextOffset(vRootElement, fSafeRange());
        const cTarget = cIsWordJump
          ? fBadgeAwareWordJump(
              cSource,
              cCaret,
              pEvent.key === 'ArrowRight' ? 'forward' : 'backward'
            )
          : fLeadingBadgeEdgeOffset(cSource, cCaret);

        if (cTarget !== null) {
          pEvent.preventDefault();
          fRestoreCaret(cTarget, pEvent.shiftKey);

          return;
        }
      }
    }

    onKeydown?.(pEvent);
  }

  // lastEmittedValue is set before `value` so the sync effect treats the
  // change as our own and does not re-render.
  function fApplyHistoryEntry(pEntry: SourceHistoryEntry) {
    if (!vRootElement) return;

    fRenderTokens(fTokenizeContent(pEntry.value));
    vLastEmittedValue = pEntry.value;
    value = pEntry.value;
    onInput?.();
    fRestoreCaret(pEntry.caret);
  }

  /**
   * Plain-text paste. preventDefault + manual insertText keeps the
   * browser from producing stray `<div>` wrappers mid-paste; insertText
   * fires `input` synchronously, so `processInput` re-tokenizes the
   * buffer and rebuilds when the pasted text carries badge or code
   * tokens.
   */
  function fHandlePasteEvent(pEvent: ClipboardEvent) {
    const cPasted = pEvent.clipboardData?.getData('text/plain');

    if (cPasted && cPasted.length > 0) {
      pEvent.preventDefault();

      // Snap a collapsed caret through the offset mapping first: at
      // element-boundary carets (e.g. right before a badge) Chromium's
      // insertText can drop the preceding text node's trailing whitespace.
      const cRange = fSafeRange();

      if (vRootElement && cRange && cRange.collapsed) {
        fRestoreCaret(fRangeToTextOffset(vRootElement, cRange));
      }

      document.execCommand('insertText', false, cPasted);
    }
  }

  // The parent's paste handler runs first and preventDefaults when it
  // consumes the event (files, quoted prompts, long text).
  function fHandlePaste(pEvent: ClipboardEvent) {
    onPaste?.(pEvent);

    if (!pEvent.defaultPrevented) {
      fHandlePasteEvent(pEvent);
    }
  }

  // The selection as markdown SOURCE (each badge contributes its full
  // `[name](file://...)` link), so copy/cut carry raw markdown and
  // pasting back re-renders the badges. Null for collapsed/outside
  // selections - native clipboard behavior is fine there.
  function fSelectionSourceSlice(): { text: string; range: Range } | null {
    if (!vRootElement) return null;

    const cRange = fSafeRange();

    if (!cRange || cRange.collapsed) return null;

    const cStartRange = cRange.cloneRange();

    cStartRange.collapse(true);

    const cSource = fSerializeContent(vRootElement);
    const cStart = fRangeToTextOffset(vRootElement, cStartRange);
    const cEnd = fRangeToTextOffset(vRootElement, cRange);

    return { range: cRange, text: cSource.slice(cStart, cEnd) };
  }

  function fHandleCopy(pEvent: ClipboardEvent) {
    const cSlice = fSelectionSourceSlice();

    if (!cSlice) return;

    pEvent.clipboardData?.setData('text/plain', cSlice.text);
    pEvent.preventDefault();
  }

  function fHandleCut(pEvent: ClipboardEvent) {
    const cSlice = fSelectionSourceSlice();

    if (!cSlice) return;

    pEvent.clipboardData?.setData('text/plain', cSlice.text);
    pEvent.preventDefault();

    // preventDefault suppresses the native deletion, so remove the
    // selection manually and re-emit.
    cSlice.range.deleteContents();
    fProcessInput('deleteByCut');
  }

  onMount(() => {
    // untrack: the DOM is managed manually from input events, so the
    // initial render must not subscribe to the value.
    fRenderTokens(fTokenizeContent(untrack(() => value)));
    vLastEmittedValue = untrack(() => value ?? '');
    fResizeHeight();
    fSyncEmptyState();
    document.addEventListener('selectionchange', fHandleSelectionChange);

    if (!cDeviceStore.isMobile) {
      vRootElement?.focus({ preventScroll: true });
    }
  });

  onDestroy(() => {
    document.removeEventListener('selectionchange', fHandleSelectionChange);
  });

  // External `value` updates. When incoming === lastEmittedValue the
  // change came from our own input, so leave the DOM alone - the
  // browser already owns the right shape.
  $effect(() => {
    const cIncoming = value ?? '';

    if (cIncoming === vLastEmittedValue) return;

    fRecordHistory(true); // external edit (mention insert, clear, ...): own undo step
    fRenderTokens(fTokenizeContent(cIncoming));
    vLastEmittedValue = cIncoming;
  });

  export function getElement() {
    return vRootElement;
  }

  export function getCaretOffset(): number {
    if (!vRootElement) return 0;

    return fRangeToTextOffset(vRootElement, fSafeRange());
  }

  // Focus first: `selection.addRange` requires it on some browsers.
  export function setCaretOffset(pOffset: number) {
    if (vRootElement && vRootElement !== document.activeElement) {
      vRootElement.focus({ preventScroll: true });
    }

    fRestoreCaret(pOffset);
  }

  export function focus() {
    if (cDeviceStore.isMobile) return;

    vRootElement?.focus({ preventScroll: true });
  }

  export function resetHeight() {
    if (vRootElement) {
      vRootElement.style.height = '';
      fResizeHeight();
    }
  }
</script>

<div class="flex-1 {className} mb-0.5">
  <div
    bind:this={vRootElement}
    aria-disabled={disabled}
    aria-multiline="true"
    aria-placeholder={placeholder}
    class={[
      'chat-form-input-rich text-md min-h-12 w-full overflow-y-auto whitespace-pre-wrap wrap-break-word border-0 bg-transparent p-0 leading-6 outline-none focus-visible:ring-0 focus-visible:ring-offset-0',
      disabled && 'cursor-not-allowed'
    ]}
    contenteditable={!disabled}
    data-placeholder={placeholder}
    oncompositionend={fHandleCompositionEnd}
    oncompositionstart={fHandleCompositionStart}
    oncopy={fHandleCopy}
    oncut={fHandleCut}
    oninput={fHandleInput}
    onkeydown={fHandleKeydown}
    onpaste={fHandlePaste}
    role="textbox"
    style="max-height: var(--max-message-height);"
    tabindex={disabled ? -1 : 0}
  ></div>
</div>

<style>
  /* pre-wrap is load-bearing: without it Chromium collapses \n in
	   text nodes and converts them to spaces while typing */
  .chat-form-input-rich {
    white-space: pre-wrap;
  }

  .chat-form-input-rich:global([data-empty='true'])::before {
    content: attr(data-placeholder);
    color: var(--muted-foreground);
    pointer-events: none;
  }

  /* Inline code - mirrors markdown-content.css */
  .chat-form-input-rich :global(code[data-code-token='code_inline']) {
    background: var(--muted);
    color: var(--muted-foreground);
    padding: 0.125rem 0.375rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
  }

  /* Fenced code block - mirrors .code-block-wrapper in markdown-content.css */
  .chat-form-input-rich :global(code[data-code-token='code_block']) {
    display: block;
    margin: 0.25rem 0;
    padding: 0.75rem 1rem;
    border: 1px solid color-mix(in oklch, var(--border) 30%, transparent);
    border-radius: 0.75rem;
    background: var(--code-background);
    color: var(--code-foreground);
    font-size: 0.875rem;
    line-height: 1.3;
  }
</style>
