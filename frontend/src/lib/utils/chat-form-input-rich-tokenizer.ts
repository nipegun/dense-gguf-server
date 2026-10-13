/**
 * Maps between the chat-form-input-rich's markdown source and the
 * badge/code/text token stream the DOM is built from. A badge is one
 * opaque source contribution (`[name](file://path)`); its own subtree
 * is never walked, and the caret cannot land inside it, so offsets
 * resolve to the nearest badge edge. Code spans (`<code data-code-token>`)
 * are EDITABLE, unlike badges: they carry the full source segment
 * (backtick fences included) as their text, so their textContent
 * serializes verbatim and source offsets map 1:1 to text offsets.
 *
 * The tokenizer emits a flat DOM (text nodes + badges + code spans),
 * but browsers restructure it on Enter (`<div>` line wrappers, `<br>`
 * shapes). Serialization folds those back into `\n` so the source
 * never diverges from what is on screen; both offset mappers
 * understand the same shapes.
 *
 * The newline separating a fenced block from adjacent content is a
 * SOURCE-level concept, never stored in the DOM: the block is
 * display:block, so a leading `\n` in the following text node would
 * render as a phantom empty line. Serialization synthesizes exactly
 * one `\n` at every block boundary and `buildFragment` strips it from
 * text tokens. A text node's own leading/trailing `\n` next to a
 * block is an ADDITIONAL blank line.
 */

import {
  fDecodeFileLinkPath,
  fFileMentionLinkRe,
  fGetMentionBadgeIconPaths,
  fGetMentionBadgeLabel
} from './mention-badge';
import {
  cCodeTokenAttr,
  cMentionBadgeClassname,
  cMentionBadgeDataAttrs,
  cMentionBadgeIconClassname,
  cMentionBadgeSvgAttributes,
  cSettingsKeys
} from '$lib/constants';
import { BooleanString, ChatFormInputRichTokenKind } from '$lib/enums';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import { cToolsStore } from '$lib/stores/tools.svelte';
import type { ChatFormInputRichToken } from '$lib/types/chat-form-input-rich';

// Block wrappers browsers insert for newlines; each folds back into a
// single `\n` during serialization.
const sBlockTagNames = new Set(['DIV', 'P']);
// `file://` is required so plain URLs stay as text; `)` terminates only
// when not followed by whitespace or `[` (adjacent badges keep working).
const cMentionBadgeRe = fFileMentionLinkRe('g');

function fBadgeSourceLength(pName: string, pPath: string): number {
  if (!pName || !pPath) return 0;

  return `[${pName}](file://${pPath})`.length;
}

/**
 * Recognize complete code spans. Fenced blocks (triple backticks,
 * optional language, possibly multiline) take priority over inline
 * spans (single backticks, single line, non-empty). Only CLOSED
 * spans match: an unclosed fence stays plain text until the closing
 * backticks land. The match includes the fences so the token's
 * source length equals its rendered text length.
 */
const cCodeSpanRe = /(```[\s\S]*?```)|(`[^`\n]+`)/g;

/**
 * Cheap gate check for `ChatForm`: does the buffer contain a
 * complete code span (inline or fenced)? Used to promote the plain
 * textarea to the chat-form-input-rich renderer.
 */
export function fContainsCodeSpan(pValue: string): boolean {
  cCodeSpanRe.lastIndex = 0;

  return cCodeSpanRe.test(pValue);
}

const cCodeFenceRe = /```/g;

/**
 * Is `offset` inside a fenced code block region? Toggle-based: an
 * odd number of ``` fences before the offset means the position
 * sits in block content. Unlike `containsCodeSpan` this also
 * counts the still-OPEN fence while the user is typing a block
 * (no closing ``` yet), so Enter can add a line instead of
 * submitting the message.
 */
export function fIsOffsetInCodeBlock(pSource: string, pOffset: number): boolean {
  let vInside = false;

  cCodeFenceRe.lastIndex = 0;

  let vMatch: RegExpExecArray | null;

  while ((vMatch = cCodeFenceRe.exec(pSource)) !== null) {
    if (vMatch.index + vMatch[0].length > pOffset) break;

    vInside = !vInside;
  }

  return vInside;
}

/**
 * Tokenize a markdown source value into the segments the
 * chat-form-input-rich will render. Code spans are carved out first
 * (their content is literal - a `file://` link inside backticks
 * must NOT render as a badge), then plain text and badges
 * interleave in the remaining gaps. Any whitespace after a badge
 * stays in a plain text token so the round trip is byte-exact.
 */
export function fTokenizeContent(pInput: string): ChatFormInputRichToken[] {
  const aTokens: ChatFormInputRichToken[] = [];

  let vCursor = 0;

  cCodeSpanRe.lastIndex = 0;

  let vMatch: RegExpExecArray | null;

  while ((vMatch = cCodeSpanRe.exec(pInput)) !== null) {
    const cStart = vMatch.index;

    if (cStart > vCursor) {
      fPushTextAndBadgeTokens(pInput.slice(vCursor, cStart), aTokens);
    }

    aTokens.push(
      vMatch[1] !== undefined
        ? { kind: ChatFormInputRichTokenKind.cCodeBlock, text: vMatch[1] }
        : { kind: ChatFormInputRichTokenKind.cCodeInline, text: vMatch[2] }
    );
    vCursor = cStart + vMatch[0].length;
  }

  if (vCursor < pInput.length) {
    fPushTextAndBadgeTokens(pInput.slice(vCursor), aTokens);
  }

  return aTokens;
}

/**
 * Tokenize a code-free segment into text and badge tokens.
 */
function fPushTextAndBadgeTokens(pInput: string, pTokens: ChatFormInputRichToken[]) {
  let vCursor = 0;

  cMentionBadgeRe.lastIndex = 0;

  let vMatch: RegExpExecArray | null;

  while ((vMatch = cMentionBadgeRe.exec(pInput)) !== null) {
    const [vWhole, vName, vPath] = vMatch;
    const cStart = vMatch.index;

    if (cStart > vCursor) {
      pTokens.push({ kind: ChatFormInputRichTokenKind.TEXT, text: pInput.slice(vCursor, cStart) });
    }

    pTokens.push({ kind: ChatFormInputRichTokenKind.BADGE, name: vName, path: vPath });
    vCursor = cStart + vWhole.length;
  }

  if (vCursor < pInput.length) {
    pTokens.push({ kind: ChatFormInputRichTokenKind.TEXT, text: pInput.slice(vCursor) });
  }
}

function fIsCodeBlockElement(node: Node | null): node is HTMLElement {
  return (
    node instanceof HTMLElement &&
    node.getAttribute(cCodeTokenAttr) === ChatFormInputRichTokenKind.cCodeBlock
  );
}

/**
 * Serialize a chat-form-input-rich subtree back to source. `<br>` and block
 * wrappers the browser inserted for newlines fold back into `\n` (a
 * trailing `<br>` is the browser's caret placeholder, not a newline);
 * any other element is transparent. Code spans serialize their
 * textContent verbatim (fences included). One separator `\n` is
 * synthesized at every fenced-block boundary (the DOM never stores
 * it), and a `<br>` adjacent to a code block is an escape hatch, not
 * a newline.
 */
export function fSerializeContent(pRoot: HTMLElement): string {
  let vOut = '';
  let vPendingBlockBoundary = false;

  const fWalk = (pParent: Node) => {
    let vFirst = true; // no source-contributing sibling seen yet

    for (const cChild of Array.from(pParent.childNodes)) {
      if (cChild.nodeType === Node.TEXT_NODE) {
        const text = cChild.textContent ?? '';

        if (text.length > 0) {
          if (vPendingBlockBoundary) {
            vOut += '\n';
            vPendingBlockBoundary = false;
          }

          vOut += text;
          vFirst = false;
        }

        continue;
      }

      if (cChild.nodeType !== Node.ELEMENT_NODE) continue;

      const cEl = cChild as HTMLElement;

      if (cEl.getAttribute(cMentionBadgeDataAttrs.BADGE) === BooleanString.cTrue) {
        const name = cEl.getAttribute(cMentionBadgeDataAttrs.NAME) ?? '';
        const cPath = cEl.getAttribute(cMentionBadgeDataAttrs.PATH) ?? '';

        if (name && cPath) {
          if (vPendingBlockBoundary) {
            vOut += '\n';
            vPendingBlockBoundary = false;
          }

          vOut += `[${name}](file://${cPath})`;
          vFirst = false;
        }

        continue;
      }

      const cCodeToken = cEl.getAttribute(cCodeTokenAttr);

      if (cCodeToken !== null) {
        const cIsBlock = cCodeToken === ChatFormInputRichTokenKind.cCodeBlock;

        if (cIsBlock && (vPendingBlockBoundary || !vFirst)) vOut += '\n';

        vPendingBlockBoundary = false;
        fWalk(cEl);
        vFirst = false;

        if (cIsBlock) vPendingBlockBoundary = true;

        continue;
      }

      if (cEl.tagName === 'BR') {
        const cIsHatch =
          fIsCodeBlockElement(cEl.previousSibling) || fIsCodeBlockElement(cEl.nextSibling);

        if (!cIsHatch && cEl.nextSibling) {
          if (vPendingBlockBoundary) {
            vOut += '\n';
            vPendingBlockBoundary = false;
          }

          vOut += '\n';
          vFirst = false;
        }

        continue;
      }

      if (sBlockTagNames.has(cEl.tagName)) {
        if (vPendingBlockBoundary || !vFirst) vOut += '\n';

        vPendingBlockBoundary = false;
        fWalk(cEl);
        vFirst = false;

        continue;
      }

      fWalk(cEl);

      if (vPendingBlockBoundary) vFirst = false;
    }
  };

  fWalk(pRoot);

  return vOut;
}

/**
 * Compare the live DOM's non-text structure against a token stream.
 * Only element contributions are compared (badges by name/path, code
 * spans by kind and source segment): text nodes are owned by the
 * browser between rebuilds, so their split/merge state is irrelevant.
 * A mismatch means token boundaries shifted (a code span was just
 * completed or broken) and the DOM needs a rebuild to restyle.
 */
export function fDomMatchesTokens(pRoot: HTMLElement, pTokens: ChatFormInputRichToken[]): boolean {
  const aExpected = pTokens.filter((pToken) => pToken.kind !== ChatFormInputRichTokenKind.TEXT);

  let vIndex = 0;

  const fWalk = (pParent: Node): boolean => {
    for (const cChild of Array.from(pParent.childNodes)) {
      if (cChild.nodeType !== Node.ELEMENT_NODE) continue;

      const cEl = cChild as HTMLElement;
      const cIsBadge = cEl.getAttribute(cMentionBadgeDataAttrs.BADGE) === BooleanString.cTrue;
      const cIsCode = cEl.getAttribute(cCodeTokenAttr) !== null;

      if (!cIsBadge && !cIsCode) {
        if (!fWalk(cEl)) return false;

        continue;
      }

      const cToken = aExpected[vIndex++];

      if (!cToken) return false;

      if (cIsBadge) {
        if (cToken.kind !== ChatFormInputRichTokenKind.BADGE) return false;

        if (cToken.name !== (cEl.getAttribute(cMentionBadgeDataAttrs.NAME) ?? '')) return false;

        if (cToken.path !== (cEl.getAttribute(cMentionBadgeDataAttrs.PATH) ?? '')) return false;

        continue;
      }

      const cCodeKind: ChatFormInputRichTokenKind =
        cEl.getAttribute(cCodeTokenAttr) === ChatFormInputRichTokenKind.cCodeBlock
          ? ChatFormInputRichTokenKind.cCodeBlock
          : ChatFormInputRichTokenKind.cCodeInline;

      if (cToken.kind !== cCodeKind) return false;

      if (
        cToken.kind === ChatFormInputRichTokenKind.cCodeInline ||
        cToken.kind === ChatFormInputRichTokenKind.cCodeBlock
      ) {
        if (cToken.text !== (cEl.textContent ?? '')) return false;
      }
    }

    return true;
  };

  return fWalk(pRoot) && vIndex === aExpected.length;
}

/**
 * Plain-text offset of a `Range` in the root; null range (selection
 * lost) falls back to buffer length. Walked against the live DOM (not
 * a clone) so a `<br>` keeps its trailing/not-trailing context. Code
 * spans count their full textContent (fences included) and the caret
 * may land inside them; synthesized block boundaries count one `\n`
 * once the caret is past them.
 */
export function fRangeToTextOffset(pRoot: HTMLElement, pRange: Range | null): number {
  if (!pRange) return fSerializeContent(pRoot).length;

  // A point is at/before the caret iff it falls inside [root start, caret].
  const cPre = pRange.cloneRange();

  cPre.selectNodeContents(pRoot);
  cPre.setEnd(pRange.endContainer, pRange.endOffset);
  const fAtOrBeforeCaret = (node: Node, pOffset: number) => cPre.comparePoint(node, pOffset) !== 1;

  let vTotal = 0;
  let vDone = false;
  // DOM position of a code block's synthesized after-boundary, set
  // when walking past a block and consumed by the next contributing
  // sibling (counts one `\n` once the caret is past it).
  let vPendingPoint: { node: Node; index: number } | null = null;

  const fWalk = (pParent: Node) => {
    let vFirst = true;

    for (const cChild of Array.from(pParent.childNodes)) {
      if (vDone) return;

      if (cChild.nodeType === Node.TEXT_NODE) {
        const text = cChild.textContent ?? '';

        if (text.length === 0) continue;

        if (vPendingPoint) {
          const { index: vIndex, node: vNode } = vPendingPoint;

          vPendingPoint = null;

          if (!fAtOrBeforeCaret(vNode, vIndex)) {
            vDone = true;

            return;
          }

          vTotal += 1;
        }

        if (!fAtOrBeforeCaret(cChild, 0)) {
          vDone = true;

          return;
        }

        if (pRange.endContainer === cChild) {
          vTotal += pRange.endOffset;
          vDone = true;

          return;
        }

        vTotal += text.length;
        vFirst = false;

        continue;
      }

      if (cChild.nodeType !== Node.ELEMENT_NODE) continue;

      const cEl = cChild as HTMLElement;
      const cParentNode = cEl.parentNode as Node;
      const cElIndex = Array.prototype.indexOf.call(cParentNode.childNodes, cEl);

      if (vPendingPoint) {
        const { index: vIndex, node: vNode } = vPendingPoint;

        vPendingPoint = null;

        if (!fAtOrBeforeCaret(vNode, vIndex)) {
          vDone = true;

          return;
        }

        vTotal += 1;
      }

      if (cEl.getAttribute(cMentionBadgeDataAttrs.BADGE) === BooleanString.cTrue) {
        const cLen = fBadgeSourceLength(
          cEl.getAttribute(cMentionBadgeDataAttrs.NAME) ?? '',
          cEl.getAttribute(cMentionBadgeDataAttrs.PATH) ?? ''
        );

        if (cLen === 0) continue;

        if (!fAtOrBeforeCaret(cParentNode, cElIndex + 1)) {
          vDone = true;

          return;
        }

        vTotal += cLen;
        vFirst = false;

        continue;
      }

      const cCodeToken = cEl.getAttribute(cCodeTokenAttr);

      if (cCodeToken !== null) {
        const cIsBlock = cCodeToken === ChatFormInputRichTokenKind.cCodeBlock;

        if (cIsBlock && !vFirst) {
          if (!fAtOrBeforeCaret(cEl, 0)) {
            vDone = true;

            return;
          }

          vTotal += 1;
        }

        fWalk(cEl);
        vFirst = false;

        if (cIsBlock) vPendingPoint = { index: cElIndex + 1, node: cParentNode };

        continue;
      }

      if (cEl.tagName === 'BR') {
        const cIsHatch =
          fIsCodeBlockElement(cEl.previousSibling) || fIsCodeBlockElement(cEl.nextSibling);

        if (cIsHatch || !cEl.nextSibling) continue;

        if (!fAtOrBeforeCaret(cParentNode, cElIndex + 1)) {
          vDone = true;

          return;
        }

        vTotal += 1;
        vFirst = false;

        continue;
      }

      if (sBlockTagNames.has(cEl.tagName)) {
        if (!vFirst) {
          if (!fAtOrBeforeCaret(cEl, 0)) {
            vDone = true;

            return;
          }

          vTotal += 1;
        }

        fWalk(cEl);
        vFirst = false;

        continue;
      }

      const cBefore = vTotal;

      fWalk(cEl);

      if (vTotal > cBefore) vFirst = false;
    }
  };

  fWalk(pRoot);

  return vTotal;
}

/**
 * Materialize a token stream into a DOM subtree: text nodes for text
 * tokens, `<span data-mention-badge="true">` elements for badges,
 * `<code data-code-token>` elements for code spans. The badge's class
 * string + inline SVG are shared with the rehype plugin via
 * `$lib/constants`.
 */
export function fBuildFragment(pTokens: ChatFormInputRichToken[]): DocumentFragment {
  const cFragment = document.createDocumentFragment();

  for (let vIndex = 0; vIndex < pTokens.length; vIndex++) {
    const cToken = pTokens[vIndex];

    if (cToken.kind === ChatFormInputRichTokenKind.TEXT) {
      let vText = cToken.text;

      // The separator \n at a fenced-block boundary is synthesized
      // at serialization time; keeping it in the DOM would render a
      // phantom empty line next to the block.
      if (
        pTokens[vIndex - 1]?.kind === ChatFormInputRichTokenKind.cCodeBlock &&
        vText.startsWith('\n')
      ) {
        vText = vText.slice(1);
      }

      if (
        pTokens[vIndex + 1]?.kind === ChatFormInputRichTokenKind.cCodeBlock &&
        vText.endsWith('\n')
      ) {
        vText = vText.slice(0, -1);
      }

      if (vText.length === 0) continue;

      cFragment.appendChild(document.createTextNode(vText));

      continue;
    }

    if (
      cToken.kind === ChatFormInputRichTokenKind.cCodeInline ||
      cToken.kind === ChatFormInputRichTokenKind.cCodeBlock
    ) {
      const cCode = document.createElement('code');

      cCode.setAttribute(cCodeTokenAttr, cToken.kind);
      cCode.textContent = cToken.text;
      cFragment.appendChild(cCode);

      continue;
    }

    // A leading badge gets an empty text node prepended: without a real
    // text position at the buffer start, the spot before the badge is
    // unreachable via keyboard (ArrowLeft/Home).
    if (!cFragment.lastChild) {
      cFragment.appendChild(document.createTextNode(''));
    }

    const badge = document.createElement('span');

    badge.setAttribute(cMentionBadgeDataAttrs.BADGE, BooleanString.cTrue);
    badge.setAttribute(cMentionBadgeDataAttrs.NAME, cToken.name);
    badge.setAttribute(cMentionBadgeDataAttrs.PATH, cToken.path);
    badge.title = fDecodeFileLinkPath(cToken.path);
    badge.className = cMentionBadgeClassname;
    badge.contentEditable = 'false';

    const svg = document.createElementNS(cMentionBadgeSvgAttributes['xmlns'], 'svg');

    for (const [vAttr, vValue] of Object.entries(cMentionBadgeSvgAttributes)) {
      svg.setAttribute(vAttr, vValue);
    }
    for (const cCls of cMentionBadgeIconClassname.split(/\s+/).filter(Boolean)) {
      svg.classList.add(cCls);
    }

    for (const cD of fGetMentionBadgeIconPaths(cToken.path)) {
      const cPath = document.createElementNS(cMentionBadgeSvgAttributes['xmlns'], 'path');

      cPath.setAttribute('d', cD);
      svg.appendChild(cPath);
    }

    const cLabel = document.createElement('span');

    cLabel.classList.add('shrink-0', 'truncate');
    cLabel.textContent = fGetMentionBadgeLabel(
      cToken.name,
      fDecodeFileLinkPath(cToken.path),
      cSettingsStore.mGetConfig(cSettingsKeys.SHOW_FULL_PATH_IN_MENTIONS),
      cToolsStore.serverHome
    );

    badge.appendChild(svg);
    badge.appendChild(cLabel);
    cFragment.appendChild(badge);
  }

  return cFragment;
}

// A sibling provides a reachable caret line when it is an element
// (badge, another block, an existing hatch) or a non-empty text node.
function fHasLineBeside(node: Node | null): boolean {
  if (!node) return false;

  if (node.nodeType === Node.ELEMENT_NODE) return true;

  return (node.textContent ?? '') !== '';
}

/**
 * A code block at the END of the buffer needs an editable line after
 * it: without one the caret cannot leave the block with
 * ArrowDown/ArrowRight. A trailing `<br>` provides that line while
 * staying transparent to serialization (skipped as a hatch), and is
 * removed again once real content takes its place.
 *
 * No hatch is added BEFORE a leading block: the empty line above it
 * is transient and managed by the component (created when the caret
 * arrows onto it, removed when the caret leaves). A transient
 * leading hatch found here is kept; the browser's lone placeholder
 * `<br>` in an empty root is left untouched.
 */
export function fSyncCodeBlockHatches(pRoot: HTMLElement) {
  for (const cChild of Array.from(pRoot.childNodes)) {
    if (cChild.nodeName !== 'BR') continue;

    const cIsPlaceholder = pRoot.childNodes.length === 1;
    const cIsLeadingHatch = !cChild.previousSibling && fIsCodeBlockElement(cChild.nextSibling);
    const cIsTrailingHatch = !cChild.nextSibling && fIsCodeBlockElement(cChild.previousSibling);

    // A hatch goes stale once real content takes over its line:
    // content before a leading hatch, content after a trailing one,
    // or a text node after the block already providing the line.
    // A `<br>` with no code block around is a real newline (browser
    // Shift+Enter shape) and stays.
    let vPrevElement = cChild.previousSibling;

    while (vPrevElement && vPrevElement.nodeType !== Node.ELEMENT_NODE) {
      vPrevElement = vPrevElement.previousSibling;
    }
    const cNearBlock =
      fIsCodeBlockElement(cChild.nextSibling) ||
      fIsCodeBlockElement(cChild.previousSibling) ||
      fIsCodeBlockElement(vPrevElement);

    if (!cIsPlaceholder && !cIsLeadingHatch && !cIsTrailingHatch && cNearBlock) {
      cChild.remove();
    }
  }

  for (const cChild of Array.from(pRoot.childNodes)) {
    if (!fIsCodeBlockElement(cChild)) continue;

    if (!fHasLineBeside(cChild.nextSibling)) {
      cChild.after(document.createElement('br'));
    }
  }
}

/**
 * Strip the separator and artificial newlines from an all-newline text
 * node directly after a fenced block. Chromium's line break at the
 * buffer end inserts an extra artificial `\n` so the new line has
 * height, and the first `\n` after a block doubles as the fence's
 * separator line (synthesized at serialization time). Removing both
 * makes Shift+Enter after a block land the caret on the line directly
 * below the block, like a plain textarea would.
 *
 * Only all-newline text nodes are touched: a node with real content
 * carries intentional blank lines and is left alone. Returns true when
 * the DOM changed.
 */
export function fStripBlockBoundaryLineBreaks(pRoot: HTMLElement): boolean {
  let vChanged = false;

  for (const cChild of Array.from(pRoot.childNodes)) {
    if (cChild.nodeType !== Node.TEXT_NODE) continue;

    if (!fIsCodeBlockElement(cChild.previousSibling)) continue;

    let vText = cChild.textContent ?? '';

    if (!/^\n{2,}$/.test(vText)) continue;

    vText = vText.slice(1);

    const cAtBufferEnd = !cChild.nextSibling || cChild.nextSibling.nodeName === 'BR';

    if (cAtBufferEnd) {
      vText = vText.slice(0, -1);
    }

    cChild.textContent = vText;
    vChanged = true;
  }

  return vChanged;
}

const cWordCharRe = /[\p{L}\p{N}_]/u;

/**
 * Word-jump target (Option+Arrow / Ctrl+Arrow) in source offsets, or null
 * when the jump crosses no badge and native word movement should handle
 * it. Badge spans are masked to word characters, so a badge counts as
 * exactly one word.
 */
export function fBadgeAwareWordJump(
  pSource: string,
  pOffset: number,
  pDirection: 'forward' | 'backward'
): number | null {
  let vMasked = '';

  const aBadgeSpans: Array<[number, number]> = [];

  for (const cToken of fTokenizeContent(pSource)) {
    const cLen =
      cToken.kind === ChatFormInputRichTokenKind.BADGE
        ? fBadgeSourceLength(cToken.name, cToken.path)
        : cToken.text.length;

    if (cToken.kind === ChatFormInputRichTokenKind.BADGE)
      aBadgeSpans.push([vMasked.length, vMasked.length + cLen]);

    vMasked += cToken.kind === ChatFormInputRichTokenKind.BADGE ? 'a'.repeat(cLen) : cToken.text;
  }

  if (aBadgeSpans.length === 0) return null;

  const fIsWord = (pIndex: number) => cWordCharRe.test(vMasked[pIndex]);
  const fSpanStartingAt = (pIndex: number) => aBadgeSpans.find(([vStart]) => vStart === pIndex);
  const fSpanEndingAt = (pIndex: number) => aBadgeSpans.find(([, vEnd]) => vEnd === pIndex);
  const cN = vMasked.length;

  let vI = pOffset;

  if (pDirection === 'forward') {
    // Entering a badge completes the word phase at the badge's end edge.
    if (!(vI < cN && fIsWord(vI))) {
      while (vI < cN && !fIsWord(vI)) vI++;
    }

    while (vI < cN && fIsWord(vI)) {
      const cSpan = fSpanStartingAt(vI);

      if (cSpan) {
        vI = cSpan[1];

        break;
      }

      vI++;
    }
  } else {
    if (!(vI > 0 && fIsWord(vI - 1))) {
      while (vI > 0 && !fIsWord(vI - 1)) vI--;
    }

    while (vI > 0 && fIsWord(vI - 1)) {
      const cSpan = fSpanEndingAt(vI);

      if (cSpan) {
        vI = cSpan[0];

        break;
      }

      vI--;
    }
  }

  if (vI === pOffset) return null;

  const cLo = Math.min(pOffset, vI);
  const cHi = Math.max(pOffset, vI);

  return aBadgeSpans.some(([vStart, vEnd]) => vStart < cHi && vEnd > cLo) ? vI : null;
}

/**
 * 0 when `caret` sits exactly at a leading badge's end edge, null
 * otherwise. Plain ArrowLeft there has no native previous position, so
 * the host snaps the caret to the buffer start manually.
 */
export function fLeadingBadgeEdgeOffset(pSource: string, pCaret: number): number | null {
  const [vFirst] = fTokenizeContent(pSource);

  if (!vFirst || vFirst.kind !== ChatFormInputRichTokenKind.BADGE) return null;

  return pCaret === fBadgeSourceLength(vFirst.name, vFirst.path) ? 0 : null;
}

/**
 * Translate a plain-text offset into a degenerate `Range` at that
 * position in the DOM; out-of-range offsets clamp to buffer end (before
 * a trailing escape hatch, not after it). Zero offset lands BEFORE a
 * badge or code span, and an offset exactly at a code span's end lands
 * AFTER it, so typing at a code span's edge extends the surrounding
 * text. Interior code-span offsets land in the element's text.
 * Understands the same block/`<br>` newline shapes as
 * `serializeContent`.
 */
export function fTextOffsetToRange(pRoot: HTMLElement, pOffset: number): Range {
  const cRange = document.createRange();

  let vRemaining = pOffset;
  let vLanded = false;
  let vPendingBlockBoundary = false;

  const fLand = (node: Node, pNodeOffset: number) => {
    cRange.setStart(node, pNodeOffset);
    cRange.setEnd(node, pNodeOffset);
    vLanded = true;
  };
  const fWalk = (pParent: Node) => {
    let vFirst = true;

    for (const cChild of Array.from(pParent.childNodes)) {
      if (vLanded) return;

      if (cChild.nodeType === Node.TEXT_NODE) {
        const text = cChild.textContent ?? '';

        if (text.length === 0) continue;

        if (vPendingBlockBoundary) {
          // The synthesized separator maps to the near edge of the
          // content that follows the block.
          vPendingBlockBoundary = false;

          if (vRemaining === 0) {
            fLand(cChild, 0);

            return;
          }

          vRemaining -= 1;
        }

        if (vRemaining <= text.length) {
          fLand(cChild, vRemaining);

          return;
        }

        vRemaining -= text.length;
        vFirst = false;

        continue;
      }

      if (cChild.nodeType !== Node.ELEMENT_NODE) continue;

      const cEl = cChild as HTMLElement;

      if (cEl.getAttribute(cMentionBadgeDataAttrs.BADGE) === BooleanString.cTrue) {
        const cLen = fBadgeSourceLength(
          cEl.getAttribute(cMentionBadgeDataAttrs.NAME) ?? '',
          cEl.getAttribute(cMentionBadgeDataAttrs.PATH) ?? ''
        );

        if (cLen === 0) continue;

        if (vPendingBlockBoundary) {
          vPendingBlockBoundary = false;

          if (vRemaining === 0) {
            cRange.setStartBefore(cEl);
            cRange.setEndBefore(cEl);
            vLanded = true;

            return;
          }

          vRemaining -= 1;
        }

        if (vRemaining <= cLen) {
          if (vRemaining === 0) {
            cRange.setStartBefore(cEl);
            cRange.setEndBefore(cEl);
          } else {
            cRange.setStartAfter(cEl);
            cRange.setEndAfter(cEl);
          }

          vLanded = true;

          return;
        }

        vRemaining -= cLen;
        vFirst = false;

        continue;
      }

      const cCodeToken = cEl.getAttribute(cCodeTokenAttr);

      if (cCodeToken !== null) {
        const cIsBlock = cCodeToken === ChatFormInputRichTokenKind.cCodeBlock;

        if (cIsBlock && (vPendingBlockBoundary || !vFirst)) {
          vPendingBlockBoundary = false;

          if (vRemaining === 0) {
            cRange.setStartBefore(cEl);
            cRange.setEndBefore(cEl);
            vLanded = true;

            return;
          }

          vRemaining -= 1;
        }

        const cLen = (cEl.textContent ?? '').length;

        if (vRemaining === 0) {
          cRange.setStartBefore(cEl);
          cRange.setEndBefore(cEl);
          vLanded = true;

          return;
        }

        if (vRemaining === cLen) {
          cRange.setStartAfter(cEl);
          cRange.setEndAfter(cEl);
          vLanded = true;

          return;
        }

        if (vRemaining < cLen) {
          fWalk(cEl);

          return;
        }

        vRemaining -= cLen;

        if (cIsBlock) vRemaining -= 1;

        vFirst = false;

        continue;
      }

      if (cEl.tagName === 'BR') {
        const cIsHatch =
          fIsCodeBlockElement(cEl.previousSibling) || fIsCodeBlockElement(cEl.nextSibling);

        if (cIsHatch) {
          // Escape hatch: no source length; offset 0 lands before it
          // so text typed there takes its place.
          if (vRemaining === 0) {
            cRange.setStartBefore(cEl);
            cRange.setEndBefore(cEl);
            vLanded = true;
          }

          continue;
        }

        if (!cEl.nextSibling) continue;

        if (vPendingBlockBoundary) {
          vPendingBlockBoundary = false;

          if (vRemaining === 0) {
            cRange.setStartBefore(cEl);
            cRange.setEndBefore(cEl);
            vLanded = true;

            return;
          }

          vRemaining -= 1;
        }

        if (vRemaining === 0) {
          cRange.setStartBefore(cEl);
          cRange.setEndBefore(cEl);
          vLanded = true;

          return;
        }

        vRemaining -= 1;
        vFirst = false;

        continue;
      }

      if (sBlockTagNames.has(cEl.tagName)) {
        if (vPendingBlockBoundary || !vFirst) {
          vPendingBlockBoundary = false;

          if (vRemaining === 0) {
            // The boundary newline belongs to the previous line.
            cRange.setStartBefore(cEl);
            cRange.setEndBefore(cEl);
            vLanded = true;

            return;
          }

          vRemaining -= 1;
        }

        fWalk(cEl);
        vFirst = false;

        continue;
      }

      const cBefore = vRemaining;

      fWalk(cEl);

      if (vRemaining < cBefore) vFirst = false;
    }
  };

  fWalk(pRoot);

  if (!vLanded) {
    const cLast = pRoot.lastChild;

    if (cLast && cLast.nodeName === 'BR') {
      cRange.setStartBefore(cLast);
      cRange.setEndBefore(cLast);
    } else {
      cRange.selectNodeContents(pRoot);
      cRange.collapse(false);
    }
  }

  return cRange;
}
