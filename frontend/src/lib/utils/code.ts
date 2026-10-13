import { cCodeBlock, cNewline } from '$lib/constants';
import hljs from 'highlight.js';

export interface IncompleteCodeBlock {
  language: string;
  code: string;
  openingIndex: number;
}

// A fence line: up to 3 leading spaces (CommonMark), 3+ backticks, then
// whatever trails on the same line.
const cFenceLineRegex = /^ {0,3}(`{3,})(.*)$/;

/**
 * Splits text glued to a closing code fence onto its own line:
 *
 *     ```ts
 *     let foo = 'bar';
 *     ```create this file on ...
 *
 * A closing fence with trailing text is not a fence to the markdown
 * parser, so the block would swallow the text as code. The chat form
 * normally keeps the fence on its own line, but older messages and
 * hand-pasted content can carry the glued form.
 *
 * Only trailing text containing whitespace is split: a single word
 * after the backticks inside a fenced block is more likely nested
 * markdown (a ```python example inside a ```md block) than glued prose.
 */
export function fSplitGluedClosingCodeFences(pMarkdown: string): string {
  if (!pMarkdown.includes('```')) return pMarkdown;

  const aLines = pMarkdown.split(cNewline);

  let vInside = false;
  let vChanged = false;

  for (let vI = 0; vI < aLines.length; vI++) {
    const cMatch = cFenceLineRegex.exec(aLines[vI]);

    if (!cMatch) continue;

    if (!vInside) {
      vInside = true;

      continue;
    }

    vInside = false;

    const cTrailing = cMatch[2];

    if (cTrailing.includes('`') || !/\s/.test(cTrailing)) continue;

    aLines[vI] = aLines[vI].slice(0, aLines[vI].length - cTrailing.length);
    aLines.splice(vI + 1, 0, cTrailing.trim());
    vI++;
    vChanged = true;
  }

  return vChanged ? aLines.join(cNewline) : pMarkdown;
}

/**
 * Strips empty lines (whitespace-only) from the start and end of code.
 *
 * Tool call payloads frequently arrive with surrounding whitespace from LLM
 * formatting (`"\nfunction ...\n"`). Preserving those newlines makes hljs emit
 * a leading/trailing empty line that `<pre>` then renders as a phantom row,
 * pushing real content away from the box edge. The trim keeps the body intact
 * so internal blank lines are still rendered as such.
 */
function fTrimCodePadding(pCode: string): string {
  return pCode
    .replace(cCodeBlock.TRIM_LEADING_PADDING_REGEX, '')
    .replace(cCodeBlock.TRIM_TRAILING_PADDING_REGEX, '');
}

function fEscapeCode(pCode: string): string {
  return pCode
    .replace(cCodeBlock.AMPERSAND_REGEX, '&amp;')
    .replace(cCodeBlock.LT_REGEX, '&lt;')
    .replace(cCodeBlock.GT_REGEX, '&gt;');
}

/** Bounded cache for highlightCode results. */
const cHighlightCacheMaxSize = 64;
const dHighlightCache = new Map<string, string>();

/**
 * Highlights code using highlight.js
 * @param code - The code to highlight
 * @param language - The programming language
 * @param autoDetect - Fall back to `highlightAuto` when `language` is unknown.
 *   Callers rendering a still-streaming block should pass false: auto-detection
 *   costs ~38ms per call and re-guesses on every chunk, so the language (and
 *   therefore the whole highlight) flickers as the block grows.
 * @returns HTML string with syntax highlighting
 */
export function fHighlightCode(pCode: string, pLanguage: string, pAutoDetect = true): string {
  if (!pCode) return '';

  // Cache key includes language and autoDetect flag since results differ.
  // During streaming, the same code string may be highlighted repeatedly
  // (e.g., when text after a code block changes but the code itself doesn't).
  const cCacheKey = `${pLanguage}:${pAutoDetect}:${pCode}`;
  const cCached = dHighlightCache.get(cCacheKey);

  if (cCached) return cCached;

  const cTrimmed = fTrimCodePadding(pCode);

  let vResult: string;

  try {
    const cLang = pLanguage.toLowerCase();
    const cIsSupported = hljs.getLanguage(cLang);

    if (cIsSupported) {
      vResult = hljs.highlight(cTrimmed, { language: cLang }).value;
    } else if (pAutoDetect) {
      vResult = hljs.highlightAuto(cTrimmed).value;
    } else {
      vResult = fEscapeCode(cTrimmed);
    }
  } catch {
    vResult = fEscapeCode(cTrimmed);
  }

  if (dHighlightCache.size >= cHighlightCacheMaxSize) {
    dHighlightCache.delete(dHighlightCache.keys().next().value!);
  }

  dHighlightCache.set(cCacheKey, vResult);

  return vResult;
}

export { fTrimCodePadding };

/**
 * Detects if markdown ends with an incomplete code block (opened but not closed).
 * Returns the code block info if found, null otherwise.
 * @param markdown - The raw markdown string to check
 * @returns IncompleteCodeBlock info or null
 */
export function fDetectIncompleteCodeBlock(pMarkdown: string): IncompleteCodeBlock | null {
  // Count all code fences in the markdown
  // A code block is incomplete if there's an odd number of ``` fences
  const cFencePattern = new RegExp(cCodeBlock.FENCE_PATTERN.source, cCodeBlock.FENCE_PATTERN.flags);
  const aFences: number[] = [];

  let vFenceMatch;

  while ((vFenceMatch = cFencePattern.exec(pMarkdown)) !== null) {
    // Store the position after the ```
    const cPos = vFenceMatch[0].startsWith(cNewline) ? vFenceMatch.index + 1 : vFenceMatch.index;

    aFences.push(cPos);
  }

  // If even number of fences (including 0), all code blocks are closed
  if (aFences.length % 2 === 0) {
    return null;
  }

  // Odd number means last code block is incomplete
  // The last fence is the opening of the incomplete block
  const cOpeningIndex = aFences[aFences.length - 1];
  const cAfterOpening = pMarkdown.slice(cOpeningIndex + 3);
  // Extract language and code content
  const cLangMatch = cAfterOpening.match(cCodeBlock.LANG_PATTERN);
  const cLanguage = cLangMatch?.[1] || cCodeBlock.DEFAULT_LANGUAGE;
  const cCodeStartIndex = cOpeningIndex + 3 + (cLangMatch?.[0]?.length ?? 0);
  const cCode = pMarkdown.slice(cCodeStartIndex);

  return {
    code: cCode,
    language: cLanguage,
    openingIndex: cOpeningIndex
  };
}
