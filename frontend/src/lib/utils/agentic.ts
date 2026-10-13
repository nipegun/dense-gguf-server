import {
  cAttachmentSavedRegex,
  MARKDOWN,
  cNewline,
  cReasoningTags,
  cSearchSummary,
  cToolResultJsonOpenRegex
} from '$lib/constants';
import {
  AgenticSectionType,
  AttachmentType,
  ContinueIntentKind,
  MessageRole,
  ToolResultKind
} from '$lib/enums';
import type { AgenticSection, ContinueIntent, ToolResultLine } from '$lib/types/agentic';
import type { ApiChatCompletionToolCall } from '$lib/types/api';
import type {
  DatabaseMessage,
  DatabaseMessageExtra,
  DatabaseMessageExtraImageFile
} from '$lib/types/database';

/**
 * Derives display sections from a single assistant message and its direct tool results.
 *
 * @param message - The assistant message
 * @param toolMessages - Tool result messages for this assistant's tool_calls
 * @param streamingToolCalls - Partial tool calls during streaming (not yet persisted)
 */
function fDeriveSingleTurnSections(
  pMessage: DatabaseMessage,
  pToolMessages: DatabaseMessage[] = [],
  pStreamingToolCalls: ApiChatCompletionToolCall[] = [],
  pIsStreaming: boolean = false
): AgenticSection[] {
  const aSections: AgenticSection[] = [];

  // 1. Reasoning content (from dedicated field)
  if (pMessage.reasoningContent) {
    const aToolCalls = fParseToolCalls(pMessage.toolCalls);
    const cHasContentAfterReasoning =
      !!pMessage.content?.trim() || aToolCalls.length > 0 || pStreamingToolCalls.length > 0;
    const cIsPending = pIsStreaming && !cHasContentAfterReasoning;

    aSections.push({
      content: pMessage.reasoningContent,
      type: cIsPending ? AgenticSectionType.cReasoningPending : AgenticSectionType.REASONING,
      wasInterrupted: !pIsStreaming && !cHasContentAfterReasoning
    });
  }

  // 2. Text content
  if (pMessage.content?.trim()) {
    aSections.push({
      content: pMessage.content,
      type: AgenticSectionType.TEXT
    });
  }

  // 3. Persisted tool calls (from message.toolCalls field)
  const aToolCalls = fParseToolCalls(pMessage.toolCalls);
  // Index tool messages by toolCallId for O(1) lookup instead of O(n) find()
  const dToolMsgById = new Map<string, DatabaseMessage>();

  for (const cTm of pToolMessages) {
    if (cTm.toolCallId && !dToolMsgById.has(cTm.toolCallId)) {
      dToolMsgById.set(cTm.toolCallId, cTm);
    }
  }

  for (const cTc of aToolCalls) {
    const cResultMsg = cTc.id ? dToolMsgById.get(cTc.id) : undefined;
    // Only show as pending/loading if we're actively streaming; otherwise it's just a tool call without result
    const cType = cResultMsg
      ? AgenticSectionType.TOOL_CALL
      : pIsStreaming
        ? AgenticSectionType.cToolCallPending
        : AgenticSectionType.TOOL_CALL;

    aSections.push({
      content: cResultMsg?.content || '',
      toolArgs: cTc.function?.arguments,
      toolCallId: cTc.id,
      toolCwd: cResultMsg?.toolCwd,
      toolName: cTc.function?.name,
      toolResult: cResultMsg?.content,
      toolResultExtras: cResultMsg?.extra,
      type: cType
    });
  }

  // 4. Streaming tool calls (not yet persisted - currently being received)
  const sPersistedIds = new Set(aToolCalls.map((pT) => pT.id).filter(Boolean));

  for (const cTc of pStreamingToolCalls) {
    // Skip if already in persisted tool calls
    if (cTc.id && sPersistedIds.has(cTc.id)) continue;

    aSections.push({
      content: '',
      toolArgs: cTc.function?.arguments,
      toolCallId: cTc.id,
      toolName: cTc.function?.name,
      type: AgenticSectionType.cToolCallStreaming
    });
  }

  return aSections;
}

interface TurnSectionsCacheEntry {
  content: string | undefined;
  extra: DatabaseMessageExtra[] | undefined;
  reasoningContent: string | undefined;
  toolCalls: string | undefined;
  toolMessageContents: (string | undefined)[];
  toolMessageExtras: (DatabaseMessageExtra[] | undefined)[];
  toolMessages: DatabaseMessage[];
  sections: AgenticSection[];
}

const cTurnSectionsCache = new WeakMap<DatabaseMessage, TurnSectionsCacheEntry>();

function fIsTurnCacheValid(
  pEntry: TurnSectionsCacheEntry,
  pMessage: DatabaseMessage,
  pToolMessages: DatabaseMessage[]
): boolean {
  if (
    pEntry.content !== pMessage.content ||
    pEntry.reasoningContent !== pMessage.reasoningContent ||
    pEntry.toolCalls !== pMessage.toolCalls ||
    pEntry.extra !== pMessage.extra
  ) {
    return false;
  }

  if (pEntry.toolMessages.length !== pToolMessages.length) return false;

  for (let vI = 0; vI < pToolMessages.length; vI++) {
    if (pEntry.toolMessages[vI] !== pToolMessages[vI]) return false;

    if (pEntry.toolMessageContents[vI] !== pToolMessages[vI].content) return false;

    if (pEntry.toolMessageExtras[vI] !== pToolMessages[vI].extra) return false;
  }

  return true;
}

/**
 * deriveSingleTurnSections with structural reuse for completed turns.
 *
 * deriveAgenticSections runs in a $derived invalidated per streamed chunk, but
 * only the last turn actually changes. Messages mutate in place and are never
 * replaced, so a WeakMap keyed by the turn's assistant message plus reference
 * checks on every field deriveSingleTurnSections reads detects any change. A
 * cache hit also returns the same section objects, keeping downstream props
 * stable so tool blocks skip their per-chunk re-derive. The streaming turn
 * recomputes uncached on every chunk.
 */
function fDeriveTurnSections(
  pMessage: DatabaseMessage,
  pToolMessages: DatabaseMessage[],
  pStreamingToolCalls: ApiChatCompletionToolCall[],
  pIsStreaming: boolean
): AgenticSection[] {
  if (pIsStreaming || pStreamingToolCalls.length > 0) {
    return fDeriveSingleTurnSections(pMessage, pToolMessages, pStreamingToolCalls, pIsStreaming);
  }

  const cCached = cTurnSectionsCache.get(pMessage);

  if (cCached && fIsTurnCacheValid(cCached, pMessage, pToolMessages)) {
    return cCached.sections;
  }

  const aSections = fDeriveSingleTurnSections(pMessage, pToolMessages, [], false);

  cTurnSectionsCache.set(pMessage, {
    content: pMessage.content,
    extra: pMessage.extra,
    reasoningContent: pMessage.reasoningContent,
    sections: aSections,
    toolCalls: pMessage.toolCalls,
    toolMessageContents: pToolMessages.map((pTm) => pTm.content),
    toolMessageExtras: pToolMessages.map((pTm) => pTm.extra),
    toolMessages: pToolMessages
  });

  return aSections;
}

/**
 * Derives display sections from structured message data.
 *
 * Handles both single-turn (one assistant + its tool results) and multi-turn
 * agentic sessions (multiple assistant + tool messages grouped together).
 *
 * When `toolMessages` contains continuation assistant messages (from multi-turn
 * agentic flows), they are processed in order to produce sections across all turns.
 *
 * @param message - The first/anchor assistant message
 * @param toolMessages - Tool result messages and continuation assistant messages
 * @param streamingToolCalls - Partial tool calls during streaming (not yet persisted)
 * @param isStreaming - Whether the message is currently being streamed
 */
export function fDeriveAgenticSections(
  pMessage: DatabaseMessage,
  pToolMessages: DatabaseMessage[] = [],
  pStreamingToolCalls: ApiChatCompletionToolCall[] = [],
  pIsStreaming: boolean = false
): AgenticSection[] {
  const cHasAssistantContinuations = pToolMessages.some((pM) => pM.role === MessageRole.cAssistant);

  if (!cHasAssistantContinuations) {
    return fDeriveTurnSections(pMessage, pToolMessages, pStreamingToolCalls, pIsStreaming);
  }

  const aSections: AgenticSection[] = [];
  const aFirstTurnToolMsgs = fCollectToolMessages(pToolMessages, 0);

  aSections.push(...fDeriveTurnSections(pMessage, aFirstTurnToolMsgs, [], false));

  let vI = aFirstTurnToolMsgs.length;

  while (vI < pToolMessages.length) {
    const cMsg = pToolMessages[vI];

    if (cMsg.role === MessageRole.cAssistant) {
      const aTurnToolMsgs = fCollectToolMessages(pToolMessages, vI + 1);
      const cIsLastTurn = vI + 1 + aTurnToolMsgs.length >= pToolMessages.length;

      aSections.push(
        ...fDeriveTurnSections(
          cMsg,
          aTurnToolMsgs,
          cIsLastTurn ? pStreamingToolCalls : [],
          cIsLastTurn && pIsStreaming
        )
      );

      vI += 1 + aTurnToolMsgs.length;
    } else {
      vI++;
    }
  }

  return aSections;
}

/**
 * Build the raw text representation shown in the "raw output" view of an
 * assistant message. Each section is formatted as it would appear in the
 * model-facing transcript, joined by blank lines.
 */
export function fBuildAssistantRawOutput(pSections: AgenticSection[]): string {
  const aParts: string[] = [];

  for (const cSection of pSections) {
    switch (cSection.type) {
      case AgenticSectionType.REASONING:
      case AgenticSectionType.cReasoningPending:
        aParts.push(`${cReasoningTags.START}${cNewline}${cSection.content}${cReasoningTags.END}`);

        break;

      case AgenticSectionType.TEXT:
        aParts.push(cSection.content);

        break;

      case AgenticSectionType.TOOL_CALL:
      case AgenticSectionType.cToolCallPending:
      case AgenticSectionType.cToolCallStreaming: {
        const dCallObj: Record<string, unknown> = { name: cSection.toolName };

        if (cSection.toolArgs) {
          try {
            dCallObj.arguments = JSON.parse(cSection.toolArgs);
          } catch {
            dCallObj.arguments = cSection.toolArgs;
          }
        }

        aParts.push(JSON.stringify(dCallObj, null, 2));

        if (cSection.toolResult) {
          aParts.push(`${cNewline}${cSection.toolResult}`);
        }

        break;
      }
    }
  }

  return aParts.join(`${cNewline}${cNewline}`);
}

/**
 * Collect consecutive tool messages starting at `startIndex`.
 */
function fCollectToolMessages(
  pMessages: DatabaseMessage[],
  pStartIndex: number
): DatabaseMessage[] {
  const aResult: DatabaseMessage[] = [];

  for (let vI = pStartIndex; vI < pMessages.length; vI++) {
    if (pMessages[vI].role === MessageRole.TOOL) {
      aResult.push(pMessages[vI]);
    } else {
      break;
    }
  }

  return aResult;
}

/**
 * Split a tool-result blob into a list and an optional "Total matches: N"
 * summary. Both file-glob and grep tools emit this format on the server:
 *
 *   <matches>
 *   ---
 *   Total matches: 42
 *
 * Returns the lines and exposes a callback for capturing the total so each
 * caller can stash it on its own meta type without taking a return-tuple.
 */
export function fSplitSearchSummaryList(
  pText: string,
  pCaptureTotal: (pN: number) => void
): { lines: string[] } {
  const cSeparatorIndex = pText.indexOf(cSearchSummary.SEPARATOR);
  const cMatchesText = cSeparatorIndex === -1 ? pText : pText.slice(0, cSeparatorIndex);
  const cSummaryText =
    cSeparatorIndex === -1 ? '' : pText.slice(cSeparatorIndex + cSearchSummary.SEPARATOR.length);
  const cTotalMatch = cSummaryText.match(cSearchSummary.TOTAL_REGEX);

  if (cTotalMatch) {
    pCaptureTotal(parseInt(cTotalMatch[1], 10));
  }

  const aLines = cMatchesText
    .split(cNewline)
    .map((pLine) => pLine.trim())
    .filter((pLine) => pLine.length > 0);

  return { lines: aLines };
}

/** Bounded cache for parseToolResultWithMedia results. */
const cToolResultLinesCacheMaxSize = 32;
const dToolResultLinesCache = new Map<string, ToolResultLine[]>();

/**
 * Parse tool result text into lines, matching media attachments (images and audio) by name.
 * Memoized: called per render during streaming on unchanged tool result
 * strings with unchanged extras.
 */
export function fParseToolResultWithMedia(
  pToolResult: string,
  pExtras?: DatabaseMessageExtra[]
): ToolResultLine[] {
  // Cache key includes image attachment names so we recompute when
  // attachments change, even if the count stays the same.
  const cImageNames = (pExtras ?? [])
    .filter((pE): pE is DatabaseMessageExtraImageFile => pE.type === AttachmentType.IMAGE)
    .map((pE) => pE.name)
    .join(cNewline);
  const cCacheKey = `${cImageNames}:${pToolResult}`;
  const cCached = dToolResultLinesCache.get(cCacheKey);

  if (cCached !== undefined) return cCached;

  const aLines = pToolResult.split(cNewline);
  const aResult = aLines.map((pLine) => {
    const cMatch = pLine.match(cAttachmentSavedRegex);

    if (!cMatch || !pExtras) return { text: pLine };

    const cAttachmentName = cMatch[1];
    const cMedia = pExtras.find(
      (pE): pE is DatabaseMessageExtraImageFile | DatabaseMessageExtraAudioFile =>
        (pE.type === AttachmentType.IMAGE || pE.type === AttachmentType.AUDIO) &&
        pE.name === cAttachmentName
    );

    return { media: cMedia, text: pLine };
  });

  if (dToolResultLinesCache.size >= cToolResultLinesCacheMaxSize) {
    dToolResultLinesCache.delete(dToolResultLinesCache.keys().next().value!);
  }

  dToolResultLinesCache.set(cCacheKey, aResult);

  return aResult;
}

/** Bounded cache for classifyToolResult results. */
const cClassifyCacheMaxSize = 32;
const dClassifyCache = new Map<string, ToolResultKind>();

/**
 * Pick a renderer tier for a tool's result content.
 *
 *   json     - trimmed content starts with `{` or `[` and parses cleanly.
 *   markdown - content shows structural markdown markers (headers, code
 *              fences, links, lists, blockquotes, tables) and should render
 *              through MarkdownContent for proper formatting.
 *   text     - everything else, rendered as plain text lines (with image
 *              attachment resolution as a side effect).
 * Memoized: called per render during streaming on unchanged content.
 */
export function fClassifyToolResult(pContent: string | undefined): ToolResultKind {
  if (!pContent) return ToolResultKind.TEXT;

  const cCached = dClassifyCache.get(pContent);

  if (cCached !== undefined) return cCached;

  const cTrimmed = pContent.trim();

  if (!cTrimmed) return ToolResultKind.TEXT;

  let vResult: ToolResultKind = ToolResultKind.TEXT;

  // Strongest signal: JSON object/array round-trips through JSON.parse.
  if (cToolResultJsonOpenRegex.test(cTrimmed)) {
    try {
      JSON.parse(cTrimmed);
      vResult = ToolResultKind.cJson;
    } catch (error) {
      console.error('[agentic] tool result looked like JSON but failed to parse:', error);
    }
  }

  if (vResult === ToolResultKind.TEXT && fLooksLikeMarkdown(cTrimmed)) {
    vResult = ToolResultKind.MARKDOWN;
  }

  if (dClassifyCache.size >= cClassifyCacheMaxSize) {
    dClassifyCache.delete(dClassifyCache.keys().next().value!);
  }

  dClassifyCache.set(pContent, vResult);

  return vResult;
}

/**
 * Heuristic detector for "is this content a markdown document rather than
 * plain text?". True when at least one well-known structural marker shows
 * up - headers, code fences, links, bold, lists, blockquotes, tables.
 * Each marker is specific enough that plain tool-output prose rarely
 * trips it, but plain text starting with `# 5` will - acceptable false
 * positive for the gain in formatting for tool results like search
 * summaries that come back already-mardown.
 */
function fLooksLikeMarkdown(pContent: string): boolean {
  // Code fences are unambiguous - triple backticks or tildes at line start.
  if (MARKDOWN.CODE_FENCE_REGEX.test(pContent)) return true;

  const aLines = pContent.split(cNewline);

  for (const cLine of aLines) {
    if (MARKDOWN.ATX_HEADING_REGEX.test(cLine)) return true;

    if (MARKDOWN.BLOCKQUOTE_REGEX.test(cLine)) return true;

    if (MARKDOWN.LIST_BULLET_REGEX.test(cLine)) return true;

    if (MARKDOWN.LIST_NUMBERED_REGEX.test(cLine)) return true;
  }

  // Inline structural markers anywhere in the body.
  if (MARKDOWN.LINK_REGEX.test(pContent)) return true;

  if (MARKDOWN.BOLD_REGEX.test(pContent)) return true;

  // Tables: a pipe-bearing header line followed by a separator row.
  if (aLines.length >= 2) {
    const cHead = aLines[0];
    const cSep = aLines[1];

    if (cHead.includes('|') && MARKDOWN.TABLE_SEPARATOR_REGEX.test(cSep)) return true;
  }

  return false;
}

/** Bounded cache for parsed tool-call JSON blobs. */
const cToolCallsCacheMaxSize = 64;
const dToolCallsParseCache = new Map<string, ApiChatCompletionToolCall[]>();

/**
 * Safely parse the toolCalls JSON string from a DatabaseMessage.
 * Memoized: the same JSON string is re-parsed on every render during
 * streaming, which is wasted CPU since tool calls don't change mid-stream.
 */
function fParseToolCalls(pToolCallsJson?: string): ApiChatCompletionToolCall[] {
  if (!pToolCallsJson) return [];

  const cCached = dToolCallsParseCache.get(pToolCallsJson);

  if (cCached) return cCached;

  let aResult: ApiChatCompletionToolCall[];

  try {
    const cParsed = JSON.parse(pToolCallsJson);

    aResult = Array.isArray(cParsed) ? cParsed : [];
  } catch {
    aResult = [];
  }

  if (dToolCallsParseCache.size >= cToolCallsCacheMaxSize) {
    dToolCallsParseCache.delete(dToolCallsParseCache.keys().next().value!);
  }

  dToolCallsParseCache.set(pToolCallsJson, aResult);

  return aResult;
}

/**
 * Check if a message has agentic content (tool calls or is part of an agentic flow).
 */
export function fHasAgenticContent(
  pMessage: DatabaseMessage,
  pToolMessages: DatabaseMessage[] = []
): boolean {
  if (pMessage.toolCalls) {
    const aTc = fParseToolCalls(pMessage.toolCalls);

    if (aTc.length > 0) return true;
  }

  return pToolMessages.length > 0;
}

/**
 * Decide how a Continue click on messages[idx] should resume generation.
 * Pure function over the persisted history snapshot.
 */
export function fClassifyContinueIntent(
  pMessages: DatabaseMessage[],
  pIdx: number
): ContinueIntent {
  const cTarget = pMessages[pIdx];

  // Defensive default: callers already filter by role, stay deterministic.
  if (!cTarget || cTarget.role !== MessageRole.cAssistant) {
    return { kind: ContinueIntentKind.cAppendText };
  }

  const cHasToolCalls = fParseToolCalls(cTarget.toolCalls).length > 0;

  if (!cHasToolCalls) {
    return { kind: ContinueIntentKind.cAppendText };
  }

  // Walk consecutive trailing tool results. The agentic loop only emits tool
  // messages directly after the assistant turn that owns them, so the first
  // non tool message marks the boundary.
  let vLastTrailingTool = pIdx;

  for (let vI = pIdx + 1; vI < pMessages.length; vI++) {
    if (pMessages[vI].role === MessageRole.TOOL) {
      vLastTrailingTool = vI;
    } else {
      break;
    }
  }

  if (vLastTrailingTool > pIdx) {
    return { kind: ContinueIntentKind.cNextTurn, truncateAfter: vLastTrailingTool };
  }

  return { kind: ContinueIntentKind.cRerunTurn, truncateAfter: pIdx - 1 };
}
