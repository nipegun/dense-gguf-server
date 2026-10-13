/**
 * OpenAI-compatible tool call type.
 */
export enum ToolCallType {
  cFunction = 'function'
}

/**
 * Types of sections in agentic content display.
 */
export enum AgenticSectionType {
  REASONING = 'reasoning',
  cReasoningPending = 'reasoning_pending',
  TEXT = 'text',
  TOOL_CALL = 'tool_call',
  cToolCallPending = 'tool_call_pending',
  cToolCallStreaming = 'tool_call_streaming'
}

/**
 * How a Continue click on an assistant message resumes generation.
 */
export enum ContinueIntentKind {
  cAppendText = 'append_text',
  cNextTurn = 'next_turn',
  cRerunTurn = 'rerun_turn'
}

/**
 * Renderer tier for a tool-result blob shown in the default tool-call block.
 */
export enum ToolResultKind {
  cJson = 'json',
  MARKDOWN = 'markdown',
  TEXT = 'text'
}

/**
 * Line classification for the unified-diff renderer of `edit_file` results.
 */
export enum DiffLineKind {
  cAdd = 'add',
  CONTEXT = 'context',
  cRemove = 'remove'
}
