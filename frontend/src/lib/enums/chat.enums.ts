export enum ChatMessageStatsView {
  cGeneration = 'generation',
  cReading = 'reading',
  cSummary = 'summary',
  cTools = 'tools'
}

export enum ChatMessageStatisticsMode {
  cGeneration = 'generation',
  cReading = 'reading',
  cSwitchable = 'switchable'
}

/**
 * Connection state of a streamed completion, drives the resume status indicator.
 */
export enum StreamConnectionState {
  cLost = 'lost',
  cResuming = 'resuming',
  cStreaming = 'streaming'
}

/**
 * Reasoning format options for API requests.
 */
export enum ReasoningFormat {
  cAuto = 'auto',
  cNone = 'none'
}

/**
 * Message roles for chat messages.
 */
export enum MessageRole {
  cAssistant = 'assistant',
  cSystem = 'system',
  TOOL = 'tool',
  cUser = 'user'
}

/**
 * Message types for different content kinds.
 */
export enum MessageType {
  ROOT = 'root',
  cSystem = 'system',
  TEXT = 'text',
  cThink = 'think'
}

/**
 * Content part types for API chat message content.
 */
export enum ContentPartType {
  cImageUrl = 'image_url',
  cInputAudio = 'input_audio',
  cInputVideo = 'input_video',
  TEXT = 'text'
}

/**
 * Error dialog types for displaying server/timeout errors.
 */
export enum ErrorDialogType {
  SERVER = 'server',
  cTimeout = 'timeout'
}

export enum ConversationSelectionMode {
  cExport = 'export',
  cImport = 'import'
}

/**
 * PDF view mode options for previewing PDF attachments.
 */
export enum PdfViewMode {
  cPages = 'pages',
  TEXT = 'text'
}

export enum ChatFormCommandAction {
  CWD = 'cwd',
  MODEL = 'model',
  PROMPT = 'prompt'
}

export enum FileMentionEntryType {
  cDirectory = 'directory',
  FILE = 'file'
}

/**
 * Kinds of tokens the chat-form-input-rich produces.
 */
export enum ChatFormInputRichTokenKind {
  BADGE = 'badge',
  cCodeBlock = 'code_block',
  cCodeInline = 'code_inline',
  TEXT = 'text'
}
