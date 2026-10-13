/**
 * Unified exports for all utility functions
 * Import utilities from '$lib/utils' for cleaner imports
 *
 * For browser-only utilities (pdf-processing, audio-recording, svg-to-png,
 * webp-to-png, process-uploaded-files, convert-files-to-extra), use:
 * import { ... } from '$lib/utils/browser-only'
 */

// API utilities
export { fGetAuthHeaders, fGetJsonHeaders, fSanitizeHeaders } from './api-headers';
export { ApiError, fApiDelete, fApiFetch, fApiFetchWithParams, fApiPost } from './api-fetch';
export { fValidateApiKey } from './api-key-validation';

// Attachment utilities
export { fGetAttachmentDisplayItems, fIsMcpPrompt, fIsMcpResource } from './attachment-display';
export {
  fIsTextFile,
  fIsImageFile,
  fIsPdfFile,
  fIsAudioFile,
  fIsVideoFile
} from './attachment-type';

// Textarea utilities
export { default as fAutoResizeTextarea } from './autoresize-textarea';

// Branching utilities
export {
  fFilterByLeafNodeId,
  fFindMessageById,
  fFindLeafNode,
  fFindDescendantMessages,
  fGetMessageSiblings,
  fBuildSiblingInfoMap
} from './branching';

// Code
export {
  fHighlightCode,
  fDetectIncompleteCodeBlock,
  fSplitGluedClosingCodeFences,
  fTrimCodePadding,
  type IncompleteCodeBlock
} from './code';

// Config helpers
export { fSetConfigValue, fGetConfigValue, fConfigToParameterRecord } from './config-helpers';

// CORS Proxy
export { fBuildProxiedIconUrl, fBuildProxiedUrl, fBuildProxiedHeaders } from './cors-proxy';

// URL utilities
export { fExtractRootDomain, fSanitizeExternalUrl, fCanonicalizeServerUrl } from './url';

// Model list helpers
export {
  fFilterModelOptions,
  fGroupFavoriteOptions,
  fGroupModelOptions,
  fWindowLocalGroups,
  type GroupedModelOptions,
  type ModelItem,
  type OrgGroup
} from './model-list';

// Progress helpers
export { fModelLoadFraction, fModelLoadProgressText } from './progress';

// Conversation utilities
export {
  fCreateMessageCountMap,
  fGetMessageCount,
  fGetConversationModel,
  fBuildConversationTree,
  type ConversationTreeItem
} from './conversation-utils';

// Clipboard utilities
export {
  fCopyToClipboard,
  fCopyCodeToClipboard,
  fFormatMessageForClipboard,
  fParseClipboardContent,
  fHasClipboardAttachments
} from './clipboard';

// File preview utilities
export { fGetFileTypeLabel } from './file-preview';
export { fGetPreviewText, fGenerateConversationTitle } from './text';

// File type utilities
export {
  fGetFileTypeCategory,
  fGetFileTypeCategoryByExtension,
  fGetFileTypeByExtension,
  fIsFileTypeSupported
} from './file-type';

// Formatting utilities
export {
  fFormatFileSize,
  fFormatParameters,
  fFormatNumber,
  fFormatJsonPretty,
  fFormatTime,
  fFormatPerformanceTime,
  fFormatAttachmentText
} from './formatters';

// IME utilities
export { fIsIMEComposing } from './is-ime-composing';

// LaTeX utilities
export { fMaskInlineLaTeX, fPreprocessLaTeX } from './latex-protection';

// Modality file validation utilities
export {
  fIsFileTypeSupportedByModel,
  fFilterFilesByModalities,
  fGenerateModalityErrorMessage
} from './modality-file-validation';

// Model name utilities
export { fIsValidModelName, fNormalizeModelName, fOrgOf, fRepoOf } from './model-names';

// Sidecar token utilities
export { fIsAuxSidecar, fIsDraftSidecar, fSidecarFromFileToken, fSidecarFromTag } from './sidecars';

// Portal utilities
export { fPortalToBody } from './portal-to-body';

// Precision utilities
export { fNormalizeFloatingPoint, fNormalizeNumber } from './precision';

// Syntax highlighting utilities
export { fGetLanguageFromFilename } from './syntax-highlight-language';

// Text file utilities
export { fIsTextFileByName, fReadFileAsText, fIsLikelyTextFile } from './text-files';

// Debounce utilities
export { fDebounce } from './debounce';

// Sanitization utilities
export { fSanitizeKeyValuePairKey, fSanitizeKeyValuePairValue } from './sanitize';

// Image error fallback utilities
export { fGetImageErrorFallbackHtml } from './image-error-fallback';

// SSE-with-JSON stream iterator (used by server tool streaming, decoupled
// from chat.service.ts which embeds its own SSE parser for resume support)
export { fExtractSseDataPayload, fParseSseJsonStream, fSplitSseRecords } from './sse';

// Stream session identity (conversation-id based)
export { fStreamIdentity } from './stream-identity';

// MCP utilities
export {
  fDetectMcpTransportFromUrl,
  fParseMcpServerSettings,
  fGetMcpLogLevelIcon,
  fGetMcpLogLevelClass,
  fIsImageMimeType,
  fParseResourcePath,
  fGetDisplayName,
  fGetResourceDisplayName,
  fIsCodeResource,
  fIsImageResource,
  fGetResourceIcon,
  fGetResourceTextContent,
  fGetResourceBlobContent,
  fDownloadResourceContent,
  fGetMcpIconUrl,
  fGetMcpServerFaviconFallback,
  fGetMcpServerLabel
} from './mcp';

// URI Template utilities
export {
  fExtractTemplateVariables,
  fExpandTemplate,
  fIsTemplateComplete,
  fNormalizeResourceUri,
  type UriTemplateVariable
} from './uri-template';

// Data URL utilities
export { fCreateBase64DataUrl } from './data-url';

// Header utilities
export { fParseHeadersToArray, fSerializeHeaders } from './headers';

// Working-directory display helpers (HOME-style tilde abbreviation)
export {
  fAbbreviateWorkingDir,
  fAbbreviateHome,
  fLastPathSegment,
  fFormatCwdMessage,
  fParseCwdMessage,
  cCwdChangedPrefix,
  cCwdClearedText,
  type CwdMessageInfo
} from './path-display';

// Working-directory picker search helpers
export {
  fSplitPathQuery,
  fBuildCaseInsensitiveGlob,
  fBuildGlobSearchArgs,
  fRankEntries,
  fJoinPath,
  fHighlightMatch,
  type PathQuery
} from './working-directory';

// Shared `file_glob_search` runner with a short-lived result cache
export { fRunGlobSearch, fRunGlobSearchWithChildren } from './glob-search';

// Mention-token detection (for the `@`-triggered file/folder mention picker)
export {
  fFindMentionToken,
  fTakeMentionDismissSnapshot,
  type MentionDismissSnapshot
} from './mention-token';

// Slash-command token detection (for the `/`-triggered command picker)
export {
  fFindCommandToken,
  fTakeCommandDismissSnapshot,
  type CommandDismissSnapshot
} from './command-token';

// Tokenization for the ChatFormInputRich (mention links + code spans <-> chip DOM)
export {
  fTokenizeContent,
  fContainsCodeSpan,
  fIsOffsetInCodeBlock,
  fDomMatchesTokens,
  fSyncCodeBlockHatches,
  fStripBlockBoundaryLineBreaks,
  fSerializeContent,
  fBuildFragment,
  fRangeToTextOffset,
  fTextOffsetToRange,
  fBadgeAwareWordJump,
  fLeadingBadgeEdgeOffset
} from './chat-form-input-rich-tokenizer';

// Source-space undo/redo history for the ChatFormInputRich
export { SourceHistory, type SourceHistoryEntry } from './source-history';

// Mention-badge visual contract (used by the ChatFormInputRich / rehype
// DOM paths that build the same chip without a Svelte mount)
export {
  fContainsFileMentionLink,
  fFileMentionLinkRe,
  fEncodeFileLinkPath,
  fDecodeFileLinkPath,
  cMentionBadgeClassname,
  cMentionBadgeIconClassname,
  cMentionBadgeSvgAttributes,
  aMentionBadgeFileIconPaths,
  aMentionBadgeFolderIconPaths,
  fGetMentionBadgeIconPaths,
  fGetMentionBadgeLabel,
  fSplitMentionSegments,
  fBuildMentionInsertion
} from './mention-badge';

// Chat template utilities
export {
  fDetectThinkingSupport,
  fDetectThinkingSupportWithReason
} from './chat-template-thinking-detector';

// Agentic content utilities (structured section derivation)
export {
  fDeriveAgenticSections,
  fBuildAssistantRawOutput,
  fParseToolResultWithMedia,
  fSplitSearchSummaryList,
  fHasAgenticContent,
  fClassifyToolResult,
  fClassifyContinueIntent
} from './agentic';

// Line-level unified diff for tool result rendering (`edit_file` block)
export {
  fComputeLineDiff,
  fPrefixFor,
  fRenderUnifiedDiff,
  type DiffLine
} from './compute-line-diff';

// Partial-incremental JSON parser for streaming tool arguments
export { fParsePartialJsonArgs } from './parse-partial-json-args';

// `exec_shell_command` result parsing
export { fParseExecShellCommandError } from './parse-exec-shell-error';
export {
  fParseExecShellCommandExitStatus,
  fIsExitCodeSummaryLine,
  type ExecShellExitStatus
} from './parse-exec-shell-status';

// Search-result parsing (web-search / fetch MCP tools)
export {
  aSupportedWebSearchToolNames,
  fExtractSearchResults,
  fExtractSearchQuery,
  fFaviconForUrl,
  fIsWebSearchToolName,
  fLooksLikeSearchResult
} from './search-results';

// Cache utilities
export { TTLCache, ReactiveTTLMap } from './cache-ttl';

// Redaction utilities
export { fRedactValue } from './redact';

// Request inspection utilities
export {
  fGetRequestUrl,
  fGetRequestMethod,
  fGetRequestBody,
  fSummarizeRequestBody,
  fFormatDiagnosticErrorMessage,
  fExtractJsonRpcMethods,
  type RequestBodySummary
} from './request-helpers';

// Abort signal utilities
export {
  fThrowIfAborted,
  fIsAbortError,
  fCreateLinkedController,
  fCreateTimeoutSignal,
  fWithAbortSignal
} from './abort';

// Tool-call meta utilities. Parsers for each server tool live next to
// their renderer family under
// `src/lib/components/app/chat/ChatMessages/ChatMessage/ChatMessageToolCall/parsers/`.
// This module only carries the helpers that genuinely cross tool
// boundaries (currently: parsing the tool-result blob into a JSON
// object).
export { fTryParseToolResultObject } from './tool-call-meta';

// Per-tool UI metadata (label + icon) used by the tool-call chrome.
// Re-exported through $lib/utils so renderer components can read the
// label without depending on $lib/constants directly.
export { fGetToolUi } from './tool-ui';

// Chat command picker

export { fGetChatCommands } from './chat-commands';

// Sandbox tool definition
export { fBuildSandboxToolDefinition } from './sandbox-tool';

// Browser `get_datetime` executor (the browser clock, not the server's)
export { fExecuteGetDatetimeTool } from './get-datetime';

// Browser fallback for the server's get_info tool
export { fDetectOs, fExecuteBrowserInfoTool } from './browser-info';

// Tool-use support detection from a chat template
export { fDetectToolUseSupport } from './chat-template-tool-detector';

// Model memory estimation
export { fMinMemoryTierGb } from './model-compatibility';

// Cryptography utilities

export { fUuid } from './uuid';

// CSS utilities
export { fRemToPx } from './css';

// Audio format helper (used by agentic store and chat service)
export { fGetAudioInputFormat } from './audio-format';
export { fGroupModelFamilies, fModelFamilyKey, type ModelFamilyGroup } from './model-families';

// Svelte actions
export { fNearViewport } from './near-viewport';
