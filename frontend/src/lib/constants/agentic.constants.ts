import type { AgenticConfig } from '$lib/types/agentic';

export const cAttachmentSavedRegex = /\[Attachment saved: ([^\]]+)\]/;

// JSON detection: an attachment placeholder also starts with `[`, but is
// plain text (`[Attachment saved: ...]`), not an array literal. Require the
// first array value (or the closing bracket for an empty array) to look like
// a valid JSON token before attempting JSON.parse.
export const cToolResultJsonOpenRegex = /^(?:\{|\[\s*(?:[[\]"{\-0-9]|true|false|null))/;

// Search-summary wire format used by file-glob and grep tools:
//   <matches>
//   ---
//   Total matches: N
export const cSearchSummary = {
  SEPARATOR: '---\n',
  TOTAL_REGEX: /Total matches:\s*(\d+)/
} as const;

// Separator rendered between stats in the tool-result footer (e.g. between a
// result message and the byte/edit count). Plain ASCII spaces bracket a hyphen
// so the whole " - " sits on one visual line even when the surrounding text
// wraps mid-paragraph.
export const cResultStatSeparator = ' - ';

// Separator between the assistant text sections of a grouped agentic
// session when they are joined for the clipboard.
export const cAgenticTextCopySeparator = '\n\n';

export const cDefaultAgenticConfig: AgenticConfig = {
  enabled: true,
  maxTurns: 100
} as const;

export const cReasoningTags = {
  END: '</think>',
  START: '<think>'
} as const;
