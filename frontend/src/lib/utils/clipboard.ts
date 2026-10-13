import { AttachmentType } from '$lib/enums';
import type {
  ClipboardAttachment,
  ClipboardMcpPromptAttachment,
  ClipboardTextAttachment,
  DatabaseMessageExtra,
  DatabaseMessageExtraMcpPrompt,
  DatabaseMessageExtraMcpResource,
  DatabaseMessageExtraTextFile,
  ParsedClipboardContent
} from '$lib/types';
import { toast } from 'svelte-sonner';

/**
 * Copy text to clipboard with toast notification
 * Uses the current Clipboard API in a secure browser context
 * @param text - Text to copy to clipboard
 * @param successMessage - Custom success message (optional)
 * @param errorMessage - Custom error message (optional)
 * @returns Promise<boolean> - True if successful, false otherwise
 */
export async function fCopyToClipboard(
  pText: string,
  pSuccessMessage = 'Copied to clipboard',
  pErrorMessage = 'Failed to copy to clipboard'
): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(pText);
    toast.success(pSuccessMessage);
    return true;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    toast.error(pErrorMessage);

    return false;
  }
}

/**
 * Copy code with HTML entity decoding and toast notification
 * @param rawCode - Raw code string that may contain HTML entities
 * @param successMessage - Custom success message (optional)
 * @param errorMessage - Custom error message (optional)
 * @returns Promise<boolean> - True if successful, false otherwise
 */
export async function fCopyCodeToClipboard(
  pRawCode: string,
  pSuccessMessage = 'Code copied to clipboard',
  pErrorMessage = 'Failed to copy code'
): Promise<boolean> {
  return fCopyToClipboard(pRawCode, pSuccessMessage, pErrorMessage);
}

/**
 * Formats a message with text attachments for clipboard copying.
 *
 * Default format (asPlainText = false):
 * ```
 * "Text message content"
 * [
 *   {"type":"TEXT","name":"filename.txt","content":"..."},
 *   {"type":"TEXT","name":"another.txt","content":"..."}
 * ]
 * ```
 *
 * Plain text format (asPlainText = true):
 * ```
 * Text message content
 *
 * file content here
 *
 * another file content
 * ```
 *
 * @param content - The message text content
 * @param extras - Optional array of message attachments
 * @param asPlainText - If true, format as plain text without JSON structure
 * @returns Formatted string for clipboard
 */
export function fFormatMessageForClipboard(
  pContent: string,
  pExtras?: DatabaseMessageExtra[],
  pAsPlainText: boolean = false
): string {
  // Filter text-like attachments (TEXT, LEGACY_CONTEXT, MCP_PROMPT, and MCP_RESOURCE types)
  const aTextAttachments =
    pExtras?.filter(
      (
        pExtra
      ): pExtra is
        | DatabaseMessageExtraTextFile
        | DatabaseMessageExtraLegacyContext
        | DatabaseMessageExtraMcpPrompt
        | DatabaseMessageExtraMcpResource =>
        pExtra.type === AttachmentType.TEXT ||
        pExtra.type === AttachmentType.MCP_PROMPT ||
        pExtra.type === AttachmentType.MCP_RESOURCE
    ) ?? [];

  if (aTextAttachments.length === 0) {
    return pContent;
  }

  if (pAsPlainText) {
    const aParts = [pContent];

    for (const cAtt of aTextAttachments) {
      aParts.push(cAtt.content);
    }

    return aParts.join('\n\n');
  }

  const aClipboardAttachments: ClipboardAttachment[] = aTextAttachments.map((pAtt) => {
    if (pAtt.type === AttachmentType.MCP_PROMPT) {
      const cMcpAtt = pAtt as DatabaseMessageExtraMcpPrompt;

      return {
        arguments: cMcpAtt.arguments,
        content: cMcpAtt.content,
        name: cMcpAtt.name,
        promptName: cMcpAtt.promptName,
        serverName: cMcpAtt.serverName,
        type: AttachmentType.MCP_PROMPT
      } as ClipboardMcpPromptAttachment;
    }

    return {
      content: pAtt.content,
      name: pAtt.name,
      type: AttachmentType.TEXT
    } as ClipboardTextAttachment;
  });

  return `${JSON.stringify(pContent)}\n${JSON.stringify(aClipboardAttachments, null, 2)}`;
}

/**
 * Parses clipboard content to extract message and text attachments.
 * Supports both plain text and the special format with attachments.
 *
 * @param clipboardText - Raw text from clipboard
 * @returns Parsed content with message and attachments
 */
export function fParseClipboardContent(pClipboardText: string): ParsedClipboardContent {
  const cDefaultResult: ParsedClipboardContent = {
    mcpPromptAttachments: [],
    message: pClipboardText,
    textAttachments: []
  };

  if (!pClipboardText.startsWith('"')) {
    return cDefaultResult;
  }

  try {
    let vStringEndIndex = -1;
    let vEscaped = false;

    for (let vI = 1; vI < pClipboardText.length; vI++) {
      const cChar = pClipboardText[vI];

      if (vEscaped) {
        vEscaped = false;

        continue;
      }

      if (cChar === '\\') {
        vEscaped = true;

        continue;
      }

      if (cChar === '"') {
        vStringEndIndex = vI;

        break;
      }
    }

    if (vStringEndIndex === -1) {
      return cDefaultResult;
    }

    const cJsonStringPart = pClipboardText.substring(0, vStringEndIndex + 1);
    const cRemainingPart = pClipboardText.substring(vStringEndIndex + 1).trim();
    const message = JSON.parse(cJsonStringPart) as string;

    if (!cRemainingPart || !cRemainingPart.startsWith('[')) {
      return {
        mcpPromptAttachments: [],
        message,
        textAttachments: []
      };
    }

    const attachments = JSON.parse(cRemainingPart) as unknown[];
    const aValidTextAttachments: ClipboardTextAttachment[] = [];
    const aValidMcpPromptAttachments: ClipboardMcpPromptAttachment[] = [];

    for (const cAtt of attachments) {
      if (fIsValidMcpPromptAttachment(cAtt)) {
        aValidMcpPromptAttachments.push({
          arguments: cAtt.arguments,
          content: cAtt.content,
          name: cAtt.name,
          promptName: cAtt.promptName,
          serverName: cAtt.serverName,
          type: AttachmentType.MCP_PROMPT
        });
      } else if (fIsValidTextAttachment(cAtt)) {
        aValidTextAttachments.push({
          content: cAtt.content,
          name: cAtt.name,
          type: AttachmentType.TEXT
        });
      }
    }

    return {
      mcpPromptAttachments: aValidMcpPromptAttachments,
      message,
      textAttachments: aValidTextAttachments
    };
  } catch {
    return cDefaultResult;
  }
}

/**
 * Type guard to validate an MCP prompt attachment object
 * @param obj The object to validate
 * @returns true if the object is a valid MCP prompt attachment
 */
function fIsValidMcpPromptAttachment(pObj: unknown): pObj is {
  type: string;
  name: string;
  serverName: string;
  promptName: string;
  content: string;
  arguments?: Record<string, string>;
} {
  if (typeof pObj !== 'object' || pObj === null) {
    return false;
  }

  const dRecord = pObj as Record<string, unknown>;

  return (
    (dRecord.type === AttachmentType.MCP_PROMPT || dRecord.type === 'MCP_PROMPT') &&
    typeof dRecord.name === 'string' &&
    typeof dRecord.serverName === 'string' &&
    typeof dRecord.promptName === 'string' &&
    typeof dRecord.content === 'string'
  );
}

/**
 * Type guard to validate a text attachment object
 * @param obj The object to validate
 * @returns true if the object is a valid text attachment
 */
function fIsValidTextAttachment(
  pObj: unknown
): pObj is { type: string; name: string; content: string } {
  if (typeof pObj !== 'object' || pObj === null) {
    return false;
  }

  const dRecord = pObj as Record<string, unknown>;

  return (
    (dRecord.type === AttachmentType.TEXT || dRecord.type === 'TEXT') &&
    typeof dRecord.name === 'string' &&
    typeof dRecord.content === 'string'
  );
}

/**
 * Checks if clipboard content contains our special format with attachments
 * @param clipboardText - Raw text from clipboard
 * @returns true if the clipboard content contains our special format with attachments
 */
export function fHasClipboardAttachments(pClipboardText: string): boolean {
  if (!pClipboardText.startsWith('"')) {
    return false;
  }

  const cParsed = fParseClipboardContent(pClipboardText);

  return cParsed.textAttachments.length > 0 || cParsed.mcpPromptAttachments.length > 0;
}
