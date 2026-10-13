import { fTranslate } from '$lib/i18n';
/**
 * ConversationTransferService - Stateless conversation import/export layer
 *
 * Owns the session file format (one JSONL record per line: a SESSION header
 * followed by MESSAGE records), ZIP archiving and browser downloads.
 * DB access and store refreshes stay in conversationsStore.
 */

import { cExportConv, cNewline, aZipMagic } from '$lib/constants';
import {
  FileExtensionText,
  MimeTypeApplication,
  MimeTypeText,
  SessionRecordType
} from '$lib/enums';
import { strFromU8, strToU8, unzipSync, zipSync } from 'fflate';

export class ConversationTransferService {
  /**
   * Triggers a browser download of the provided exported conversation data
   * @param data - The exported conversation payload (a single conversation with its messages)
   * @param filename - Filename; if omitted, a deterministic name is generated
   */
  static mDownloadConversationFile(pData: ExportedConversation, pFilename?: string): void {
    const { conv: vConversation, messages: vMsgs } = pData;

    if (!vConversation) {
      console.error('Invalid data: missing conversation');

      return;
    }

    const cDownloadFilename =
      pFilename ?? ConversationTransferService.mGenerateConversationFilename(vConversation, vMsgs);
    const jsonl = ConversationTransferService.mSerializeSessionToJsonl(pData);
    const cBlob = new Blob([jsonl], { type: MimeTypeText.JSONL });

    ConversationTransferService.mTriggerDownload(cBlob, cDownloadFilename);
  }

  /**
   * Triggers a browser download of multiple conversations as a `.zip`, one
   * `.jsonl` file per conversation.
   * @param data - The conversations to export
   */
  static mDownloadConversationsArchive(pData: ExportedConversation[]): void {
    if (pData.length === 0) {
      console.error('Invalid data: no conversations to export');

      return;
    }

    const sUsedNames = new Set<string>();
    const dFiles: Record<string, Uint8Array> = {};

    for (const session of pData) {
      const cBaseName = ConversationTransferService.mGenerateConversationFilename(
        session.conv,
        session.messages
      );

      // Disambiguate any duplicate filenames within the archive.
      let vEntryName = cBaseName;
      let vSuffix = 1;

      while (sUsedNames.has(vEntryName)) {
        vEntryName = cBaseName.replace(
          new RegExp(`${FileExtensionText.JSONL}$`),
          `_${vSuffix++}${FileExtensionText.JSONL}`
        );
      }
      sUsedNames.add(vEntryName);

      dFiles[vEntryName] = strToU8(ConversationTransferService.mSerializeSessionToJsonl(session));
    }

    const cArchiveName = `${new Date().toISOString().split(cExportConv.ISO_DATE_TIME_SEPARATOR)[0]}_conversations${FileExtensionText.cZip}`;
    const cZipped = zipSync(dFiles);
    const cBlob = new Blob([cZipped], { type: MimeTypeApplication.cZip });

    ConversationTransferService.mTriggerDownload(cBlob, cArchiveName);
  }

  /**
   * Generates a sanitized filename for a conversation export
   * @param conversation - The conversation metadata
   * @param msgs - Optional array of messages belonging to the conversation
   * @returns The generated filename string
   */
  static mGenerateConversationFilename(
    pConversation: { id?: string; name?: string },
    pMsgs?: DatabaseMessage[]
  ): string {
    const cConversationName = (pConversation.name ?? '').trim().toLowerCase();
    const cSanitizedName = cConversationName
      .replace(cExportConv.NON_ALPHANUMERIC_REGEX, cExportConv.NONALNUM_REPLACEMENT)
      .replace(cExportConv.MULTIPLE_UNDERSCORE_REGEX, '_')
      .substring(0, cExportConv.NAME_SUFFIX_MAX_LENGTH);
    // If we have messages, use the timestamp of the newest message
    const cReferenceDate = pMsgs?.length
      ? new Date(Math.max(...pMsgs.map((pM) => pM.timestamp)))
      : new Date();
    const cIso = cReferenceDate.toISOString().slice(0, cExportConv.ISO_TIMESTAMP_SLICE);
    const cFormattedDate = cIso
      .replace(cExportConv.ISO_DATE_TIME_SEPARATOR, cExportConv.ISO_DATE_TIME_SEPARATOR_REPLACEMENT)
      .replaceAll(cExportConv.ISO_TIME_SEPARATOR, cExportConv.ISO_TIME_SEPARATOR_REPLACEMENT);
    const cTrimmedConvId = pConversation.id?.slice(0, cExportConv.ID_TRIM_LENGTH) ?? '';

    return `${cFormattedDate}_conv_${cTrimmedConvId}_${cSanitizedName}${FileExtensionText.JSONL}`;
  }

  /**
   * Parses an import file into conversations, accepting the current JSONL and
   * ZIP formats only. The format comes from the
   * contents, so an import works whatever the file is named.
   * @param file - The user-selected file
   * @returns The parsed conversations with their messages
   */
  static async mParseImportFile(pFile: File): Promise<ExportedConversation[]> {
    const cBytes = new Uint8Array(await pFile.arrayBuffer());

    if (aZipMagic.every((pByte, pIndex) => cBytes[pIndex] === pByte)) {
      const entries = unzipSync(cBytes);
      const aSessions: ExportedConversation[] = [];

      for (const [vEntryName, vEntryBytes] of Object.entries(entries)) {
        if (!vEntryName.toLowerCase().endsWith(FileExtensionText.JSONL)) continue;

        aSessions.push(...ConversationTransferService.mParseSessionsJsonl(strFromU8(vEntryBytes)));
      }

      return aSessions;
    }

    const text = strFromU8(cBytes);

    if (ConversationTransferService.mIsSessionsJsonl(text)) {
      return ConversationTransferService.mParseSessionsJsonl(text);
    }

    throw new Error(fTranslate('dgs_8b3dd836b0d5'));
  }

  /**
   * Parses the JSONL session format produced by {@link serializeSessionToJsonl}.
   * A `SessionRecordType.SESSION` line starts a new session; following
   * `SessionRecordType.MESSAGE` lines are appended to it. Supports multiple
   * sessions in a single file.
   * @param text - The JSONL file contents
   * @returns The parsed conversations with their messages
   */
  static mParseSessionsJsonl(pText: string): ExportedConversation[] {
    const aSessions: ExportedConversation[] = [];

    let vCurrent: ExportedConversation | null = null;

    for (const cLine of pText.split(cNewline)) {
      const cTrimmed = cLine.trim();

      if (!cTrimmed) continue;

      const cRecord = JSON.parse(cTrimmed);

      if (cRecord.type === SessionRecordType.SESSION) {
        // Drop the discriminator and harness marker; the rest is the conversation.
        const cConv = { ...cRecord };

        delete cConv.type;
        delete cConv.harness;
        vCurrent = { conv: cConv as DatabaseConversation, messages: [] };
        aSessions.push(vCurrent);
      } else if (cRecord.type === SessionRecordType.MESSAGE) {
        if (!vCurrent) {
          throw new Error(fTranslate('text_8b7bddc8d9c3'));
        }

        const message = cRecord.message as DatabaseMessage;

        // `toolCalls` is parsed to an array on export; the DB stores it as a string.
        if (message.toolCalls !== undefined && typeof message.toolCalls !== 'string') {
          message.toolCalls = JSON.stringify(message.toolCalls);
        }

        vCurrent.messages.push(message);
      }
      // Ignore unknown record types for forward compatibility.
    }

    return aSessions;
  }

  /**
   * Serializes a session (a conversation with its messages) as JSONL.
   * The first line is the session header (a `SessionRecordType.SESSION` record
   * carrying the conversation properties); each subsequent line is a single message.
   * @param data - The exported conversation payload
   * @returns The JSONL string (one record per line)
   */
  static mSerializeSessionToJsonl(pData: ExportedConversation): string {
    const { conv: vConv, messages: vMessages } = pData;
    const cSessionLine = JSON.stringify({
      harness: cExportConv.HARNESS,
      type: SessionRecordType.SESSION,
      ...vConv
    });
    const cMessageLines = vMessages.map((pMessage: DatabaseMessage) => {
      // `toolCalls` is stored as a JSON string; drop it when empty, otherwise parse it.
      const { toolCalls: vToolCalls, ...vRest } = pMessage;
      const cNormalized = vToolCalls ? { ...vRest, toolCalls: JSON.parse(vToolCalls) } : vRest;

      return JSON.stringify({ message: cNormalized, type: SessionRecordType.MESSAGE });
    });

    return [cSessionLine, ...cMessageLines].join(cNewline);
  }

  /**
   * Reports whether the text is the JSONL session format, whose first non-empty
   * line is a `SessionRecordType.SESSION` record. An unsupported JSON export starts
   * with an array or an object that has no such discriminator.
   * @param text - The file contents
   */
  private static mIsSessionsJsonl(pText: string): boolean {
    const cTrimmed = pText.trimStart();
    const cLineEnd = cTrimmed.indexOf(cNewline);
    const cFirstLine = cLineEnd === -1 ? cTrimmed : cTrimmed.slice(0, cLineEnd);

    try {
      return JSON.parse(cFirstLine).type === SessionRecordType.SESSION;
    } catch {
      // Not a standalone JSON record, so not the JSONL format.
      return false;
    }
  }

  /**
   * Triggers a browser download of a blob under the given filename.
   */
  private static mTriggerDownload(pBlob: Blob, pFilename: string): void {
    const url = URL.createObjectURL(pBlob);
    const cA = document.createElement('a');

    cA.href = url;
    cA.download = pFilename;
    document.body.appendChild(cA);
    cA.click();
    document.body.removeChild(cA);
    URL.revokeObjectURL(url);
  }
}
