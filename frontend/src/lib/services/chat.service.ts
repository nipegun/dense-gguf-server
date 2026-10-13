import { fTranslate } from '$lib/i18n';
import { cQueueStore } from '$lib/stores/queue.svelte';
import { fApplyResponseLanguage } from '$lib/utils/response-language';
/**
 * ChatService - Stateless chat completion and streaming API layer
 *
 * Wraps the /chat/completions and /stream endpoints: request building, SSE
 * parsing, streaming callbacks, resume/probe logic and pre-encode KV-cache
 * warming. No reactive state; consumed by chatStore and its managers.
 */

import { fGetAudioInputFormat } from '../utils/audio-format';
import { fCapImageDataURLSize } from '../utils/cap-img-size';
import {
  cApiChat,
  cApiSlots,
  cApiStream,
  cControlAction,
  HEADERS,
  dReasoningEffortTokens,
  cSettingsKeys,
  cSseDataPrefix,
  cSseDoneMarker,
  cSseLineSeparator,
  cStreamQueryParams,
  cStreamResumeLocalstorageKeyPrefix,
  cStreamVisibilityKickMs
} from '$lib/constants';
import {
  AttachmentLabel,
  AttachmentType,
  ContentPartType,
  MessageRole,
  ReasoningFormat,
  StreamConnectionState
} from '$lib/enums';
import { cModelsStore } from '$lib/stores/models/index.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import type { DatabaseMessageExtraMcpPrompt, DatabaseMessageExtraMcpResource } from '$lib/types';
import type {
  ApiChatCompletionToolCall,
  ApiChatMessageContentPart,
  ApiChatMessageData,
  ApiStreamSession
} from '$lib/types/api';
import { fIsAbortError } from '$lib/utils/abort';
import { ApiError } from '$lib/utils/api-fetch';
import { fGetAuthHeaders, fGetJsonHeaders } from '$lib/utils/api-headers';
import { fFormatAttachmentText } from '$lib/utils/formatters';
import { fStreamIdentity } from '$lib/utils/stream-identity';

interface ResumableStreamState {
  bytesReceived: number;
  updatedAt: number;

  // model frozen at POST time, lets a reload rebuild the exact conv::model identity the
  // server keyed the session under. null when the POST carried no explicit model
  model?: string | null;
}

function fStreamStorageKey(pConversationId: string): string {
  return cStreamResumeLocalstorageKeyPrefix + pConversationId;
}

export class ChatService {
  // Per-chunk localStorage writes are throttled to at most one per
  // conversation per interval (saveStreamStateThrottled). The resume offset
  // only needs to be roughly current: on resume the server retransmits from
  // a line boundary and the client discards its partial line. Guaranteed
  // immediate writes happen at stream start, at resume boundaries and when
  // the page goes hidden or away (pagehide/visibilitychange), so a reload
  // always finds a usable offset.
  private static readonly vStreamStateSaveIntervalMs = 500;

  private static dStreamStateSaveTrackers = new Map<
    string,
    { lastSavedAt: number; model: string | null; pendingBytes: number | null }
  >();

  /**
   * Checks whether all server slots are currently idle (not processing any requests).
   * Queries the /slots endpoint (requires --slots flag on the server).
   * Returns true if all slots are idle, false if any is processing.
   * If the endpoint is unavailable or errors out, returns true (best-effort fallback).
   *
   * @param signal - Optional AbortSignal to cancel the request if needed
   * @param model - Optional model name to check slots for (required in ROUTER mode)
   * @returns {Promise<boolean>} Promise that resolves to true if all slots are idle, false if any is processing
   */
  static async mAreAllSlotsIdle(pModel?: string | null, pSignal?: AbortSignal): Promise<boolean> {
    try {
      const url = pModel ? `${cApiSlots.LIST}?model=${encodeURIComponent(pModel)}` : cApiSlots.LIST;
      const cRes = await fetch(url, { signal: pSignal });

      if (!cRes.ok) return true;

      const aSlots: { is_processing: boolean }[] = await cRes.json();

      return aSlots.every((pS) => !pS.is_processing);
    } catch {
      return true;
    }
  }

  /**
   * Cancels the server-side replay buffer for a conversation, freeing its slot.
   */
  static async mCancelServerStream(pConversationId: string, pModel?: string | null): Promise<void> {
    if (!pConversationId) return;

    try {
      const cId = fStreamIdentity(pConversationId, pModel);

      await fetch(ChatService.mBuildStreamUrl(cId), {
        headers: fGetAuthHeaders(),
        method: 'DELETE'
      });
    } catch (vE) {
      console.warn('cancelServerStream failed:', vE);
    }
  }

  static mClearStreamState(pConversationId: string): void {
    if (!pConversationId) return;

    ChatService.dStreamStateSaveTrackers.delete(pConversationId);

    try {
      localStorage.removeItem(fStreamStorageKey(pConversationId));
    } catch {
      // nothing to do
    }
  }

  /**
   * Converts a database message with attachments to API chat message format.
   * Processes various attachment types (images, text files, PDFs) and formats them
   * as content parts suitable for the chat completion API.
   */
  static async mConvertDbMessageToApiChatMessageData(
    pMessage: DatabaseMessage & { extra?: DatabaseMessageExtra[] }
  ): Promise<ApiChatMessageData> {
    // Handle tool result messages (role: 'tool')
    if (pMessage.role === MessageRole.TOOL && pMessage.toolCallId) {
      return {
        content: pMessage.content,
        role: MessageRole.TOOL,
        tool_call_id: pMessage.toolCallId
      };
    }

    // Parse tool calls for assistant messages
    let vToolCalls: ApiChatCompletionToolCall[] | undefined;

    if (pMessage.toolCalls) {
      try {
        vToolCalls = JSON.parse(pMessage.toolCalls);
      } catch {
        // Ignore parse errors for malformed tool calls
      }
    }

    if (!pMessage.extra || pMessage.extra.length === 0) {
      const cResult: ApiChatMessageData = {
        content: pMessage.content,
        role: pMessage.role as MessageRole
      };

      if (pMessage.reasoningContent) {
        cResult.reasoning_content = pMessage.reasoningContent;
      }

      if (vToolCalls && vToolCalls.length > 0) {
        cResult.tool_calls = vToolCalls;
      }

      return cResult;
    }

    const aContentParts: ApiChatMessageContentPart[] = [];
    const cTextFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraTextFile =>
        pExtra.type === AttachmentType.TEXT
    );

    for (const cTextFile of cTextFiles) {
      aContentParts.push({
        text: fFormatAttachmentText(AttachmentLabel.FILE, cTextFile.name, cTextFile.content),
        type: ContentPartType.TEXT
      });
    }

    const cImageFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraImageFile =>
        pExtra.type === AttachmentType.IMAGE
    );

    for (const image of cImageFiles) {
      const cMaxImageResolution = cSettingsStore.mGetConfig(cSettingsKeys.MAX_IMAGE_RESOLUTION);
      // Caps the resolution and bakes the jpeg exif orientation in one pass,
      // untouched images pass through as is
      const cBase64Url = await fCapImageDataURLSize(image.base64Url, cMaxImageResolution);

      aContentParts.push({
        image_url: { url: cBase64Url },
        type: ContentPartType.cImageUrl
      });
    }

    const cAudioFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraAudioFile =>
        pExtra.type === AttachmentType.AUDIO
    );

    for (const audio of cAudioFiles) {
      aContentParts.push({
        input_audio: {
          data: audio.base64Data,
          format: fGetAudioInputFormat(audio.mimeType)
        },
        type: ContentPartType.cInputAudio
      });
    }

    if (pMessage.content) {
      aContentParts.push({
        text: pMessage.content,
        type: ContentPartType.TEXT
      });
    }

    const cVideoFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraVideoFile =>
        pExtra.type === AttachmentType.VIDEO
    );

    for (const video of cVideoFiles) {
      aContentParts.push({
        input_video: {
          data: video.base64Data,
          format: video.mimeType.includes('mp4')
            ? 'mp4'
            : video.mimeType.includes('ogg')
              ? 'ogg'
              : 'auto'
        },
        type: ContentPartType.cInputVideo
      });
    }

    const cPdfFiles = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraPdfFile =>
        pExtra.type === AttachmentType.PDF
    );

    for (const pdfFile of cPdfFiles) {
      if (pdfFile.processedAsImages && pdfFile.images) {
        for (let vI = 0; vI < pdfFile.images.length; vI++) {
          aContentParts.push({
            image_url: { url: pdfFile.images[vI] },
            type: ContentPartType.cImageUrl
          });
        }
      } else {
        aContentParts.push({
          text: fFormatAttachmentText(AttachmentLabel.PDF_FILE, pdfFile.name, pdfFile.content),
          type: ContentPartType.TEXT
        });
      }
    }

    const cMcpPrompts = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraMcpPrompt =>
        pExtra.type === AttachmentType.MCP_PROMPT
    );

    for (const mcpPrompt of cMcpPrompts) {
      aContentParts.push({
        text: fFormatAttachmentText(
          AttachmentLabel.MCP_PROMPT,
          mcpPrompt.name,
          mcpPrompt.content,
          mcpPrompt.serverName
        ),
        type: ContentPartType.TEXT
      });
    }

    const cMcpResources = pMessage.extra.filter(
      (pExtra: DatabaseMessageExtra): pExtra is DatabaseMessageExtraMcpResource =>
        pExtra.type === AttachmentType.MCP_RESOURCE
    );

    for (const mcpResource of cMcpResources) {
      aContentParts.push({
        text: fFormatAttachmentText(
          AttachmentLabel.MCP_RESOURCE,
          mcpResource.name,
          mcpResource.content,
          mcpResource.serverName
        ),
        type: ContentPartType.TEXT
      });
    }

    const cResult: ApiChatMessageData = {
      content: aContentParts,
      role: pMessage.role as MessageRole
    };

    if (pMessage.reasoningContent) {
      cResult.reasoning_content = pMessage.reasoningContent;
    }

    if (vToolCalls && vToolCalls.length > 0) {
      cResult.tool_calls = vToolCalls;
    }

    return cResult;
  }

  /**
   * Fetch the full replay of a server-side stream from byte 0. Returns the raw Response so the
   * caller can pipe it through the SSE parser like a fresh stream.
   */
  static async mFetchStreamReplay(pStreamId: string): Promise<Response> {
    const cResp = await fetch(ChatService.mBuildStreamUrl(pStreamId, 0), {
      headers: fGetAuthHeaders()
    });

    if (!cResp.ok) {
      throw new ApiError(`Stream replay failed with HTTP ${cResp.status}`, cResp.status);
    }

    return cResp;
  }

  // write a throttled-but-not-yet-persisted offset immediately; used at
  // resume boundaries and on pagehide/visibilitychange so the persisted
  // offset is the freshest one when it matters
  static mFlushStreamState(pConversationId: string): void {
    const cTracker = ChatService.dStreamStateSaveTrackers.get(pConversationId);

    if (!cTracker || cTracker.pendingBytes === null) return;

    const { model: vModel, pendingBytes: vPendingBytes } = cTracker;

    cTracker.lastSavedAt = Date.now();
    cTracker.pendingBytes = null;

    ChatService.mWriteStreamState(pConversationId, vPendingBytes, vModel);
  }

  /**
   * Sends a streaming chat completion request for generating a chat title.
   * Delegates to `sendMessage` for fetch, SSE parsing, and error handling.
   *
   * @param message - The single message to send (a user message containing the title generation prompt)
   * @param model - Optional model name to use (required in ROUTER mode)
   * @param signal - Optional AbortSignal to cancel the request
   * @returns {Promise<string>} The aggregated title text, or empty string if request failed
   * @static
   */
  static async mGenerateTitle(
    pMessage: ApiChatMessageData,
    pModel?: string | null,
    pSignal?: AbortSignal
  ): Promise<string> {
    let vTitleResponse = '';

    try {
      await ChatService.mSendMessage(
        [pMessage],
        {
          custom: { chat_template_kwargs: { enable_thinking: false } },
          model: pModel || undefined,
          onChunk: (pChunk: string) => {
            vTitleResponse += pChunk;
          },
          stream: true
        },
        undefined,
        pSignal
      );
    } catch {
      return '';
    }

    return vTitleResponse;
  }

  static mGetStreamState(pConversationId: string): ResumableStreamState | null {
    if (!pConversationId) return null;

    try {
      const cRaw = localStorage.getItem(fStreamStorageKey(pConversationId));

      if (!cRaw) return null;

      const cParsed = JSON.parse(cRaw) as ResumableStreamState;

      if (!cParsed || typeof cParsed.bytesReceived !== 'number') return null;

      return cParsed;
    } catch {
      return null;
    }
  }

  /**
   * Handles streaming response from the chat completion API.
   */
  static async mHandleStreamResponse(
    pResponse: Response,
    pOnChunk?: (pChunk: string) => void,
    pOnComplete?: (
      pResponse: string,
      pReasoningContent?: string,
      pTimings?: ChatMessageTimings,
      pToolCalls?: string
    ) => void,
    pOnError?: (pError: Error) => void,
    pOnReasoningChunk?: (pChunk: string) => void,
    pOnToolCallChunk?: (pChunk: string) => void,
    pOnModel?: (pModel: string) => void,
    pOnCompletionId?: (pId: string) => void,
    pOnTimings?: (
      pTimings?: ChatMessageTimings,
      pPromptProgress?: ChatMessagePromptProgress
    ) => void,
    pConversationId?: string,
    pAbortSignal?: AbortSignal,
    pOnConnectionState?: (pState: StreamConnectionState) => void,
    pStreamModel?: string | null
  ): Promise<void> {
    let vReader = pResponse.body?.getReader();

    if (!vReader) {
      throw new Error(fTranslate('text_9fc4fcabe3ce'));
    }

    // bytesParsed is the absolute server side buffer offset of the next byte to parse
    // segmentStartOffset is the absolute offset where the current reader started, reset on resume
    // segmentBytesRead is wire bytes read by the current reader
    let vBytesParsed = 0;
    let vSegmentStartOffset = 0;
    let vSegmentBytesRead = 0;
    let vLastByteAt = Date.now();
    // each resume must produce at least one byte to be retried again
    // if a resume returns 200 but yields nothing, we abandon
    // since the session has a bounded size, the total number of retries is bounded by construction
    let vMadeProgress = true;

    const cEncoder = new TextEncoder();

    if (pConversationId) {
      ChatService.mSaveStreamState(pConversationId, 0, pStreamModel);
    }

    pOnConnectionState?.(StreamConnectionState.cStreaming);

    let vDecoder = new TextDecoder();
    let vAggregatedContent = '';
    let vFullReasoningContent = '';
    let aAggregatedToolCalls: ApiChatCompletionToolCall[] = [];
    let vLastTimings: ChatMessageTimings | undefined;
    let vStreamFinished = false;
    let vModelEmitted = false;
    let vIdEmitted = false;
    let vToolCallIndexOffset = 0;
    let vHasOpenToolCallBatch = false;

    const fFinalizeOpenToolCallBatch = () => {
      if (!vHasOpenToolCallBatch) {
        return;
      }

      vToolCallIndexOffset = aAggregatedToolCalls.length;
      vHasOpenToolCallBatch = false;
    };
    const fProcessToolCallDelta = (pToolCalls?: ApiChatCompletionToolCallDelta[]) => {
      if (!pToolCalls || pToolCalls.length === 0) {
        return;
      }

      aAggregatedToolCalls = ChatService.mMergeToolCallDeltas(
        aAggregatedToolCalls,
        pToolCalls,
        vToolCallIndexOffset
      );

      if (aAggregatedToolCalls.length === 0) {
        return;
      }

      vHasOpenToolCallBatch = true;

      const cSerializedToolCalls = JSON.stringify(aAggregatedToolCalls);

      if (import.meta.env.DEV && import.meta.env.VITE_DEBUG) {
        console.log('[ChatService] Aggregated tool calls:', cSerializedToolCalls);
      }

      if (!cSerializedToolCalls) {
        return;
      }

      if (!pAbortSignal?.aborted) {
        pOnToolCallChunk?.(cSerializedToolCalls);
      }
    };
    const fOnVisibilityChange = () => {
      if (typeof document === 'undefined') return;

      if (document.visibilityState === 'hidden') {
        // the tab is going to the background and the OS may throttle or
        // drop the socket shortly; persist the freshest resume offset now
        if (pConversationId) ChatService.mFlushStreamState(pConversationId);

        return;
      }

      if (vStreamFinished) return;

      if (!pConversationId) return;

      // the bytes have been quiet for too long, the OS likely killed the socket
      // kicking the reader unblocks reader.read with done=true so the outer loop can resume
      if (Date.now() - vLastByteAt > cStreamVisibilityKickMs) {
        vReader!.cancel().catch(() => {});
      }
    };
    const fOnPageHide = () => {
      // a reload or navigation is about to happen; make sure the resume
      // offset that getStreamState() will read is not a stale throttled one
      if (pConversationId) ChatService.mFlushStreamState(pConversationId);
    };

    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', fOnVisibilityChange);
      window.addEventListener('pagehide', fOnPageHide);
    }

    try {
      let vChunk = '';

      // outer loop drives the resume cycle, swaps reader on premature end of stream
      while (true) {
        while (true) {
          if (pAbortSignal?.aborted) break;

          let vDone: boolean;
          let vValue: Uint8Array | undefined;

          try {
            const r = await vReader.read();

            vDone = r.done;
            vValue = r.value;
          } catch (vReadErr) {
            // reader.read() rejects with TypeError when the underlying connection drops
            // instead of just resolving with done=true. treat it like done so the outer
            // loop swaps reader via the resume path
            if (fIsAbortError(vReadErr)) {
              throw vReadErr;
            }

            console.warn('reader.read() rejected, treating as premature end:', vReadErr);
            vDone = true;
            vValue = undefined;
          }

          if (vDone) break;

          if (pAbortSignal?.aborted) break;

          if (vValue && vValue.byteLength > 0) {
            vSegmentBytesRead += vValue.byteLength;
            vLastByteAt = Date.now();

            if (!vMadeProgress) {
              vMadeProgress = true;
              pOnConnectionState?.(StreamConnectionState.cStreaming);
            }
          }

          vChunk += vDecoder.decode(vValue, { stream: true });
          const aLines = vChunk.split(cSseLineSeparator);

          vChunk = aLines.pop() || '';

          // the persisted offset must point right after the last fully parsed line,
          // the trailing `chunk` is partial bytes still waiting for a newline
          if (pConversationId) {
            const cTailBytes = cEncoder.encode(vChunk).byteLength;

            vBytesParsed = vSegmentStartOffset + vSegmentBytesRead - cTailBytes;
            ChatService.mSaveStreamStateThrottled(pConversationId, vBytesParsed, pStreamModel);
          }

          for (const cLine of aLines) {
            if (pAbortSignal?.aborted) break;

            if (cLine.startsWith(cSseDataPrefix)) {
              const data = cLine.slice(cSseDataPrefix.length).trim();

              if (data === cSseDoneMarker) {
                vStreamFinished = true;

                continue;
              }

              try {
                const cParsed: ApiChatCompletionStreamChunk = JSON.parse(data);
                if (cParsed.dgs_queue) {
                  cQueueStore.mSet(pConversationId, cParsed.dgs_queue);
                  continue;
                }
                cQueueStore.mSet(pConversationId);
                const cChoice = cParsed.choices?.[0];
                const cContent = cChoice?.delta?.content;
                const cReasoningContent = cChoice?.delta?.reasoning_content;
                const cToolCalls = cChoice?.delta?.tool_calls;
                const cTimings = cParsed.timings;
                const cPromptProgress = cParsed.prompt_progress;
                const cChunkModel = ChatService.mExtractModelName(cParsed);

                if (cChunkModel && !vModelEmitted) {
                  vModelEmitted = true;
                  pOnModel?.(cChunkModel);
                }

                if (cParsed.id && !vIdEmitted) {
                  vIdEmitted = true;
                  pOnCompletionId?.(cParsed.id);
                }

                if (cPromptProgress) {
                  ChatService.mNotifyTimings(undefined, cPromptProgress, pOnTimings);
                }

                if (cTimings) {
                  ChatService.mNotifyTimings(cTimings, cPromptProgress, pOnTimings);
                  vLastTimings = cTimings;
                }

                if (cContent) {
                  fFinalizeOpenToolCallBatch();
                  vAggregatedContent += cContent;

                  if (!pAbortSignal?.aborted) {
                    pOnChunk?.(cContent);
                  }
                }

                if (cReasoningContent) {
                  fFinalizeOpenToolCallBatch();
                  vFullReasoningContent += cReasoningContent;

                  if (!pAbortSignal?.aborted) {
                    pOnReasoningChunk?.(cReasoningContent);
                  }
                }

                fProcessToolCallDelta(cToolCalls);
              } catch (vE) {
                console.error('Error parsing JSON chunk:', vE);
              }
            }
          }

          if (pAbortSignal?.aborted) break;

          if (vStreamFinished) break;
        }

        // inner reader done, decide whether to try a resume
        if (pAbortSignal?.aborted) break;

        if (vStreamFinished) break;

        if (!pConversationId) break;

        if (!vMadeProgress) {
          pOnConnectionState?.(StreamConnectionState.cLost);
          pOnError?.(new Error(fTranslate('text_8c5a46b14100')));

          break;
        }

        pOnConnectionState?.(StreamConnectionState.cResuming);
        vMadeProgress = false;

        // the server resends starting at bytesParsed, discard any partial line we held, it
        // will be retransmitted from a clean line boundary. reuse the frozen model, not the
        // live dropdown
        // resumeStream reads the offset from localStorage, so persist the
        // freshest bytesParsed before asking the server to replay from it
        ChatService.mFlushStreamState(pConversationId);
        const cResumeResp = await ChatService.mResumeStream(
          pConversationId,
          pAbortSignal,
          pStreamModel
        ).catch(() => null);

        // an abort landing during the resume request is intentional, not a lost connection
        if (pAbortSignal?.aborted) break;

        if (!cResumeResp || cResumeResp.status !== 200) {
          pOnConnectionState?.(StreamConnectionState.cLost);
          pOnError?.(new Error(fTranslate('text_67a374e326a1')));

          break;
        }

        const cNewReader = cResumeResp.body?.getReader();

        if (!cNewReader) break;

        try {
          vReader.releaseLock();
        } catch {
          /* ignore */
        }
        vReader = cNewReader;
        vDecoder = new TextDecoder();
        vChunk = '';
        vSegmentStartOffset = vBytesParsed;
        vSegmentBytesRead = 0;
        vLastByteAt = Date.now();
      }

      if (pAbortSignal?.aborted) return;

      if (vStreamFinished) {
        fFinalizeOpenToolCallBatch();

        if (pConversationId) {
          ChatService.mClearStreamState(pConversationId);
        }

        const cFinalToolCalls =
          aAggregatedToolCalls.length > 0 ? JSON.stringify(aAggregatedToolCalls) : undefined;

        pOnComplete?.(
          vAggregatedContent,
          vFullReasoningContent || undefined,
          vLastTimings,
          cFinalToolCalls
        );
      }
    } catch (error) {
      const cErr = error instanceof Error ? error : new Error(fTranslate('text_38f63d7c9f4d'));

      pOnError?.(cErr);

      throw cErr;
    } finally {
      cQueueStore.mSet(pConversationId);
      if (typeof document !== 'undefined') {
        document.removeEventListener('visibilitychange', fOnVisibilityChange);
        window.removeEventListener('pagehide', fOnPageHide);
      }

      try {
        vReader.releaseLock();
      } catch {
        /* ignore */
      }
    }
  }

  /**
   * Look up server-side stream sessions for the given conversation ids. Ids carry the frozen
   * conv::model identity when a model was bound at POST time.
   */
  static async mLookupStreamSessions(pConversationIds: string[]): Promise<ApiStreamSession[]> {
    const cResp = await fetch(cApiStream.LOOKUP, {
      body: JSON.stringify({ conversation_ids: pConversationIds }),
      headers: fGetJsonHeaders(),
      method: 'POST'
    });

    if (!cResp.ok) {
      throw new ApiError(`Stream lookup failed with HTTP ${cResp.status}`, cResp.status);
    }

    const cBody = (await cResp.json()) as unknown;

    if (!Array.isArray(cBody)) {
      throw new Error(fTranslate('dgs_eda45eb7f1ec'));
    }

    return cBody as ApiStreamSession[];
  }

  /**
   * Normalizes an array of messages (database or already-API-shaped) into
   * API chat message data, converting DB messages and dropping empty system
   * messages. Shared by sendMessage, preEncode and the agentic flow.
   */
  static async mNormalizeMessagesForApi(
    pMessages: ApiChatMessageData[] | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[]
  ): Promise<ApiChatMessageData[]> {
    return (
      await Promise.all(
        pMessages.map((pMsg) => {
          if ('id' in pMsg && 'convId' in pMsg && 'timestamp' in pMsg) {
            return ChatService.mConvertDbMessageToApiChatMessageData(
              pMsg as DatabaseMessage & { extra?: DatabaseMessageExtra[] }
            );
          }

          return pMsg as ApiChatMessageData;
        })
      )
    ).filter((pMsg: { role: ChatRole; content: string | ApiChatMessageContentPart[] }) => {
      // Filter out empty system messages
      if (pMsg.role === MessageRole.cSystem) {
        const cContent = typeof pMsg.content === 'string' ? pMsg.content : '';

        return cContent.trim().length > 0;
      }

      return true;
    });
  }

  /**
   * Fire-and-forget request to pre-encode the conversation in the server's KV cache.
   * Re-submits the full conversation with n_predict=0 so the server processes the prompt
   * without generating tokens, warming the cache for the next turn.
   */
  static async mPreEncode(
    pMessages: ApiChatMessageData[] | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[],
    pModel?: string | null,
    pExcludeReasoning?: boolean,
    pSignal?: AbortSignal
  ): Promise<void> {
    const aNormalizedMessages: ApiChatMessageData[] =
      await ChatService.mNormalizeMessagesForApi(pMessages);
    const dRequestBody: Record<string, unknown> = {
      dgs_client: 'webui',
      messages: fApplyResponseLanguage(
        aNormalizedMessages,
        Boolean(cSettingsStore.vConfig.respondInInterfaceLanguage)
      ).map((pMsg: ApiChatMessageData) => {
        const dMapped: Record<string, unknown> = {
          content: pMsg.content,
          role: pMsg.role,
          tool_call_id: pMsg.tool_call_id,
          tool_calls: pMsg.tool_calls
        };

        if (!pExcludeReasoning && pMsg.reasoning_content) {
          dMapped.reasoning_content = pMsg.reasoning_content;
        }

        return dMapped;
      }),
      n_predict: 0,
      stream: false
    };

    if (pModel) {
      dRequestBody.model = pModel;
    }

    try {
      await fetch(cApiChat.COMPLETIONS, {
        body: JSON.stringify(dRequestBody),
        headers: fGetJsonHeaders(),
        method: 'POST',
        signal: pSignal
      });
    } catch (error) {
      if (!fIsAbortError(error)) {
        console.warn('[ChatService] Pre-encode request failed:', error);
      }
    }
  }

  // probe the resume route status without consuming the stream: the SSE route has no HEAD,
  // so issue the GET and abort it right after the status line. 0 on network error
  static async mProbeResumeStatus(pStreamId: string): Promise<number> {
    if (!pStreamId) return 0;

    const cAc = new AbortController();

    try {
      const cResp = await fetch(ChatService.mBuildStreamUrl(pStreamId, 0), {
        headers: fGetAuthHeaders(),
        signal: cAc.signal
      });

      cAc.abort();

      return cResp.status;
    } catch {
      return 0;
    }
  }

  static async mResumeStream(
    pConversationId: string,
    pSignal?: AbortSignal,
    pModel?: string | null
  ): Promise<Response | null> {
    if (!pConversationId) return null;

    const cState = ChatService.mGetStreamState(pConversationId);
    const cFrom = cState?.bytesReceived ?? 0;
    const cId = fStreamIdentity(pConversationId, pModel);
    const url = ChatService.mBuildStreamUrl(cId, cFrom);

    return await fetch(url, { headers: fGetAuthHeaders(), method: 'GET', signal: pSignal });
  }

  /**
   * Rebuild the stream identity for a resume. The model persisted at POST time wins, including a
   * stored null which means the POST carried no explicit model so the identity stays the bare conv
   * id. Only fall back to the caller supplied current model when nothing was persisted.
   */
  static mResumeStreamIdentity(
    pConversationId: string,
    pState: ResumableStreamState | null,
    pFallbackModel: string | null
  ): string {
    const model = pState && pState.model !== undefined ? pState.model : pFallbackModel;

    return fStreamIdentity(pConversationId, model);
  }

  // persist the running byte count and the frozen model for a conversation, a later visit
  // resumes the SSE replay at the right offset under the same conv::model
  // identity. Writes immediately; the per-chunk read loop uses the throttled
  // variant instead.
  static mSaveStreamState(
    pConversationId: string,
    pBytesReceived: number,
    pModel?: string | null
  ): void {
    if (!pConversationId) return;

    ChatService.mWriteStreamState(pConversationId, pBytesReceived, pModel);
    // record the write so a throttled save landing inside the interval
    // holds its value pending instead of re-writing
    ChatService.dStreamStateSaveTrackers.set(pConversationId, {
      lastSavedAt: Date.now(),
      model: pModel ?? null,
      pendingBytes: null
    });
  }

  // throttled variant for the per-chunk read loop: writes at most once per
  // conversation per STREAM_STATE_SAVE_INTERVAL_MS, holding the latest value
  // pending until the interval elapses or flushStreamState() forces it out
  static mSaveStreamStateThrottled(
    pConversationId: string,
    pBytesReceived: number,
    pModel?: string | null
  ): void {
    if (!pConversationId) return;

    const cTracker = ChatService.dStreamStateSaveTrackers.get(pConversationId) ?? {
      lastSavedAt: 0,
      model: null,
      pendingBytes: null
    };

    cTracker.model = pModel ?? null;

    if (Date.now() - cTracker.lastSavedAt >= ChatService.vStreamStateSaveIntervalMs) {
      cTracker.lastSavedAt = Date.now();
      cTracker.pendingBytes = null;
      ChatService.mWriteStreamState(pConversationId, pBytesReceived, pModel);
    } else {
      cTracker.pendingBytes = pBytesReceived;
    }

    ChatService.dStreamStateSaveTrackers.set(pConversationId, cTracker);
  }

  /**
   * Pick the running session to splice into when discoverActiveStream lists candidates for a
   * conversation. Finalized sessions are not candidates: their final content was already written
   * to the DB by the original onComplete handler, so attaching to them would replay a buffer that
   * may not match what the DB holds. A continue session's buffer holds only the appended deltas,
   * not the pre continue prefix, so replaying it as a fresh generation would erase the original.
   *
   * Among running sessions we tie break on the most recent started_at, which covers the case of
   * multiple inferences left running on the same conversation.
   */
  static mSelectActiveStream(
    pSessions: ApiStreamSession[] | null | undefined
  ): ApiStreamSession | null {
    if (!Array.isArray(pSessions) || pSessions.length === 0) {
      return null;
    }

    const aRunning = pSessions.filter((pS) => !pS.is_done);

    if (aRunning.length === 0) {
      return null;
    }

    return aRunning.reduce((pBest, pCur) => (pCur.started_at > pBest.started_at ? pCur : pBest));
  }

  /**
   * Sends a chat completion request to the dgs-server.
   * Supports both streaming and non-streaming responses with comprehensive parameter configuration.
   * Automatically converts database messages with attachments to the appropriate API format.
   *
   * @param messages - Array of chat messages to send to the API (supports both ApiChatMessageData and DatabaseMessage with attachments)
   * @param options - Configuration options for the chat completion request. See `SettingsChatServiceOptions` type for details.
   * @returns {Promise<string | void>} that resolves to the complete response string (non-streaming) or void (streaming)
   * @throws {Error} if the request fails or is aborted
   */
  static async mSendMessage(
    pMessages: ApiChatMessageData[] | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[],
    pOptions: SettingsChatServiceOptions = {},
    pConversationId?: string,
    pSignal?: AbortSignal
  ): Promise<string | void> {
    const {
      backend_sampling: vBackendSampling,
      continueFinalMessage: vContinueFinalMessage,
      custom: vCustom,
      // Config options
      disableReasoningParsing: vDisableReasoningParsing,
      dry_allowed_length: vDryAllowedLength,
      dry_base: vDryBase,
      dry_multiplier: vDryMultiplier,
      dry_penalty_last_n: vDryPenaltyLastN,
      dynatemp_exponent: vDynatempExponent,
      // Sampling parameters
      dynatemp_range: vDynatempRange,
      enableThinking: vEnableThinking,
      excludeReasoningFromContext: vExcludeReasoningFromContext,
      frequency_penalty: vFrequencyPenalty,
      max_tokens: vMaxTokens,
      min_p: vMinP,
      onChunk: vOnChunk,
      onComplete: vOnComplete,
      onCompletionId: vOnCompletionId,
      onConnectionState: vOnConnectionState,
      onError: vOnError,
      onModel: vOnModel,
      onReasoningChunk: vOnReasoningChunk,
      onTimings: vOnTimings,
      onToolCallChunk: vOnToolCallChunk,
      presence_penalty: vPresencePenalty,
      reasoningEffort: vReasoningEffort,
      // Penalty parameters
      repeat_last_n: vRepeatLastN,
      repeat_penalty: vRepeatPenalty,
      // Other parameters
      samplers: vSamplers,
      stream: vStream,
      // Generation parameters
      temperature: vTemperature,
      timings_per_token: vTimingsPerToken,
      // Tools for function calling
      tools,
      top_k: vTopK,
      top_p: vTopP,
      typ_p: vTypP,
      xtc_probability: vXtcProbability,
      xtc_threshold: vXtcThreshold
    } = pOptions;
    const aNormalizedMessages: ApiChatMessageData[] =
      await ChatService.mNormalizeMessagesForApi(pMessages);

    // Filter out image attachments if the model doesn't support vision
    if (pOptions.model && !cModelsStore.props.mModelSupportsVision(pOptions.model)) {
      aNormalizedMessages.forEach((pMsg) => {
        if (Array.isArray(pMsg.content)) {
          pMsg.content = pMsg.content.filter((pPart: ApiChatMessageContentPart) => {
            if (pPart.type === ContentPartType.cImageUrl) {
              console.info(
                `[ChatService] Skipping image attachment in message history (model "${pOptions.model}" does not support vision)`
              );

              return false;
            }

            return true;
          });

          // If only text remains and it's a single part, simplify to string
          if (
            pMsg.content.length === 1 &&
            pMsg.content[0].type === ContentPartType.TEXT &&
            typeof pMsg.content[0].text === 'string'
          ) {
            pMsg.content = pMsg.content[0].text;
          }
        }
      });
    }

    const cRequestBody: ApiChatCompletionRequest = {
      dgs_client: 'webui',
      messages: fApplyResponseLanguage(
        aNormalizedMessages,
        Boolean(cSettingsStore.vConfig.respondInInterfaceLanguage)
      ).map((pMsg: ApiChatMessageData) => {
        const cMapped: ApiChatCompletionRequest['messages'][0] = {
          content: pMsg.content,
          role: pMsg.role,
          tool_call_id: pMsg.tool_call_id,
          tool_calls: pMsg.tool_calls
        };

        // Include reasoning_content from the dedicated field
        if (!vExcludeReasoningFromContext && pMsg.reasoning_content) {
          cMapped.reasoning_content = pMsg.reasoning_content;
        }

        return cMapped;
      }),
      return_progress: vStream ? true : undefined,
      sse_ping_interval: vStream ? 1 : undefined,
      stream: vStream,
      tools: tools && tools.length > 0 ? tools : undefined
    };

    // Include model in request if provided (required in ROUTER mode)
    if (pOptions.model) {
      cRequestBody.model = pOptions.model;
    }

    cRequestBody.reasoning_format = vDisableReasoningParsing
      ? ReasoningFormat.cNone
      : ReasoningFormat.cAuto;

    const cReasoningBudgetTokens =
      vEnableThinking && vReasoningEffort ? (dReasoningEffortTokens[vReasoningEffort] ?? -1) : -1;

    // an explicit user choice injects the kwarg, otherwise it is omitted so
    // the server default applies (--reasoning flag or chat template)
    if (vEnableThinking !== undefined) {
      cRequestBody.chat_template_kwargs = {
        ...(cRequestBody.chat_template_kwargs ?? {}),
        enable_thinking: vEnableThinking
      };
    }

    if (cReasoningBudgetTokens >= 0) {
      cRequestBody.thinking_budget_tokens = cReasoningBudgetTokens;
    }

    // arms the budget sampler so reasoning can be ended at runtime via the control endpoint
    cRequestBody.reasoning_control = true;

    if (vContinueFinalMessage) {
      cRequestBody.continue_final_message = true;
      cRequestBody.add_generation_prompt = false;
    }

    if (vTemperature !== undefined) cRequestBody.temperature = vTemperature;

    if (vMaxTokens !== undefined) {
      // Set max_tokens to -1 (infinite) when explicitly configured as 0 or null
      cRequestBody.max_tokens = vMaxTokens !== null && vMaxTokens !== 0 ? vMaxTokens : -1;
    }

    if (vDynatempRange !== undefined) cRequestBody.dynatemp_range = vDynatempRange;

    if (vDynatempExponent !== undefined) cRequestBody.dynatemp_exponent = vDynatempExponent;

    if (vTopK !== undefined) cRequestBody.top_k = vTopK;

    if (vTopP !== undefined) cRequestBody.top_p = vTopP;

    if (vMinP !== undefined) cRequestBody.min_p = vMinP;

    if (vXtcProbability !== undefined) cRequestBody.xtc_probability = vXtcProbability;

    if (vXtcThreshold !== undefined) cRequestBody.xtc_threshold = vXtcThreshold;

    if (vTypP !== undefined) cRequestBody.typ_p = vTypP;

    if (vRepeatLastN !== undefined) cRequestBody.repeat_last_n = vRepeatLastN;

    if (vRepeatPenalty !== undefined) cRequestBody.repeat_penalty = vRepeatPenalty;

    if (vPresencePenalty !== undefined) cRequestBody.presence_penalty = vPresencePenalty;

    if (vFrequencyPenalty !== undefined) cRequestBody.frequency_penalty = vFrequencyPenalty;

    if (vDryMultiplier !== undefined) cRequestBody.dry_multiplier = vDryMultiplier;

    if (vDryBase !== undefined) cRequestBody.dry_base = vDryBase;

    if (vDryAllowedLength !== undefined) cRequestBody.dry_allowed_length = vDryAllowedLength;

    if (vDryPenaltyLastN !== undefined) cRequestBody.dry_penalty_last_n = vDryPenaltyLastN;

    if (vSamplers !== undefined) {
      cRequestBody.samplers =
        typeof vSamplers === 'string'
          ? vSamplers.split(';').filter((pS: string) => pS.trim())
          : vSamplers;
    }

    if (vBackendSampling !== undefined) cRequestBody.backend_sampling = vBackendSampling;

    if (vTimingsPerToken !== undefined) cRequestBody.timings_per_token = vTimingsPerToken;

    if (vCustom) {
      try {
        const cCustomParams = typeof vCustom === 'string' ? JSON.parse(vCustom) : vCustom;

        Object.assign(cRequestBody, cCustomParams);
      } catch (error) {
        console.warn('Failed to parse custom parameters:', error);
      }
    }

    try {
      const dHeaders: Record<string, string> = { ...fGetJsonHeaders() };

      // tag streaming requests with the conversation id, this single header is the opt in for the
      // server side replay buffer and powers discoverActiveStream on tab reopen. with an explicit
      // model the ::model suffix keeps the per model session distinct
      if (vStream && pConversationId) {
        dHeaders[HEADERS.X_CONVERSATION_ID_HEADER] = fStreamIdentity(
          pConversationId,
          pOptions.model
        );
        // persist the pending stream before the fetch: a reload during the model load or
        // the prompt processing must still find its way back to the session once it exists
        ChatService.mSaveStreamState(pConversationId, 0, pOptions.model ?? null);
      }

      const cResponse = await fetch(cApiChat.COMPLETIONS, {
        body: JSON.stringify(cRequestBody),
        headers: dHeaders,
        method: 'POST',
        signal: pSignal
      });

      if (!cResponse.ok) {
        // a rejected request (including one cancelled by a stop during the model load)
        // leaves nothing to resume
        if (pConversationId) {
          ChatService.mClearStreamState(pConversationId);
        }

        const error = await ChatService.mParseErrorResponse(cResponse);

        if (vOnError) {
          vOnError(error);
        }

        throw error;
      }

      if (vStream) {
        await ChatService.mHandleStreamResponse(
          cResponse,
          vOnChunk,
          vOnComplete,
          vOnError,
          vOnReasoningChunk,
          vOnToolCallChunk,
          vOnModel,
          vOnCompletionId,
          vOnTimings,
          pConversationId,
          pSignal,
          vOnConnectionState,
          pOptions.model
        );

        return;
      } else {
        return ChatService.mHandleNonStreamResponse(
          cResponse,
          vOnComplete,
          vOnError,
          vOnToolCallChunk,
          vOnModel
        );
      }
    } catch (error) {
      if (fIsAbortError(error)) {
        console.log('Chat completion request was aborted');

        return;
      }

      let vUserFriendlyError: Error;

      if (error instanceof Error) {
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
          vUserFriendlyError = new Error(fTranslate('text_557aabe13eac'));
          vUserFriendlyError.name = 'NetworkError';
        } else if (error.message.includes('ECONNREFUSED')) {
          vUserFriendlyError = new Error(fTranslate('text_448ea20860db'));
          vUserFriendlyError.name = 'NetworkError';
        } else if (error.message.includes('ETIMEDOUT')) {
          vUserFriendlyError = new Error(fTranslate('text_d14a410ba4ad'));
          vUserFriendlyError.name = 'TimeoutError';
        } else {
          vUserFriendlyError = error;
        }
      } else {
        vUserFriendlyError = new Error(fTranslate('text_23a991241806'));
      }

      console.error('Error in sendMessage:', error);

      if (vOnError) {
        vOnError(vUserFriendlyError);
      }

      throw vUserFriendlyError;
    }
  }

  /**
   * Ends the current reasoning block of a running completion, targeted by its
   * chat completion id (streamed back as `id`). Matching the completion rather
   * than a slot index avoids a TOCTOU: a finished completion simply matches
   * nothing server side. The model is carried so the router forwards to the
   * right child, single model ignores it. Returns true on success.
   */
  static async mStopReasoning(pCompletionId: string, pModel?: string | null): Promise<boolean> {
    if (!pCompletionId) {
      console.error(
        'stopReasoning: no completion id for the active message, cannot target the running completion'
      );

      return false;
    }

    const dBody: Record<string, unknown> = {
      action: cControlAction.END_REASONING,
      id: pCompletionId
    };

    if (pModel) dBody.model = pModel;

    try {
      const cRes = await fetch(cApiChat.CONTROL, {
        body: JSON.stringify(dBody),
        headers: fGetJsonHeaders(),
        method: 'POST'
      });
      const data = await cRes.json().catch(() => null);

      if (!cRes.ok || data?.success !== true) {
        console.error('stopReasoning: control request failed', {
          completionId: pCompletionId,
          response: data,
          status: cRes.status
        });

        return false;
      }

      return true;
    } catch (error) {
      console.error('stopReasoning: control request threw', { completionId: pCompletionId, error });

      return false;
    }
  }

  // build the replay route url for a stream identity, from is the resume byte offset, omitted
  // for the cancel route
  private static mBuildStreamUrl(pStreamId: string, pFrom?: number): string {
    const cQuery = `${cStreamQueryParams.CONV_ID}=${encodeURIComponent(pStreamId)}`;
    const cOffset = pFrom === undefined ? '' : `&${cStreamQueryParams.FROM}=${pFrom}`;

    return `${cApiStream.BASE}?${cQuery}${cOffset}`;
  }

  /**
   * Extracts model name from Chat Completions API response data.
   * Handles various response formats including streaming chunks and final responses.
   *
   * WORKAROUND: In single model mode, dgs-server returns a default/incorrect model name
   * in the response. We override it with the actual model name from serverStore.
   *
   * @param data - Raw response data from the Chat Completions API
   * @returns Model name string if found, undefined otherwise
   * @private
   */
  private static mExtractModelName(pData: unknown): string | undefined {
    const fAsRecord = (pValue: unknown): Record<string, unknown> | undefined => {
      return typeof pValue === 'object' && pValue !== null
        ? (pValue as Record<string, unknown>)
        : undefined;
    };
    const fGetTrimmedString = (pValue: unknown): string | undefined => {
      return typeof pValue === 'string' && pValue.trim() ? pValue.trim() : undefined;
    };
    const dRoot = fAsRecord(pData);

    if (!dRoot) return undefined;

    // 1) root (some implementations provide `model` at the top level)
    const cRootModel = fGetTrimmedString(dRoot.model);

    if (cRootModel) {
      return cRootModel;
    }

    // 2) streaming choice (delta) or final response (message)
    const dFirstChoice = Array.isArray(dRoot.choices) ? fAsRecord(dRoot.choices[0]) : undefined;

    if (!dFirstChoice) {
      return undefined;
    }

    // priority: delta.model (first chunk) else message.model (final response)
    const cDeltaModel = fGetTrimmedString(fAsRecord(dFirstChoice.delta)?.model);

    if (cDeltaModel) {
      return cDeltaModel;
    }

    const cMessageModel = fGetTrimmedString(fAsRecord(dFirstChoice.message)?.model);

    if (cMessageModel) {
      return cMessageModel;
    }

    // avoid guessing from non-standard locations (metadata, etc.)
    return undefined;
  }

  /**
   * Handles non-streaming response from the chat completion API.
   * Parses the JSON response and extracts the generated content.
   *
   * @param response - The fetch Response object containing the JSON data
   * @param onComplete - Optional callback invoked when response is successfully parsed
   * @param onError - Optional callback invoked if an error occurs while parsing
   * @returns {Promise<string>} Promise that resolves to the generated content string
   * @throws {Error} if the response cannot be parsed or is malformed
   */
  private static async mHandleNonStreamResponse(
    pResponse: Response,
    pOnComplete?: (
      pResponse: string,
      pReasoningContent?: string,
      pTimings?: ChatMessageTimings,
      pToolCalls?: string
    ) => void,
    pOnError?: (pError: Error) => void,
    pOnToolCallChunk?: (pChunk: string) => void,
    pOnModel?: (pModel: string) => void
  ): Promise<string> {
    try {
      const cResponseText = await pResponse.text();

      if (!cResponseText.trim()) {
        const cNoResponseError = new Error(fTranslate('text_c2d62ca92c00'));

        throw cNoResponseError;
      }

      const data: ApiChatCompletionResponse = JSON.parse(cResponseText);
      const cResponseModel = ChatService.mExtractModelName(data);

      if (cResponseModel) {
        pOnModel?.(cResponseModel);
      }

      const cContent = data.choices[0]?.message?.content || '';
      const cReasoningContent = data.choices[0]?.message?.reasoning_content;
      const cToolCalls = data.choices[0]?.message?.tool_calls;

      let vSerializedToolCalls: string | undefined;

      if (cToolCalls && cToolCalls.length > 0) {
        const aMergedToolCalls = ChatService.mMergeToolCallDeltas([], cToolCalls);

        if (aMergedToolCalls.length > 0) {
          vSerializedToolCalls = JSON.stringify(aMergedToolCalls);

          if (vSerializedToolCalls) {
            pOnToolCallChunk?.(vSerializedToolCalls);
          }
        }
      }

      if (!cContent.trim() && !vSerializedToolCalls) {
        const cNoResponseError = new Error(fTranslate('text_c2d62ca92c00'));

        throw cNoResponseError;
      }

      pOnComplete?.(cContent, cReasoningContent, undefined, vSerializedToolCalls);

      return cContent;
    } catch (error) {
      const cErr = error instanceof Error ? error : new Error(fTranslate('text_65837ce7a4be'));

      pOnError?.(cErr);

      throw cErr;
    }
  }

  /**
   * Merges tool call deltas into an existing array of tool calls.
   * Handles both existing and new tool calls, updating existing ones and adding new ones.
   *
   * @param existing - The existing array of tool calls to merge into
   * @param deltas - The array of tool call deltas to merge
   * @param indexOffset - Optional offset to apply to the index of new tool calls
   * @returns {ApiChatCompletionToolCall[]} The merged array of tool calls
   */
  private static mMergeToolCallDeltas(
    pExisting: ApiChatCompletionToolCall[],
    pDeltas: ApiChatCompletionToolCallDelta[],
    pIndexOffset = 0
  ): ApiChatCompletionToolCall[] {
    const aResult = pExisting.map((pCall) => ({
      ...pCall,
      function: pCall.function ? { ...pCall.function } : undefined
    }));

    for (const cDelta of pDeltas) {
      const cIndex =
        typeof cDelta.index === 'number' && cDelta.index >= 0
          ? cDelta.index + pIndexOffset
          : aResult.length;

      while (aResult.length <= cIndex) {
        aResult.push({ function: undefined });
      }

      const cTarget = aResult[cIndex]!;

      if (cDelta.id) {
        cTarget.id = cDelta.id;
      }

      if (cDelta.type) {
        cTarget.type = cDelta.type;
      }

      if (cDelta.function) {
        const cFn = cTarget.function ? { ...cTarget.function } : {};

        if (cDelta.function.name) {
          cFn.name = cDelta.function.name;
        }

        if (cDelta.function.arguments) {
          cFn.arguments = (cFn.arguments ?? '') + cDelta.function.arguments;
        }

        cTarget.function = cFn;
      }
    }

    return aResult;
  }

  /**
   * Calls the onTimings callback with timing data from streaming response.
   *
   * @param timings - Timing information from the Chat Completions API response
   * @param promptProgress - Prompt processing progress data
   * @param onTimingsCallback - Callback function to invoke with timing data
   * @private
   */
  private static mNotifyTimings(
    pTimings: ChatMessageTimings | undefined,
    pPromptProgress: ChatMessagePromptProgress | undefined,
    pOnTimingsCallback:
      | ((pTimings?: ChatMessageTimings, pPromptProgress?: ChatMessagePromptProgress) => void)
      | undefined
  ): void {
    if (!pOnTimingsCallback || (!pTimings && !pPromptProgress)) return;

    pOnTimingsCallback(pTimings, pPromptProgress);
  }

  /**
   * Parses error response and creates appropriate error with context information
   * @param response - HTTP response object
   * @returns Promise<Error> - Parsed error with context info if available
   */
  private static async mParseErrorResponse(
    pResponse: Response
  ): Promise<Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }> {
    try {
      const cErrorText = await pResponse.text();
      const cErrorData: ApiErrorResponse = JSON.parse(cErrorText);
      const message = cErrorData.error?.message || 'Unknown server error';
      const error = new Error(message) as Error & {
        contextInfo?: { n_prompt_tokens: number; n_ctx: number };
      };

      error.name = pResponse.status === 400 ? 'ServerError' : 'HttpError';

      if (
        cErrorData.error &&
        'n_prompt_tokens' in cErrorData.error &&
        'n_ctx' in cErrorData.error
      ) {
        error.contextInfo = {
          n_ctx: cErrorData.error.n_ctx,
          n_prompt_tokens: cErrorData.error.n_prompt_tokens
        };
      }

      return error;
    } catch {
      const cFallback = new Error(
        `Server error (${pResponse.status}): ${pResponse.statusText}`
      ) as Error & {
        contextInfo?: { n_prompt_tokens: number; n_ctx: number };
      };

      cFallback.name = 'HttpError';

      return cFallback;
    }
  }

  // write the resume state straight to localStorage, bypassing the throttle
  private static mWriteStreamState(
    pConversationId: string,
    pBytesReceived: number,
    pModel?: string | null
  ): void {
    try {
      const cState: ResumableStreamState = {
        bytesReceived: pBytesReceived,
        model: pModel ?? null,
        updatedAt: Date.now()
      };

      localStorage.setItem(fStreamStorageKey(pConversationId), JSON.stringify(cState));
    } catch {
      // localStorage may be full or disabled, silently ignore
    }
  }
}
