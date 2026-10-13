/**
 * ToolsService - Stateless server tools API layer
 *
 * Fetches the server's /tools listing and streams tool execution results.
 * No reactive state; consumed by toolsStore.
 */

import { base } from '$app/paths';
import { cApiTools, HEADERS } from '$lib/constants';
import { ToolResponseField } from '$lib/enums';
import type { ServerToolInfo, ToolExecutionResult } from '$lib/types';
import { fApiFetch } from '$lib/utils';
import { fGetJsonHeaders } from '$lib/utils/api-headers';
import { fParseSseJsonStream, type SseJsonEvent } from '$lib/utils/sse';

export class ToolsService {
  /**
   * Execute a server tool on the server.
   *
   * @param cwd - Working directory for the tool call, sent as the
   * x-tool-cwd request header. The server resolves relative paths
   * against it; the model cannot override it.
   */
  static async mExecuteTool(
    pToolName: string,
    pParams: Record<string, unknown>,
    pSignal?: AbortSignal,
    pCwd?: string
  ): Promise<ToolExecutionResult> {
    const dResult = await fApiFetch<Record<string, unknown>>(cApiTools.EXECUTE, {
      body: JSON.stringify({ params: pParams, tool: pToolName }),
      headers: pCwd ? { [HEADERS.X_TOOL_CWD_HEADER]: pCwd } : undefined,
      method: 'POST',
      signal: pSignal
    });

    if (ToolResponseField.ERROR in dResult) {
      return { content: String(dResult[ToolResponseField.ERROR]), isError: true };
    }

    if (ToolResponseField.cPlainText in dResult) {
      return { content: String(dResult[ToolResponseField.cPlainText]), isError: false };
    }

    return { content: JSON.stringify(dResult), isError: false };
  }

  /**
   * Execute a server tool and return the raw JSON response. Unlike
   * executeTool, this preserves structured fields (e.g. file_glob_search's
   * `entries` and `base`) that the flattened ToolExecutionResult drops.
   *
   * @param respType - sent as the x-resp-type request header. Only read_file
   * honors it, with `base64` to get the raw bytes instead of decoded text.
   */
  static async mExecuteToolRaw(
    pToolName: string,
    pParams: Record<string, unknown>,
    pSignal?: AbortSignal,
    pCwd?: string,
    pRespType?: string
  ): Promise<Record<string, unknown>> {
    const dHeaders: Record<string, string> = {};

    if (pCwd) dHeaders[HEADERS.X_TOOL_CWD_HEADER] = pCwd;

    if (pRespType) dHeaders[HEADERS.X_RESP_TYPE_HEADER] = pRespType;

    return fApiFetch<Record<string, unknown>>(cApiTools.EXECUTE, {
      body: JSON.stringify({ params: pParams, tool: pToolName }),
      headers: Object.keys(dHeaders).length > 0 ? dHeaders : undefined,
      method: 'POST',
      signal: pSignal
    });
  }

  /**
   * Fetch the list of server tools from the server.
   *
   * @returns Array of tool definitions in OpenAI-compatible format
   */
  static async mList(): Promise<ServerToolInfo[]> {
    return fApiFetch<ServerToolInfo[]>(cApiTools.LIST);
  }

  /**
   * Stream a server tool's output chunks from the server. The server
   * `POST /tools` endpoint with `{stream: true}` emits `data: {"chunk": "..."}`
   * events followed by a terminal `data: {"done": true}` (optionally with
   * `error`). Yields the chunk string for each partial event.
   *
   * The terminal event's `error` field, if present, is yielded as a final
   * synthetic chunk prefixed with an error marker so the accumulated content
   * already carries the failure context for the caller.
   *
   * Throws synchronously if the server rejects the request (e.g. tool does
   * not support streaming, or 4xx/5xx response). The HTTP fetch goes through
   * a minimal text/event-stream reader since the chat SSE parser in
   * chat.service.ts embeds extra resume logic that is unnecessary here.
   */
  static async *mStreamTool(
    pToolName: string,
    pParams: Record<string, unknown>,
    pSignal?: AbortSignal,
    pCwd?: string
  ): AsyncGenerator<ToolStreamEvent> {
    const dHeaders = fGetJsonHeaders();

    if (pCwd) dHeaders[HEADERS.X_TOOL_CWD_HEADER] = pCwd;

    const cResponse = await fetch(`${base}${cApiTools.EXECUTE}`, {
      body: JSON.stringify({ params: pParams, stream: true, tool: pToolName }),
      headers: dHeaders,
      method: 'POST',
      signal: pSignal
    });

    if (!cResponse.ok || !cResponse.body) {
      const cDetail = await fFormatNonOkResponse(cResponse);

      throw new Error(cDetail);
    }

    const cIterator = fParseSseJsonStream<ToolServerEvent>(cResponse, pSignal);

    while (true) {
      const cNext: IteratorResult<SseJsonEvent<ToolServerEvent>> = await cIterator.next();

      if (cNext.done) return;

      const cEvent = cNext.value.data;

      if (cEvent.chunk !== undefined) {
        yield { chunk: cEvent.chunk, done: false };
      }

      if (cEvent.done) {
        yield { chunk: null, done: true, error: cEvent.error };

        return;
      }
    }
  }
}

/**
 * One event from streaming a tool's output.
 *   - During execution: `chunk` is a non-empty text fragment, `done: false`.
 *   - On terminal event: `done: true`, `error` populated if the call failed,
 *     and `chunk` is null.
 */
export interface ToolStreamEvent {
  chunk: string | null;
  done: boolean;
  error?: string;
}

/** Wire shape of one SSE event from `POST /tools?stream=true`. */
interface ToolServerEvent {
  chunk?: string;
  done?: boolean;
  error?: string;
}

async function fFormatNonOkResponse(pResponse: Response): Promise<string> {
  const cStatus = `${pResponse.status} ${pResponse.statusText}`.trim();

  try {
    const cErrBody = (await pResponse.clone().json()) as { error?: string; message?: string };

    if (cErrBody?.error) return `${cStatus}: ${cErrBody.error}`;

    if (cErrBody?.message) return `${cStatus}: ${cErrBody.message}`;
  } catch (error) {
    console.error('[tools] Non-JSON error response, falling back to raw text:', error);
    try {
      const text = await pResponse.text();

      if (text.trim()) return `${cStatus}: ${text.trim()}`;
    } catch (error) {
      console.error('[tools] Failed to read error response as text:', error);
    }
  }

  return cStatus || `HTTP ${pResponse.status}`;
}
