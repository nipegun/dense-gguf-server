import {
  cSseDataPrefix,
  cSseDoneMarker,
  cSseLineSeparator,
  cSseRecordSeparator
} from '$lib/constants';

/**
 * Minimal SSE-with-JSON stream iterator.
 *
 * Yields one event per `\n\n`-separated record. Each event payload is the
 * decoded `data:` field after JSON-parsing. A `[DONE]` sentinel terminates
 * the stream early. Malformed records - any record whose `data:` payload
 * fails `JSON.parse` - are skipped silently: usually a transient mid-stream
 * fault that the caller should not have to special-case, and the noise of
 * logging every occurrence on long-running streams outweighs the diagnostic
 * value.
 *
 * Less ambitious than ChatService.handleStreamResponse (no resume, no byte
 * offset tracking) - suitable for one-shot streams like `/tools?stream=true`
 * where the consumer just reads chunks until done.
 */

export interface SseJsonEvent<T = unknown> {
  data: T;
}

/**
 * Splits a raw SSE byte buffer into complete records on the blank-line
 * boundary, returning the leftover partial record separately. Shared by the
 * record-based consumers (parseSseJsonStream, models.service).
 */
export function fSplitSseRecords(pBuffer: string): { records: string[]; rest: string } {
  const aParts = pBuffer.split(cSseRecordSeparator);

  return { records: aParts.slice(0, -1), rest: aParts[aParts.length - 1] ?? '' };
}

/**
 * Extracts the joined `data:` payload from one SSE record (the data lines
 * concatenated with a newline), or an empty string when the record carries
 * no data lines. Used by models.service to parse status envelopes.
 */
export function fExtractSseDataPayload(pRecord: string): string {
  return pRecord
    .split(cSseLineSeparator)
    .filter((pLine) => pLine.startsWith(cSseDataPrefix))
    .map((pLine) => pLine.slice(cSseDataPrefix.length).trim())
    .join(cSseLineSeparator);
}

export async function* fParseSseJsonStream<T = unknown>(
  pResponse: Response,
  pSignal?: AbortSignal
): AsyncGenerator<SseJsonEvent<T>> {
  const cReader = pResponse.body?.getReader();

  if (!cReader) return;

  const cDecoder = new TextDecoder();

  let vBuffer = '';

  try {
    while (true) {
      if (pSignal?.aborted) return;

      const { done: vDone, value: vValue } = await cReader.read();

      if (vDone) break;

      vBuffer += cDecoder.decode(vValue, { stream: true });
      const { records: aRecords, rest: vRest } = fSplitSseRecords(vBuffer);

      vBuffer = vRest;

      for (const cRecord of aRecords) {
        if (!cRecord) continue;

        for (const cLine of cRecord.split(cSseLineSeparator)) {
          if (!cLine.startsWith(cSseDataPrefix)) continue;

          const cPayload = cLine.slice(cSseDataPrefix.length).trim();

          if (cPayload === cSseDoneMarker) return;

          if (!cPayload) continue;

          try {
            yield { data: JSON.parse(cPayload) as T };
          } catch {
            // Skip silently per the function contract above.
          }
        }
      }
    }
  } finally {
    try {
      cReader.releaseLock();
    } catch (error) {
      console.error('[sse] failed to release reader lock:', error);
    }
  }
}
