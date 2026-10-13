/**
 * Browser executor for the `get_datetime` tool. It runs in the browser, so it
 * reports the user's own clock and time zone instead of the server's UTC time -
 * a chat about "tomorrow" means the user's tomorrow, not the host's.
 *
 * @see buildGetDatetimeToolDefinition in constants/get-datetime.ts - tool schema sent to the LLM
 */

import type { ToolExecutionResult } from '$lib/types';

function fPad(pValue: number): string {
  return String(pValue).padStart(2, '0');
}

/** ISO 8601 in local time, e.g. `2026-08-17T14:05:09+02:00` */
function fLocalIsoString(pDate: Date): string {
  // getTimezoneOffset() counts minutes behind UTC, ISO 8601 counts them ahead
  const cOffset = -pDate.getTimezoneOffset();
  const cSign = cOffset < 0 ? '-' : '+';
  const cAbsOffset = Math.abs(cOffset);
  const cDay = `${pDate.getFullYear()}-${fPad(pDate.getMonth() + 1)}-${fPad(pDate.getDate())}`;
  const cTime = `${fPad(pDate.getHours())}:${fPad(pDate.getMinutes())}:${fPad(pDate.getSeconds())}`;

  return `${cDay}T${cTime}${cSign}${fPad(Math.floor(cAbsOffset / 60))}:${fPad(cAbsOffset % 60)}`;
}

/** The `result` field keeps the shape the `get_datetime` renderer already reads. */
export function fExecuteGetDatetimeTool(): ToolExecutionResult {
  const cNow = new Date();

  return {
    content: JSON.stringify({
      result: fLocalIsoString(cNow),
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    }),
    isError: false
  };
}
