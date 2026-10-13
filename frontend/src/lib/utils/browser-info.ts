/**
 * Browser fallback for the server's `get_info` tool, offered only when the
 * server does not serve one (dgs-server without --agent). It tells the model
 * which OS the browser runs on and that there is no local file or shell access,
 * so it does not plan around tools that are not there.
 *
 * @see server_tool_get_info in tools/server/server-tools.cpp - the served variant
 * @see buildBrowserInfoToolDefinition in constants/browser-info.ts - tool schema sent to the LLM
 */

import { browser } from '$app/environment';
import { cBrowserInfoNote, aBrowserInfoOsUaPatterns, cBrowserInfoOsUnknown } from '$lib/constants';
import type { ToolExecutionResult } from '$lib/types';

export function fDetectOs(pUserAgent: string): string {
  for (const [vPattern, vOs] of aBrowserInfoOsUaPatterns) {
    if (vPattern.test(pUserAgent)) return vOs;
  }

  return cBrowserInfoOsUnknown;
}

/**
 * Result shape mirrors the server tool's JSON so the `get_info` renderer reads
 * `os` the same way, minus `cwd` - there is no working directory to report.
 */
export function fExecuteBrowserInfoTool(): ToolExecutionResult {
  return {
    content: JSON.stringify({
      note: cBrowserInfoNote,
      os: browser ? fDetectOs(navigator.userAgent) : cBrowserInfoOsUnknown
    }),
    isError: false
  };
}
