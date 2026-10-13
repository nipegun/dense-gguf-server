/**
 * SandboxService - Runs untrusted code in a sandboxed worker
 *
 * Executes model-generated code inside a CSP-restricted, opaque-origin
 * iframe worker with output and timeout limits. No reactive state; consumed
 * by toolsStore for code-execution tools.
 */

import { fBuildSandboxHarness } from './sandbox-harness';
import {
  cNewline,
  cSandboxEmptyOutput,
  cSandboxOutputMaxChars,
  cSandboxTimeoutMsDefault,
  cSandboxTimeoutMsMax,
  cSandboxToolName,
  cSandboxTruncationNotice
} from '$lib/constants';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import type { ToolExecutionResult } from '$lib/types';

/** Cached harnesses keyed by whether nerdamer is included. */
const dHarnessCache: Record<string, string> = {};

/**
 * Build the sandbox harness. When symbolic math is enabled, loads the
 * nerdamer prelude lazily; otherwise builds a plain harness with an empty
 * prelude. Cached per variant so toggling the setting is instant.
 */
async function fGetHarness(): Promise<string> {
  const cEnabled = !!cSettingsStore.vConfig.symbolicMathEnabled;
  const cKey = cEnabled ? 'nerdamer' : 'plain';

  if (!dHarnessCache[cKey]) {
    if (cEnabled) {
      const { default: vNerdamerJs } = await import('virtual:nerdamer');

      dHarnessCache[cKey] = fBuildSandboxHarness(vNerdamerJs);
    } else {
      dHarnessCache[cKey] = fBuildSandboxHarness('');
    }
  }

  return dHarnessCache[cKey];
}

interface SandboxReply {
  logs?: unknown;
  result?: unknown;
  error?: unknown;
}

function fFormatReply(pReply: SandboxReply): ToolExecutionResult {
  const aLines: string[] = [];

  if (Array.isArray(pReply.logs)) {
    for (const cLine of pReply.logs) aLines.push(String(cLine));
  }

  if (pReply.error != null) {
    aLines.push(`Error: ${String(pReply.error)}`);
  } else if (pReply.result != null) {
    aLines.push(`=> ${String(pReply.result)}`);
  }

  let vContent = aLines.join(cNewline);

  if (!vContent) vContent = cSandboxEmptyOutput;

  if (vContent.length > cSandboxOutputMaxChars) {
    vContent = `${vContent.slice(0, cSandboxOutputMaxChars)}${cNewline}${cSandboxTruncationNotice}`;
  }

  return { content: vContent, isError: pReply.error != null };
}

export class SandboxService {
  /**
   * Execute a browser sandbox tool call and return its output.
   * One disposable iframe per execution, removed on completion,
   * timeout or abort. Removing the iframe terminates the worker
   * at the browser level, so runaway code cannot outlive it.
   */
  static async mExecuteTool(
    pToolName: string,
    pParams: Record<string, unknown>,
    pSignal?: AbortSignal
  ): Promise<ToolExecutionResult> {
    if (pToolName !== cSandboxToolName) {
      return { content: `Unknown browser tool: ${pToolName}`, isError: true };
    }

    const cCode = typeof pParams.code === 'string' ? pParams.code : '';

    if (!cCode) {
      return { content: 'Missing required parameter: code', isError: true };
    }

    const cHarness = await fGetHarness();
    const cRequested = Number(pParams.timeout_ms);
    const cTimeoutMs =
      Number.isFinite(cRequested) && cRequested > 0
        ? Math.min(cRequested, cSandboxTimeoutMsMax)
        : cSandboxTimeoutMsDefault;

    return new Promise<ToolExecutionResult>((pResolve, pReject) => {
      const cIframe = document.createElement('iframe');

      cIframe.setAttribute('sandbox', 'allow-scripts');
      cIframe.style.display = 'none';
      cIframe.srcdoc = cHarness;

      let vSettled = false;

      const fCleanup = () => {
        vSettled = true;
        clearTimeout(cTimer);
        window.removeEventListener('message', fOnMessage);
        pSignal?.removeEventListener('abort', fOnAbort);
        cIframe.remove();
      };
      const fFinish = (pResult: ToolExecutionResult) => {
        if (vSettled) return;

        fCleanup();
        pResolve(pResult);
      };
      const fOnAbort = () => {
        if (vSettled) return;

        fCleanup();
        pReject(new DOMException('Sandbox execution aborted', 'AbortError'));
      };
      const fOnMessage = (pEvent: MessageEvent) => {
        if (pEvent.source !== cIframe.contentWindow) return;

        fFinish(fFormatReply((pEvent.data ?? {}) as SandboxReply));
      };
      const cTimer = setTimeout(
        () => fFinish({ content: `Execution timed out after ${cTimeoutMs} ms`, isError: true }),
        cTimeoutMs
      );

      window.addEventListener('message', fOnMessage);
      pSignal?.addEventListener('abort', fOnAbort);
      cIframe.onload = () => cIframe.contentWindow?.postMessage({ code: cCode }, '*');
      document.body.appendChild(cIframe);
    });
  }
}
