// Meta parser for `exec_shell_command` tool calls. Surfaces the
// command text from args `command` / `cmd` / `shell_command` aliases.
// The exit-status and error parsing live in their own utilities
// (`parse-exec-shell-status.ts` / `parse-exec-shell-error.ts`) - this
// file only deals with what's strictly about *calling* the tool, since
// the error / exit status elide from call-section to result-section.

import { fParseToolArgs } from './_shared';
import { BuiltInTool } from '$lib/enums';
import type { AgenticSection } from '$lib/types';

export type ExecShellCommandMeta = {
  command: string;
};

export function fParseExecShellCommandMeta(pSection: AgenticSection): ExecShellCommandMeta | null {
  const dArgs = fParseToolArgs(BuiltInTool.cServerExecShellCommand, pSection);

  if (!dArgs) return null;

  const cCommandRaw = dArgs.command ?? dArgs.cmd ?? dArgs.shell_command;

  if (typeof cCommandRaw !== 'string' || !cCommandRaw) return null;

  return { command: cCommandRaw };
}
