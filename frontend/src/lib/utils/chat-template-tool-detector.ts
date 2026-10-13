/**
 * Detects whether a model's chat template supports tool calling.
 *
 * There is no server flag for tool support, so we infer it from the chat
 * template. A template that accepts a `tools` array or emits tool-call tokens
 * is treated as tool-capable.
 */

/** Tool-call tokens emitted by the template for assistant tool calls, matched case-insensitively. */
const aToolCallTokens = ['tool_call', 'tool_calls', 'function_call', 'tool_use', '<tool', '<|tool'];
/** Jinja reference to the `tools` array passed in by the caller. */
const cJinjaToolsVar = /\{\{[^{}]*\btools\b[^{}]*\}\}|\{%[^{}]*\btools\b[^{}]*%\}/i;

export function fDetectToolUseSupport(pT: string): boolean {
  if (!pT) return false;

  if (cJinjaToolsVar.test(pT)) return true;

  const cTemplate = pT.toLowerCase();

  return aToolCallTokens.some((pToken) => cTemplate.includes(pToken));
}
