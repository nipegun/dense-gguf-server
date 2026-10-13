import { fTranslate } from '$lib/i18n';
import { cCliFlags } from './cli-flags.constants';
import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

// get_info is served by the server, but the browser falls back to this
// implementation when the server does not provide it - same wire name.
export const cBrowserInfoToolName = BuiltInTool.cServerGetInfo;

/** UA token to OS name, first match wins - Android and iOS UAs also carry the Linux / Mac OS X tokens */
export const aBrowserInfoOsUaPatterns: readonly [RegExp, string][] = [
  [/Windows NT/, 'Windows'],
  [/Android/, 'Android'],
  [/iPhone|iPad|iPod/, 'iOS'],
  [/CrOS/, 'ChromeOS'],
  [/Mac OS X/, 'macOS'],
  [/Linux/, 'Linux']
];

export const cBrowserInfoOsUnknown = 'unknown';

/** Sent to the model as the `note` field of the tool result, next to the OS name */
export const cBrowserInfoNote = `This environment is browser-only, it cannot read or modify local files, and it cannot run shell commands. To get local file access, tell user to launch dgs-server with the ${cCliFlags.AGENT} argument.`;

export function fBuildBrowserInfoToolDefinition(): OpenAIToolDefinition {
  return {
    function: {
      description: fTranslate('dgs_78ca7c029136'),
      name: cBrowserInfoToolName,
      parameters: {
        properties: {},
        required: [],
        type: JsonSchemaType.cObject
      }
    },
    type: ToolCallType.cFunction
  };
}
