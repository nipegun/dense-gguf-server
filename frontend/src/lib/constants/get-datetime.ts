import { fTranslate } from '$lib/i18n';
import { BuiltInTool, JsonSchemaType, ToolCallType } from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const cGetDatetimeToolName = BuiltInTool.cBrowserGetDatetime;

export function fBuildGetDatetimeToolDefinition(): OpenAIToolDefinition {
  return {
    function: {
      description: fTranslate('dgs_53bc32aba024'),
      name: cGetDatetimeToolName,
      parameters: {
        properties: {},
        required: [],
        type: JsonSchemaType.cObject
      }
    },
    type: ToolCallType.cFunction
  };
}
