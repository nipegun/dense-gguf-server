/**
 * ReadMediaService - Reads local media files for the read_media tool
 *
 * Encodes image and audio files as base64 data URLs with the metadata the
 * model needs. No reactive state; consumed by toolsStore.
 */

import { ToolsService } from './tools.service';
import {
  cFileExtensionSeparator,
  cFilePathSeparatorRegex,
  cNewline,
  cPrefixFile,
  cPrefixMime,
  cPrefixSize,
  dReadMediaAudioMime,
  dReadMediaImageMime,
  cRespTypeBase64
} from '$lib/constants';
import { BuiltInTool, ToolResponseField } from '$lib/enums';
import type { ToolExecutionResult } from '$lib/types';

/** Modalities of the model the tool call runs for. */
export interface ReadMediaCapabilities {
  audio: boolean;
  vision: boolean;
}

/** Lowercase extension of a path, without the dot. Empty when the file name has none. */
function fFileExtension(pPath: string): string {
  const name = pPath.split(cFilePathSeparatorRegex).pop() ?? '';
  const cDot = name.lastIndexOf(cFileExtensionSeparator);

  return cDot > 0 ? name.slice(cDot + 1).toLowerCase() : '';
}

/**
 * **ReadMediaService** - browser executor for the `read_media` tool
 *
 * The tool is synthetic: no such tool exists on the server. It reads the file
 * through the server `read_file` tool with the `base64` response type, then
 * turns the bytes into a data URI line. The agentic store lifts that line into
 * an image or audio attachment on the tool result message, which is what makes
 * the model perceive the file instead of reading a wall of base64.
 *
 * Living in the browser is what lets it exist only for models that can
 * actually use the result - the server has no idea which model is selected.
 *
 * @see buildReadMediaToolDefinition in constants/read-media.ts - tool schema sent to the LLM
 * @see agenticStore in stores/agentic/index.svelte.ts - tool dispatch and attachment extraction
 */
export class ReadMediaService {
  static async mExecuteTool(
    pParams: Record<string, unknown>,
    pCapabilities: ReadMediaCapabilities,
    pSignal?: AbortSignal,
    pCwd?: string
  ): Promise<ToolExecutionResult> {
    const cPath = typeof pParams.path === 'string' ? pParams.path : '';

    if (!cPath) {
      return { content: 'Error: missing "path" argument.', isError: true };
    }

    const cExtension = fFileExtension(cPath);
    const cImageMime = dReadMediaImageMime[cExtension];
    const cAudioMime = dReadMediaAudioMime[cExtension];

    let vResolvedMime: string | undefined;

    if (cImageMime && pCapabilities.vision) vResolvedMime = cImageMime;
    else if (cAudioMime && pCapabilities.audio) vResolvedMime = cAudioMime;

    if (!vResolvedMime) {
      const aSupported = [
        ...(pCapabilities.vision ? Object.keys(dReadMediaImageMime) : []),
        ...(pCapabilities.audio ? Object.keys(dReadMediaAudioMime) : [])
      ];
      // an unreadable-by-this-model file is a dead end, so say why instead of failing silently
      const cReason =
        cImageMime || cAudioMime
          ? `the current model cannot perceive ".${cExtension}" files`
          : `".${cExtension}" is not a supported media type`;

      return {
        content: `Error: ${cReason}. Supported: ${aSupported.join(', ')}.`,
        isError: true
      };
    }

    const dRaw = await ToolsService.mExecuteToolRaw(
      BuiltInTool.cServerReadFile,
      { path: cPath },
      pSignal,
      pCwd,
      cRespTypeBase64
    );

    if (ToolResponseField.ERROR in dRaw) {
      return { content: String(dRaw[ToolResponseField.ERROR]), isError: true };
    }

    const cBase64 = typeof dRaw.base64 === 'string' ? dRaw.base64 : '';

    if (!cBase64) {
      return { content: `Error: no data returned for ${cPath}.`, isError: true };
    }

    const cSizeBytes = typeof dRaw.size_bytes === 'number' ? dRaw.size_bytes : 0;
    const cContent = [
      `${cPrefixFile}${cPath}`,
      `${cPrefixSize}${cSizeBytes} bytes`,
      `${cPrefixMime}${vResolvedMime}`,
      `data:${vResolvedMime};base64,${cBase64}`
    ].join(cNewline);

    return { content: cContent, isError: false };
  }
}
