import { fTranslate } from '$lib/i18n';
import {
  BuiltInTool,
  JsonSchemaType,
  MimeTypeAudio,
  MimeTypeImage,
  ToolCallType
} from '$lib/enums';
import type { OpenAIToolDefinition } from '$lib/types';

export const cReadMediaToolName = BuiltInTool.cBrowserReadMedia;

// header lines of the tool result, parsed back by the read_media renderer
export const cPrefixFile = 'File: ';
export const cPrefixSize = 'Size: ';
export const cPrefixMime = 'MIME: ';

/** Byte count of the `Size: ` header line, e.g. `Size: 12345 bytes` -> capture group 1 is `12345`. */
export const cReadMediaSizeRegex = new RegExp(`^${cPrefixSize}\\s*(\\d+)\\s*bytes`);

/** Image extensions the tool accepts. The server decodes images with stb_image, which has no webp or tiff. */
export const dReadMediaImageMime: Record<string, string> = {
  gif: MimeTypeImage.cGif,
  jpeg: MimeTypeImage.cJpeg,
  jpg: MimeTypeImage.cJpeg,
  png: MimeTypeImage.cPng
} as const;

/** Audio extensions the tool accepts. The `input_audio` API only takes wav and mp3. */
export const dReadMediaAudioMime: Record<string, string> = {
  mp3: MimeTypeAudio.cMp3Mpeg,
  wav: MimeTypeAudio.cWav
} as const;

/**
 * Build the read_media tool definition for the modalities the active model has.
 * At least one of the two flags must be true, otherwise the tool is not offered
 * at all - a model that cannot see or hear has nothing to do with the bytes.
 */
export function fBuildReadMediaToolDefinition(
  pSupportsVision: boolean,
  pSupportsAudio: boolean
): OpenAIToolDefinition {
  const aKinds: string[] = [];

  if (pSupportsVision) aKinds.push(`images (${Object.keys(dReadMediaImageMime).join(', ')})`);

  if (pSupportsAudio) aKinds.push(`audio (${Object.keys(dReadMediaAudioMime).join(', ')})`);

  return {
    function: {
      description: `Read a media file and attach it to the conversation so it can be perceived directly. Supports ${aKinds.join(' and ')}.`,
      name: cReadMediaToolName,
      parameters: {
        properties: {
          path: {
            description: fTranslate('dgs_f2787ade63eb'),
            type: JsonSchemaType.cString
          }
        },
        required: ['path'],
        type: JsonSchemaType.cObject
      }
    },
    type: ToolCallType.cFunction
  };
}
