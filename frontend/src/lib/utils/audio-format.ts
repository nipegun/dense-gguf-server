import { FileTypeAudio, MimeTypeAudio } from '$lib/enums';
import type { AudioInputFormat } from '$lib/types/api';

/**
 * Map a MIME type to the AudioInputFormat expected by the API.
 */
export function fGetAudioInputFormat(pMimeType: string): AudioInputFormat {
  const cNormalizedMimeType = pMimeType.trim().toLowerCase();

  if (
    cNormalizedMimeType === MimeTypeAudio.cWav ||
    cNormalizedMimeType === MimeTypeAudio.cWave ||
    cNormalizedMimeType === MimeTypeAudio.cXWav ||
    cNormalizedMimeType === MimeTypeAudio.cXWave ||
    cNormalizedMimeType === MimeTypeAudio.cVndWave ||
    cNormalizedMimeType === MimeTypeAudio.cXPnWav
  ) {
    return FileTypeAudio.cWav;
  }

  return FileTypeAudio.cMp3;
}
