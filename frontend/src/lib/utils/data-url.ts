/**
 * Creates a base64 data URL from MIME type and base64-encoded data.
 *
 * @param mimeType - The MIME type (e.g., 'image/png', 'audio/mp3')
 * @param base64Data - The base64-encoded data
 * @returns A data URL string in format 'data:{mimeType};base64,{data}'
 */
export function fCreateBase64DataUrl(pMimeType: string, pBase64Data: string): string {
  return `data:${pMimeType};base64,${pBase64Data}`;
}
