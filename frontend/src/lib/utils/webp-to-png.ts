import { fTranslate } from '$lib/i18n';
import { FileExtensionImage, MimeTypeImage } from '$lib/enums';

/**
 * Convert a WebP base64 data URL to a PNG data URL
 * @param base64UrlWebp - The WebP base64 data URL to convert
 * @param backgroundColor - Background color for the PNG (default: 'white')
 * @returns Promise resolving to PNG data URL
 */
export function fWebpBase64UrlToPngDataURL(
  pBase64UrlWebp: string,
  pBackgroundColor: string = 'white'
): Promise<string> {
  return new Promise((pResolve, pReject) => {
    try {
      const cImg = new Image();

      cImg.onload = () => {
        const cCanvas = document.createElement('canvas');
        const cCtx = cCanvas.getContext('2d');

        if (!cCtx) {
          pReject(new Error(fTranslate('text_1eb30178a933')));

          return;
        }

        const cTargetWidth = cImg.naturalWidth || 300;
        const cTargetHeight = cImg.naturalHeight || 300;

        cCanvas.width = cTargetWidth;
        cCanvas.height = cTargetHeight;

        if (pBackgroundColor) {
          cCtx.fillStyle = pBackgroundColor;
          cCtx.fillRect(0, 0, cCanvas.width, cCanvas.height);
        }

        cCtx.drawImage(cImg, 0, 0, cTargetWidth, cTargetHeight);

        pResolve(cCanvas.toDataURL(MimeTypeImage.cPng));
      };

      cImg.onerror = () => {
        pReject(new Error(fTranslate('text_9eefd1f13e8f')));
      };

      cImg.src = pBase64UrlWebp;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const cErrorMessage = `Error converting WebP to PNG: ${message}`;

      console.error(cErrorMessage, error);
      pReject(new Error(cErrorMessage));
    }
  });
}

/**
 * Check if a file is a WebP based on its MIME type
 * @param file - The file to check
 * @returns True if the file is a WebP
 */
export function fIsWebpFile(pFile: File): boolean {
  return (
    pFile.type === MimeTypeImage.cWebp ||
    pFile.name.toLowerCase().endsWith(FileExtensionImage.cWebp)
  );
}

/**
 * Check if a MIME type represents a WebP
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type is image/webp
 */
export function fIsWebpMimeType(pMimeType: string): boolean {
  return pMimeType === MimeTypeImage.cWebp;
}
