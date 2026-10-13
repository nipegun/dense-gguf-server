import { fTranslate } from '$lib/i18n';
import { MimeTypeImage } from '$lib/enums';

/**
 * Convert an SVG base64 data URL to a PNG data URL
 * @param base64UrlSvg - The SVG base64 data URL to convert
 * @param backgroundColor - Background color for the PNG (default: 'white')
 * @returns Promise resolving to PNG data URL
 */
export function fSvgBase64UrlToPngDataURL(
  pBase64UrlSvg: string,
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
        pReject(new Error(fTranslate('text_eec057d447a0')));
      };

      cImg.src = pBase64UrlSvg;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const cErrorMessage = `Error converting SVG to PNG: ${message}`;

      console.error(cErrorMessage, error);
      pReject(new Error(cErrorMessage));
    }
  });
}

/**
 * Check if a file is an SVG based on its MIME type
 * @param file - The file to check
 * @returns True if the file is an SVG
 */
export function fIsSvgFile(pFile: File): boolean {
  return pFile.type === MimeTypeImage.SVG;
}

/**
 * Check if a MIME type represents an SVG
 * @param mimeType - The MIME type to check
 * @returns True if the MIME type is image/svg+xml
 */
export function fIsSvgMimeType(pMimeType: string): boolean {
  return pMimeType === MimeTypeImage.SVG;
}
