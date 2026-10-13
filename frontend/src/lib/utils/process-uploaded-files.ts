import { fTranslate } from '$lib/i18n';
import { fHeicFileToJpegDataURL, fIsHeicMimeType } from './heic-to-jpeg';
import { fConvertPDFToText } from './pdf-processing';
import { fIsSvgMimeType, fSvgBase64UrlToPngDataURL } from './svg-to-png';
import { fIsWebpMimeType, fWebpBase64UrlToPngDataURL } from './webp-to-png';
import { cSettingsKeys } from '$lib/constants';
import { FileTypeCategory } from '$lib/enums';
import { cModelsStore } from '$lib/stores/models/index.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import { fGetFileTypeCategory } from '$lib/utils';
import { toast } from 'svelte-sonner';

/**
 * Read a file as a data URL (base64 encoded)
 * @param file - The file to read
 * @returns Promise resolving to the data URL string
 */
function fReadFileAsDataURL(pFile: File): Promise<string> {
  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();

    cReader.onload = () => pResolve(cReader.result as string);
    cReader.onerror = () => pReject(cReader.error);
    cReader.readAsDataURL(pFile);
  });
}

/**
 * Read a file as UTF-8 text
 * @param file - The file to read
 * @returns Promise resolving to the text content
 */
function fReadFileAsUTF8(pFile: File): Promise<string> {
  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();

    cReader.onload = () => pResolve(cReader.result as string);
    cReader.onerror = () => pReject(cReader.error);
    cReader.readAsText(pFile);
  });
}

/**
 * Process uploaded files into ChatUploadedFile format with previews and content
 *
 * This function processes various file types and generates appropriate previews:
 * - Images: Base64 data URLs with format normalization (SVG/WebP → PNG)
 * - Text files: UTF-8 content extraction
 * - PDFs: Metadata only (processed later in conversion pipeline)
 * - Audio: Base64 data URLs for preview
 *
 * @param files - Array of File objects to process
 * @returns Promise resolving to array of ChatUploadedFile objects
 */
export async function fProcessFilesToChatUploaded(
  pFiles: File[],
  pActiveModelId?: string
): Promise<ChatUploadedFile[]> {
  const aResults: ChatUploadedFile[] = [];

  for (const file of pFiles) {
    const cId = Date.now().toString() + Math.random().toString(36).substr(2, 9);
    const cBase: ChatUploadedFile = {
      file,
      id: cId,
      name: file.name,
      size: file.size,
      type: file.type
    };

    try {
      if (fGetFileTypeCategory(file.type) === FileTypeCategory.IMAGE) {
        let vPreview = await fReadFileAsDataURL(file);

        // Normalize SVG and WebP to PNG, and HEIC to compressed JPEG, in previews
        if (fIsSvgMimeType(file.type)) {
          try {
            vPreview = await fSvgBase64UrlToPngDataURL(vPreview);
          } catch (vErr) {
            console.error('Failed to convert SVG to PNG:', vErr);
          }
        } else if (fIsWebpMimeType(file.type)) {
          try {
            vPreview = await fWebpBase64UrlToPngDataURL(vPreview);
          } catch (vErr) {
            console.error('Failed to convert WebP to PNG:', vErr);
          }
        } else if (fIsHeicMimeType(file.type)) {
          try {
            vPreview = await fHeicFileToJpegDataURL(file);
          } catch (vErr) {
            console.error('Failed to convert HEIC to PNG:', vErr);

            continue;
          }
        }

        aResults.push({ ...cBase, preview: vPreview });
      } else if (fGetFileTypeCategory(file.type) === FileTypeCategory.PDF) {
        // Extract text content from PDF for preview
        try {
          const cTextContent = await fConvertPDFToText(file);

          aResults.push({ ...cBase, textContent: cTextContent });
        } catch (vErr) {
          console.warn('Failed to extract text from PDF, adding without content:', vErr);
          aResults.push(cBase);
        }

        // Show suggestion toast if vision model is available but PDF as image is disabled
        const cHasVisionSupport = pActiveModelId
          ? cModelsStore.props.mModelSupportsVision(pActiveModelId)
          : false;
        const cCurrentConfig = cSettingsStore.vConfig;

        if (cHasVisionSupport && !cCurrentConfig.pdfAsImage) {
          toast.info(fTranslate('dgs_6c73de0a9199'), {
            action: {
              label: fTranslate('text_89b167823e87'),
              onClick: () => {
                cSettingsStore.mUpdateConfig(cSettingsKeys.PDF_AS_IMAGE, true);
                toast.success(fTranslate('text_28a5e30e1a90'), {
                  duration: 3000
                });
              }
            },
            duration: 8000
          });
        }
      } else if (fGetFileTypeCategory(file.type) === FileTypeCategory.AUDIO) {
        // Generate preview URL for audio files
        const cPreview = await fReadFileAsDataURL(file);

        aResults.push({ ...cBase, preview: cPreview });
      } else if (fGetFileTypeCategory(file.type) === FileTypeCategory.VIDEO) {
        // Generate preview URL for video files
        const cPreview = await fReadFileAsDataURL(file);

        aResults.push({ ...cBase, preview: cPreview });
      } else {
        // Fallback: treat unknown files as text
        try {
          const cTextContent = await fReadFileAsUTF8(file);

          aResults.push({ ...cBase, textContent: cTextContent });
        } catch (vErr) {
          console.warn('Failed to read file as text, adding without content:', vErr);
          aResults.push(cBase);
        }
      }
    } catch (error) {
      console.error('Error processing file', file.name, error);
      aResults.push(cBase);
    }
  }

  return aResults;
}
