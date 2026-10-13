import { fTranslate } from '$lib/i18n';
import { fConvertPDFToImage, fConvertPDFToText } from './pdf-processing';
import { fIsSvgMimeType, fSvgBase64UrlToPngDataURL } from './svg-to-png';
import { fIsLikelyTextFile, fReadFileAsText } from './text-files';
import { fIsWebpMimeType, fWebpBase64UrlToPngDataURL } from './webp-to-png';
import { cSettingsKeys } from '$lib/constants';
import { AttachmentType, FileTypeCategory, SpecialFileType } from '$lib/enums';
import { cModelsStore } from '$lib/stores/models/index.svelte';
import { cSettingsStore } from '$lib/stores/settings/index.svelte';
import type { ChatUploadedFile, DatabaseMessageExtra, FileProcessingResult } from '$lib/types';
import { fGetFileTypeCategory } from '$lib/utils';
import { toast } from 'svelte-sonner';

function fReadFileAsBase64(pFile: File): Promise<string> {
  return new Promise((pResolve, pReject) => {
    const cReader = new FileReader();

    cReader.onload = () => {
      // Extract base64 data without the data URL prefix
      const cDataUrl = cReader.result as string;
      const cBase64 = cDataUrl.split(',')[1];

      pResolve(cBase64);
    };

    cReader.onerror = () => pReject(cReader.error);

    cReader.readAsDataURL(pFile);
  });
}

export async function fParseFilesToMessageExtras(
  pFiles: ChatUploadedFile[],
  pActiveModelId?: string
): Promise<FileProcessingResult> {
  const aExtras: DatabaseMessageExtra[] = [];
  const aEmptyFiles: string[] = [];

  for (const file of pFiles) {
    if (file.type === SpecialFileType.MCP_PROMPT && file.mcpPrompt) {
      aExtras.push({
        arguments: file.mcpPrompt.arguments,
        content: file.textContent ?? '',
        name: file.name,
        promptName: file.mcpPrompt.promptName,
        serverName: file.mcpPrompt.serverName,
        size: file.size,
        type: AttachmentType.MCP_PROMPT
      });

      continue;
    }

    if (fGetFileTypeCategory(file.type) === FileTypeCategory.IMAGE) {
      if (file.preview) {
        let vBase64Url = file.preview;

        if (fIsSvgMimeType(file.type)) {
          try {
            vBase64Url = await fSvgBase64UrlToPngDataURL(vBase64Url);
          } catch (error) {
            console.error('Failed to convert SVG to PNG for database storage:', error);
          }
        } else if (fIsWebpMimeType(file.type)) {
          try {
            vBase64Url = await fWebpBase64UrlToPngDataURL(vBase64Url);
          } catch (error) {
            console.error('Failed to convert WebP to PNG for database storage:', error);
          }
        }

        aExtras.push({
          base64Url: vBase64Url,
          name: file.name,
          size: file.size,
          type: AttachmentType.IMAGE
        });
      }
    } else if (fGetFileTypeCategory(file.type) === FileTypeCategory.AUDIO) {
      // Process audio files (MP3 and WAV)
      try {
        const cBase64Data = await fReadFileAsBase64(file.file);

        aExtras.push({
          base64Data: cBase64Data,
          mimeType: file.type,
          name: file.name,
          size: file.size,
          type: AttachmentType.AUDIO
        });
      } catch (error) {
        console.error(`Failed to process audio file ${file.name}:`, error);
      }
    } else if (fGetFileTypeCategory(file.type) === FileTypeCategory.VIDEO) {
      // Process video files (MP4, etc)
      try {
        const cBase64Data = await fReadFileAsBase64(file.file);

        aExtras.push({
          base64Data: cBase64Data,
          mimeType: file.type,
          name: file.name,
          size: file.size,
          type: AttachmentType.VIDEO
        });
      } catch (error) {
        console.error(`Failed to process video file ${file.name}:`, error);
      }
    } else if (fGetFileTypeCategory(file.type) === FileTypeCategory.PDF) {
      try {
        // Always get base64 data for preview functionality
        const cBase64Data = await fReadFileAsBase64(file.file);
        const cCurrentConfig = cSettingsStore.vConfig;
        // Use per-model vision check for router mode
        const cHasVisionSupport = pActiveModelId
          ? cModelsStore.props.mModelSupportsVision(pActiveModelId)
          : false;

        // Force PDF-to-text for non-vision models
        let vShouldProcessAsImages = Boolean(cCurrentConfig.pdfAsImage) && cHasVisionSupport;

        // If user had pdfAsImage enabled but model doesn't support vision, update setting and notify
        if (cCurrentConfig.pdfAsImage && !cHasVisionSupport) {
          console.log('Non-vision model detected: forcing PDF-to-text mode and updating settings');

          // Update the setting in localStorage
          cSettingsStore.mUpdateConfig(cSettingsKeys.PDF_AS_IMAGE, false);

          // Show toast notification to user
          toast.warning(fTranslate('text_cc2587a65a82'), {
            duration: 5000
          });

          vShouldProcessAsImages = false;
        }

        if (vShouldProcessAsImages) {
          // Process PDF as images (only for vision models)
          try {
            const aImages = await fConvertPDFToImage(file.file);

            // Show success toast for PDF image processing
            toast.success(
              `PDF "${file.name}" processed as ${aImages.length} images for vision model.`,
              {
                duration: 3000
              }
            );

            aExtras.push({
              base64Data: cBase64Data,
              content: `PDF file with ${aImages.length} pages`,
              images: aImages,
              name: file.name,
              processedAsImages: true,
              size: file.size,
              type: AttachmentType.PDF
            });
          } catch (vImageError) {
            console.warn(
              `Failed to process PDF ${file.name} as images, falling back to text:`,
              vImageError
            );

            // Fallback to text processing
            const cContent = await fConvertPDFToText(file.file);

            aExtras.push({
              base64Data: cBase64Data,
              content: cContent,
              name: file.name,
              processedAsImages: false,
              size: file.size,
              type: AttachmentType.PDF
            });
          }
        } else {
          // Process PDF as text (default or forced for non-vision models)
          const cContent = await fConvertPDFToText(file.file);

          // Show success toast for PDF text processing
          toast.success(`PDF "${file.name}" processed as text content.`, {
            duration: 3000
          });

          aExtras.push({
            base64Data: cBase64Data,
            content: cContent,
            name: file.name,
            processedAsImages: false,
            size: file.size,
            type: AttachmentType.PDF
          });
        }
      } catch (error) {
        console.error(`Failed to process PDF file ${file.name}:`, error);
      }
    } else {
      try {
        const cContent = await fReadFileAsText(file.file);

        // Check if file is empty
        if (cContent.trim() === '') {
          console.warn(`File ${file.name} is empty and will be skipped`);
          aEmptyFiles.push(file.name);
        } else if (fIsLikelyTextFile(cContent)) {
          aExtras.push({
            content: cContent,
            name: file.name,
            size: file.size,
            type: AttachmentType.TEXT
          });
        } else {
          console.warn(`File ${file.name} appears to be binary and will be skipped`);
        }
      } catch (error) {
        console.error(`Failed to read file ${file.name}:`, error);
      }
    }
  }

  return { emptyFiles: aEmptyFiles, extras: aExtras };
}
