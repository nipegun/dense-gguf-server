/**
 * File validation utilities based on model modalities
 * Ensures only compatible file types are processed based on model capabilities
 */

import { FileTypeCategory } from '$lib/enums';
import type { ModalityCapabilities } from '$lib/types';
import { fGetFileTypeCategory } from '$lib/utils';

/**
 * Check if a file type is supported by the given modalities
 * @param filename - The filename to check
 * @param mimeType - The MIME type of the file
 * @param capabilities - The modality capabilities to check against
 * @returns true if the file type is supported
 */
export function fIsFileTypeSupportedByModel(
  pFilename: string,
  pMimeType: string | undefined,
  pCapabilities: ModalityCapabilities
): boolean {
  const cCategory = pMimeType ? fGetFileTypeCategory(pMimeType) : null;

  // If we can't determine the category from MIME type, fall back to general support check
  if (!cCategory) {
    // For unknown types, only allow if they might be text files
    // This is a conservative approach for edge cases
    return true; // Let the existing isFileTypeSupported handle this
  }

  switch (cCategory) {
    case FileTypeCategory.TEXT:
      // Text files are always supported
      return true;

    case FileTypeCategory.PDF:
      // PDFs are always supported (will be processed as text for non-vision models)
      return true;

    case FileTypeCategory.IMAGE:
      // Images require vision support
      return pCapabilities.hasVision;

    case FileTypeCategory.AUDIO:
      // Audio files require audio support
      return pCapabilities.hasAudio;

    case FileTypeCategory.VIDEO:
      // Video files require video support
      return pCapabilities.hasVideo;

    default:
      // Unknown categories - be conservative and allow
      return true;
  }
}

/**
 * Filter files based on model modalities and return supported/unsupported lists
 * @param files - Array of files to filter
 * @param capabilities - The modality capabilities to check against
 * @returns Object with supportedFiles and unsupportedFiles arrays
 */
export function fFilterFilesByModalities(
  pFiles: File[],
  pCapabilities: ModalityCapabilities
): {
  supportedFiles: File[];
  unsupportedFiles: File[];
  modalityReasons: Record<string, string>;
} {
  const aSupportedFiles: File[] = [];
  const aUnsupportedFiles: File[] = [];
  const dModalityReasons: Record<string, string> = {};
  const { hasAudio: vHasAudio, hasVideo: vHasVideo, hasVision: vHasVision } = pCapabilities;

  for (const file of pFiles) {
    const cCategory = fGetFileTypeCategory(file.type);

    let vIsSupported = true;
    let vReason = '';

    switch (cCategory) {
      case FileTypeCategory.IMAGE:
        if (!vHasVision) {
          vIsSupported = false;
          vReason = 'Images require a vision-capable model';
        }

        break;

      case FileTypeCategory.AUDIO:
        if (!vHasAudio) {
          vIsSupported = false;
          vReason = 'Audio files require an audio-capable model';
        }

        break;

      case FileTypeCategory.VIDEO:
        if (!vHasVideo) {
          vIsSupported = false;
          vReason = 'Video files require a video-capable model';
        }

        break;

      case FileTypeCategory.TEXT:
      case FileTypeCategory.PDF:
        // Always supported
        break;

      default:
        vIsSupported = false;
        vReason = 'Unsupported attachment type';
        break;
    }

    if (vIsSupported) {
      aSupportedFiles.push(file);
    } else {
      aUnsupportedFiles.push(file);
      dModalityReasons[file.name] = vReason;
    }
  }

  return {
    modalityReasons: dModalityReasons,
    supportedFiles: aSupportedFiles,
    unsupportedFiles: aUnsupportedFiles
  };
}

/**
 * Generate a user-friendly error message for unsupported files
 * @param unsupportedFiles - Array of unsupported files
 * @param modalityReasons - Reasons why files are unsupported
 * @param capabilities - The modality capabilities to check against
 * @returns Formatted error message
 */
export function fGenerateModalityErrorMessage(
  pUnsupportedFiles: File[],
  pModalityReasons: Record<string, string>,
  pCapabilities: ModalityCapabilities
): string {
  if (pUnsupportedFiles.length === 0) return '';

  const { hasAudio: vHasAudio, hasVideo: vHasVideo, hasVision: vHasVision } = pCapabilities;

  let vMessage = '';

  if (pUnsupportedFiles.length === 1) {
    const file = pUnsupportedFiles[0];
    const cReason = pModalityReasons[file.name];

    vMessage = `The file "${file.name}" cannot be uploaded: ${cReason}.`;
  } else {
    const cFileNames = pUnsupportedFiles.map((pF) => pF.name).join(', ');

    vMessage = `The following files cannot be uploaded: ${cFileNames}.`;
  }

  // Add helpful information about what is supported
  const aSupportedTypes: string[] = ['text files', 'PDFs'];

  if (vHasVision) aSupportedTypes.push('images');

  if (vHasAudio) aSupportedTypes.push('audio files');

  if (vHasVideo) aSupportedTypes.push('video files');

  vMessage += ` This model supports: ${aSupportedTypes.join(', ')}.`;

  return vMessage;
}
