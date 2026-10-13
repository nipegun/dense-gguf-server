/**
 * File upload lifecycle for the ChatScreen form.
 *
 * Owns the queue of processed `ChatUploadedFile`, the rejection-by-capability
 * dialog state, and the dual-layer validation pipeline (general format +
 * model modality). The caller provides the active model's capabilities and ID
 * as reactive getters so validation tracks the model in real time.
 */

import { fFilterFilesByModalities, fIsFileTypeSupported } from '$lib/utils';
import { fProcessFilesToChatUploaded } from '$lib/utils/browser-only';

interface UseChatScreenFileUploadOptions {
  capabilities: () => { hasVision: boolean; hasAudio: boolean; hasVideo: boolean };
  activeModelId: () => string | null | undefined;
}

export interface FileErrorData {
  generallyUnsupported: File[];
  modalityUnsupported: File[];
  modalityReasons: Record<string, string>;
  supportedTypes: string[];
}

export function fUseChatScreenFileUpload(pOptions: UseChatScreenFileUploadOptions) {
  let aUploadedFiles = $state<ChatUploadedFile[]>([]);
  let vShowFileErrorDialog = $state(false);
  let vFileErrorData = $state<FileErrorData>({
    generallyUnsupported: [],
    modalityReasons: {},
    modalityUnsupported: [],
    supportedTypes: []
  });

  async function fProcessFiles(pFiles: File[]) {
    const aGenerallySupported: File[] = [];
    const aGenerallyUnsupported: File[] = [];

    for (const file of pFiles) {
      if (fIsFileTypeSupported(file.name, file.type)) {
        aGenerallySupported.push(file);
      } else {
        aGenerallyUnsupported.push(file);
      }
    }

    const {
      modalityReasons: dModalityReasons,
      supportedFiles: aSupportedFiles,
      unsupportedFiles: aUnsupportedFiles
    } = fFilterFilesByModalities(aGenerallySupported, pOptions.capabilities());
    const aAllUnsupportedFiles = [...aGenerallyUnsupported, ...aUnsupportedFiles];

    if (aAllUnsupportedFiles.length > 0) {
      const aSupportedTypes: string[] = ['text files', 'PDFs'];
      const cCaps = pOptions.capabilities();

      if (cCaps.hasVision) aSupportedTypes.push('images');

      if (cCaps.hasAudio) aSupportedTypes.push('audio files');

      if (cCaps.hasVideo) aSupportedTypes.push('video files');

      vFileErrorData = {
        generallyUnsupported: aGenerallyUnsupported,
        modalityReasons: dModalityReasons,
        modalityUnsupported: aUnsupportedFiles,
        supportedTypes: aSupportedTypes
      };
      vShowFileErrorDialog = true;
    }

    if (aSupportedFiles.length > 0) {
      const aProcessed = await fProcessFilesToChatUploaded(
        aSupportedFiles,
        pOptions.activeModelId() ?? undefined
      );

      aUploadedFiles = [...aUploadedFiles, ...aProcessed];
    }
  }

  function fHandleFileUpload(pFiles: File[]) {
    return fProcessFiles(pFiles);
  }

  function fHandleFileRemove(pFileId: string) {
    aUploadedFiles = aUploadedFiles.filter((pF) => pF.id !== pFileId);
  }

  return {
    get fileErrorData() {
      return vFileErrorData;
    },
    handleFileRemove: fHandleFileRemove,
    handleFileUpload: fHandleFileUpload,
    get showFileErrorDialog() {
      return vShowFileErrorDialog;
    },
    set showFileErrorDialog(pValue) {
      vShowFileErrorDialog = pValue;
    },
    get uploadedFiles() {
      return aUploadedFiles;
    },
    set uploadedFiles(pValue) {
      aUploadedFiles = pValue;
    }
  };
}
