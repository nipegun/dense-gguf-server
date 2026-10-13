import { fSetChatMessageEditContext } from '$lib/contexts';
import { MessageRole } from '$lib/enums';
import { fParseFilesToMessageExtras } from '$lib/utils/convert-files-to-extra';

interface UseChatMessageEditContextOptions {
  getContent: () => string;
  getExtras: () => DatabaseMessageExtra[];
  showSaveOnlyOption?: boolean;
  onSave: (pContent: string, pExtras?: DatabaseMessageExtra[]) => void;
}

export function fUseChatMessageEditContext(pOptions: UseChatMessageEditContextOptions) {
  let vIsEditing = $state(false);
  let vEditedContent = $state('');
  let aEditedExtras = $state<DatabaseMessageExtra[]>([]);
  let aEditedUploadedFiles = $state<ChatUploadedFile[]>([]);

  function fHandleEdit() {
    vEditedContent = pOptions.getContent();
    aEditedExtras = [...pOptions.getExtras()];
    aEditedUploadedFiles = [];
    vIsEditing = true;
  }

  async function fHandleSaveEdit() {
    const cTrimmed = vEditedContent.trim();

    if (!cTrimmed && aEditedExtras.length === 0 && aEditedUploadedFiles.length === 0) return;

    let aFinalExtras: DatabaseMessageExtra[] = $state.snapshot(aEditedExtras);

    if (aEditedUploadedFiles.length > 0) {
      const cPlainFiles = $state.snapshot(aEditedUploadedFiles);
      const cResult = await fParseFilesToMessageExtras(cPlainFiles);
      const aNewExtras = cResult?.extras || [];

      aFinalExtras = [...aFinalExtras, ...aNewExtras];
    }

    pOptions.onSave(cTrimmed, aFinalExtras.length > 0 ? aFinalExtras : undefined);
    vIsEditing = false;
  }

  function fHandleCancelEdit() {
    vIsEditing = false;
  }

  fSetChatMessageEditContext({
    cancel: fHandleCancelEdit,
    get editedContent() {
      return vEditedContent;
    },
    get editedExtras() {
      return aEditedExtras;
    },
    get editedUploadedFiles() {
      return aEditedUploadedFiles;
    },
    get isEditing() {
      return vIsEditing;
    },
    get messageRole() {
      return MessageRole.cUser;
    },
    get originalContent() {
      return pOptions.getContent();
    },
    get originalExtras() {
      return pOptions.getExtras();
    },
    save: fHandleSaveEdit,
    saveOnly: fHandleSaveEdit,
    setContent: (pC: string) => {
      vEditedContent = pC;
    },
    setExtras: (pE: DatabaseMessageExtra[]) => {
      aEditedExtras = pE;
    },
    setUploadedFiles: (pF: ChatUploadedFile[]) => {
      aEditedUploadedFiles = pF;
    },
    get shouldBranchAfterEdit() {
      return false;
    },
    get showBranchAfterEditOption() {
      return false;
    },
    get showSaveOnlyOption() {
      return pOptions.showSaveOnlyOption ?? false;
    },
    startEdit: fHandleEdit
  });

  return {
    handleCancelEdit: fHandleCancelEdit,
    handleEdit: fHandleEdit,
    handleSaveEdit: fHandleSaveEdit,
    get isEditing() {
      return vIsEditing;
    }
  };
}
