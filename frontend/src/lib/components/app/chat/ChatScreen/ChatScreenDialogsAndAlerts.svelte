<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Trash2 } from '@lucide/svelte';
  import {
    DialogChatError,
    DialogConfirmation,
    DialogEmptyFileAlert,
    DialogFileUploadError
  } from '$lib/components/app';
  import { ErrorDialogType } from '$lib/enums';

  let {
    activeErrorDialog,
    emptyFileNames,
    fileUpload,
    handleDeleteConfirm,
    handleErrorDialogOpenChange,
    showDeleteDialog,
    showEmptyFileDialog
  } = $props();
</script>

<DialogFileUploadError
  bind:open={fileUpload.showFileErrorDialog}
  fileErrorData={fileUpload.fileErrorData}
/>

<DialogConfirmation
  bind:open={showDeleteDialog}
  cancelText="Cancel"
  confirmText="Delete"
  description={fTranslate('text_7b20577dc1ec')}
  icon={Trash2}
  onCancel={() => (showDeleteDialog = false)}
  onConfirm={handleDeleteConfirm}
  title={fTranslate('text_9ccd1fb969eb')}
  variant="destructive"
/>

<DialogEmptyFileAlert
  bind:open={showEmptyFileDialog}
  emptyFiles={emptyFileNames}
  onOpenChange={(pOpen) => {
    if (!pOpen) {
      emptyFileNames = [];
    }
  }}
/>

<DialogChatError
  contextInfo={activeErrorDialog?.contextInfo}
  message={activeErrorDialog?.message ?? ''}
  onOpenChange={handleErrorDialogOpenChange}
  open={Boolean(activeErrorDialog)}
  type={activeErrorDialog?.type ?? ErrorDialogType.SERVER}
/>
