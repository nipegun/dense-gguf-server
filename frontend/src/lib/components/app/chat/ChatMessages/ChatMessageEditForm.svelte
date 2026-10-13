<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { AlertTriangle, X } from '@lucide/svelte';
  import { ChatForm, DialogConfirmation } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import { Switch } from '$lib/components/ui/switch';
  import { fGetChatMessageEditContext } from '$lib/contexts';
  import { KeyboardKey, MessageRole } from '$lib/enums';
  import { cChatStore } from '$lib/stores';
  import { fProcessFilesToChatUploaded } from '$lib/utils/browser-only';

  const cEditCtx = fGetChatMessageEditContext();

  let vSaveWithoutRegenerate = $state(false);
  let vShowDiscardDialog = $state(false);
  let vBranchAfterEdit = $state(false);

  let vIsUserMessage = $derived(cEditCtx.messageRole === MessageRole.cUser);
  let vIsAssistantMessage = $derived(cEditCtx.messageRole === MessageRole.cAssistant);

  let vHasUnsavedChanges = $derived.by(() => {
    if (cEditCtx.editedContent !== cEditCtx.originalContent) return true;

    if (cEditCtx.editedUploadedFiles.length > 0) return true;

    const cExtrasChanged =
      cEditCtx.editedExtras.length !== cEditCtx.originalExtras.length ||
      cEditCtx.editedExtras.some((pExtra, pI) => pExtra !== cEditCtx.originalExtras[pI]);

    if (cExtrasChanged) return true;

    return false;
  });

  let vHasAttachments = $derived(
    (cEditCtx.editedExtras && cEditCtx.editedExtras.length > 0) ||
      (cEditCtx.editedUploadedFiles && cEditCtx.editedUploadedFiles.length > 0)
  );

  let vCanSubmit = $derived(cEditCtx.editedContent.trim().length > 0 || vHasAttachments);

  function fHandleGlobalKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === KeyboardKey.cEscape) {
      pEvent.preventDefault();
      fAttemptCancel();
    }
  }

  function fAttemptCancel() {
    if (vHasUnsavedChanges) {
      vShowDiscardDialog = true;
    } else {
      cEditCtx.cancel();
    }
  }

  function fHandleSubmit() {
    if (!vCanSubmit) return;

    if (vIsUserMessage && vSaveWithoutRegenerate && cEditCtx.showSaveOnlyOption) {
      cEditCtx.saveOnly();
    } else {
      if (vIsAssistantMessage && cEditCtx.setShouldBranchAfterEdit) {
        cEditCtx.setShouldBranchAfterEdit(vBranchAfterEdit);
      }

      cEditCtx.save();
    }

    vSaveWithoutRegenerate = false;
    vBranchAfterEdit = false;
  }

  function fHandleAttachmentRemove(pIndex: number) {
    const aNewExtras = [...cEditCtx.editedExtras];

    aNewExtras.splice(pIndex, 1);
    cEditCtx.setExtras(aNewExtras);
  }

  function fHandleUploadedFileRemove(pFileId: string) {
    const aNewFiles = cEditCtx.editedUploadedFiles.filter((pF) => pF.id !== pFileId);

    cEditCtx.setUploadedFiles(aNewFiles);
  }

  async function fHandleFilesAdd(pFiles: File[]) {
    const aProcessed = await fProcessFilesToChatUploaded(pFiles);

    cEditCtx.setUploadedFiles([...cEditCtx.editedUploadedFiles, ...aProcessed]);
  }

  $effect(() => {
    cChatStore.mSetEditModeActive(fHandleFilesAdd);

    return () => {
      cChatStore.mClearEditMode();
    };
  });
</script>

<svelte:window onkeydown={fHandleGlobalKeydown} />

<div class="relative w-full max-w-[80%]">
  <ChatForm
    bind:uploadedFiles={cEditCtx.editedUploadedFiles}
    attachments={cEditCtx.editedExtras}
    onAttachmentRemove={fHandleAttachmentRemove}
    onFilesAdd={fHandleFilesAdd}
    onSubmit={fHandleSubmit}
    onUploadedFileRemove={fHandleUploadedFileRemove}
    onValueChange={cEditCtx.setContent}
    placeholder={fTranslate('text_0bc9ade7fa33')}
    showAddButton={cEditCtx.messageRole === MessageRole.cUser}
    showModelSelector={cEditCtx.messageRole === MessageRole.cUser}
    value={cEditCtx.editedContent}
  />
</div>

<div class="mt-2 flex w-full max-w-[80%] items-center justify-between">
  {#if vIsUserMessage && cEditCtx.showSaveOnlyOption}
    <div class="flex items-center gap-2">
      <Switch bind:checked={vSaveWithoutRegenerate} class="scale-75" id="save-only-switch" />

      <label class="cursor-pointer text-xs text-muted-foreground" for="save-only-switch">
        {fTranslate('text_d6ba6ecbbf7a')}
      </label>
    </div>
  {:else if vIsAssistantMessage}
    <div class="flex items-center gap-2">
      <Switch bind:checked={vBranchAfterEdit} class="scale-75" id="branch-after-edit" />

      <label class="cursor-pointer text-xs text-muted-foreground" for="branch-after-edit">
        {fTranslate('text_cc2c478ba8cd')}
      </label>
    </div>
  {:else}
    <div></div>
  {/if}

  <Button class="h-7 px-3 text-xs" onclick={fAttemptCancel} size="sm" variant="ghost">
    <X class="mr-1 h-3 w-3" />

    {fTranslate('text_19766ed6ccb2')}
  </Button>
</div>

<DialogConfirmation
  bind:open={vShowDiscardDialog}
  cancelText="Keep editing"
  confirmText="Discard"
  description={fTranslate('text_e8515d324597')}
  icon={AlertTriangle}
  onCancel={() => (vShowDiscardDialog = false)}
  onConfirm={cEditCtx.cancel}
  title={fTranslate('text_85bcf41666a5')}
  variant="destructive"
/>
