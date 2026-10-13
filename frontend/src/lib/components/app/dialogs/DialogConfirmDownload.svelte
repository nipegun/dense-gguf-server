<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import DialogConfirmation from '$lib/components/app/dialogs/DialogConfirmation.svelte';
  import { ModelDownloadConfirmAction } from '$lib/enums';
  import { cModelsStore } from '$lib/stores';

  interface Props {
    /** Action being confirmed; drives the wording. */
    action: ModelDownloadConfirmAction;
    /** `<repo>:<tag>` the action targets. */
    repoWithTag: string;
    onClose: () => void;
    /** Overrides the default store removal; defaults to removing the entry. */
    onConfirm?: (pRepoWithTag: string) => void;
    open?: boolean;
  }

  let { action, onClose, onConfirm, open = true, repoWithTag }: Props = $props();

  // Both actions resolve through the same store removal (cancelDownload drops a
  // running download's partial files or a cached model's files); only the copy
  // differs. One component so the discover chips and the selector rows word the
  // destructive confirmations identically.
  const cCopy = {
    [ModelDownloadConfirmAction.cCancel]: {
      cancelText: 'Keep downloading',
      confirmText: 'Cancel download',
      description: (pName: string) =>
        `This stops the download of ${pName} and removes the partial files. Pause it instead to keep the progress.`,
      title: fTranslate('dgs_de5b23e69cd1')
    },
    [ModelDownloadConfirmAction.DELETE]: {
      cancelText: 'Keep model',
      confirmText: 'Delete',
      description: (pName: string) =>
        `This permanently removes ${pName} from disk. You can download it again later.`,
      title: fTranslate('dgs_b7828ba00ee1')
    }
  } as const;

  let vCopy = $derived(cCopy[action]);
  let vDisplayName = $derived(cModelsStore.mToDisplayName(repoWithTag));

  function fConfirm() {
    if (onConfirm) onConfirm(repoWithTag);
    else void cModelsStore.status.mCancelDownload(repoWithTag);

    onClose();
  }
</script>

<DialogConfirmation
  cancelText={vCopy.cancelText}
  confirmText={vCopy.confirmText}
  description={vCopy.description(vDisplayName)}
  onCancel={onClose}
  onConfirm={fConfirm}
  {open}
  title={vCopy.title}
  variant="destructive"
/>
