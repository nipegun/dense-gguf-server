<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { RotateCcw } from '@lucide/svelte';
  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { Button } from '$lib/components/ui/button';
  import { cSettingsStore } from '$lib/stores';

  interface Props {
    onReset?: () => void;
    onSave?: () => void;
  }

  let { onReset, onSave }: Props = $props();

  let vShowResetDialog = $state(false);

  function fHandleResetClick() {
    vShowResetDialog = true;
  }

  function fHandleConfirmReset() {
    cSettingsStore.mForceSyncWithServerDefaults();
    onReset?.();

    vShowResetDialog = false;
  }

  function fHandleSave() {
    onSave?.();
  }
</script>

<div class="sticky bottom-0 mx-auto mt-4 flex w-full justify-between pb-4 md:pb-0">
  <div class="flex gap-2">
    <Button onclick={fHandleResetClick} variant="outline">
      <RotateCcw class="h-3 w-3" />

      {fTranslate('text_bc5b45ae7b60')}
    </Button>
  </div>

  <Button onclick={fHandleSave}>{fTranslate('text_7f3a3b142814')}</Button>
</div>

<AlertDialog.fRoot bind:open={vShowResetDialog}>
  <AlertDialog.Content>
    <AlertDialog.Header>
      <AlertDialog.Title>{fTranslate('text_7047e493541d')}</AlertDialog.Title>

      <AlertDialog.Description>
        {fTranslate('dgs_395d23b1d04b')}
      </AlertDialog.Description>
    </AlertDialog.Header>

    <AlertDialog.Footer>
      <AlertDialog.Cancel>{fTranslate('text_19766ed6ccb2')}</AlertDialog.Cancel>

      <AlertDialog.Action onclick={fHandleConfirmReset}
        >{fTranslate('text_c0334bae1053')}</AlertDialog.Action
      >
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.fRoot>
