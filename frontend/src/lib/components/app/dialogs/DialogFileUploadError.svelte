<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import * as AlertDialog from '$lib/components/ui/alert-dialog';

  interface Props {
    open: boolean;
    fileErrorData: {
      generallyUnsupported: File[];
      modalityUnsupported: File[];
      modalityReasons: Record<string, string>;
      supportedTypes: string[];
    };
    onOpenChange?: (pOpen: boolean) => void;
  }

  let { fileErrorData, onOpenChange, open = $bindable() }: Props = $props();

  function fHandleOpenChange(pNewOpen: boolean) {
    open = pNewOpen;

    onOpenChange?.(pNewOpen);
  }
</script>

<AlertDialog.fRoot onOpenChange={fHandleOpenChange} {open}>
  <AlertDialog.fPortal>
    <AlertDialog.Overlay />

    <AlertDialog.Content class="flex max-w-md flex-col">
      <AlertDialog.Header>
        <AlertDialog.Title>{fTranslate('text_41529cd15701')}</AlertDialog.Title>

        <AlertDialog.Description class="text-sm text-muted-foreground">
          {fTranslate('text_74a0549d84db')}
        </AlertDialog.Description>
      </AlertDialog.Header>

      <div class="!max-h-[50vh] min-h-0 flex-1 space-y-4 overflow-y-auto">
        {#if fileErrorData.generallyUnsupported.length > 0}
          <div class="space-y-2">
            <h4 class="text-sm font-medium text-destructive">{fTranslate('text_5bc2dbe4d9ab')}</h4>

            <div class="space-y-1">
              {#each fileErrorData.generallyUnsupported as vFile (vFile.name)}
                <div class="rounded-md bg-destructive/10 px-3 py-2">
                  <p class="font-mono text-sm break-all text-destructive">
                    {vFile.name}
                  </p>

                  <p class="mt-1 text-xs text-muted-foreground">
                    {fTranslate('text_8ae7419118b2')}
                  </p>
                </div>
              {/each}
            </div>
          </div>
        {/if}

        {#if fileErrorData.modalityUnsupported.length > 0}
          <div class="space-y-2">
            <div class="space-y-1">
              {#each fileErrorData.modalityUnsupported as vFile (vFile.name)}
                <div class="rounded-md bg-destructive/10 px-3 py-2">
                  <p class="font-mono text-sm break-all text-destructive">
                    {vFile.name}
                  </p>

                  <p class="mt-1 text-xs text-muted-foreground">
                    {fileErrorData.modalityReasons[vFile.name] || 'Not supported by current model'}
                  </p>
                </div>
              {/each}
            </div>
          </div>
        {/if}
      </div>

      <div class="rounded-md bg-muted/50 p-3">
        <h4 class="mb-2 text-sm font-medium">{fTranslate('text_47cd4672fca3')}</h4>

        <p class="text-sm text-muted-foreground">
          {fileErrorData.supportedTypes.join(', ')}
        </p>
      </div>

      <AlertDialog.Footer>
        <AlertDialog.Action onclick={() => fHandleOpenChange(false)}
          >{fTranslate('text_5ad3dbd1242a')}</AlertDialog.Action
        >
      </AlertDialog.Footer>
    </AlertDialog.Content>
  </AlertDialog.fPortal>
</AlertDialog.fRoot>
