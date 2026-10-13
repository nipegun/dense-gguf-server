<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelsManager from '$lib/components/app/models/ModelsManager/ModelsManager.svelte';
  import * as Dialog from '$lib/components/ui/dialog';
  import { ModelIcon } from '$lib/constants';

  interface Props {
    open?: boolean;
    onOpenChange?: (pOpen: boolean) => void;
  }

  let { onOpenChange, open = $bindable(false) }: Props = $props();

  function fHandleOpenChange(pValue: boolean) {
    open = pValue;
    onOpenChange?.(pValue);
  }
</script>

<Dialog.fRoot onOpenChange={fHandleOpenChange} {open}>
  <Dialog.Content
    class="max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! max-md:border-0! md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-380! flex flex-col p-4 pb-0"
    onOpenAutoFocus={(pEvent) => pEvent.preventDefault()}
  >
    <Dialog.Header class="flex flex-row items-center pr-8">
      <Dialog.Title class="flex min-h-7 items-center gap-2">
        <ModelIcon class="h-5 w-5 shrink-0" />

        <span>{fTranslate('dgs_d17d2d78d76e')}</span>
      </Dialog.Title>

      <Dialog.Description class="sr-only">
        {fTranslate('dgs_9fef41247f7c')}
      </Dialog.Description>
    </Dialog.Header>

    <div class="min-h-0 flex-1">
      <ModelsManager class="h-full" />
    </div>
  </Dialog.Content>
</Dialog.fRoot>
