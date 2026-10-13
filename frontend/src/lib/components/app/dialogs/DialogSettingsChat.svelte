<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Settings, X } from '@lucide/svelte';
  import { SettingsChat } from '$lib/components/app/settings';
  import * as Dialog from '$lib/components/ui/dialog';
  import { cPanelCloseMobileClass } from '$lib/constants';
  import { cDeviceStore } from '$lib/stores';

  interface Props {
    open?: boolean;
    onOpenChange?: (pOpen: boolean) => void;
    initialSection?: string;
  }

  let { initialSection, onOpenChange, open = $bindable(false) }: Props = $props();

  function fHandleOpenChange(pValue: boolean) {
    open = pValue;
    onOpenChange?.(pValue);
  }
</script>

<Dialog.fRoot onOpenChange={fHandleOpenChange} {open}>
  <Dialog.Content
    class="max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! max-md:border-0! md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-6xl! flex flex-col p-0 md:p-6 gap-0"
    onOpenAutoFocus={(pEvent) => cDeviceStore.isMobile && pEvent.preventDefault()}
  >
    <Dialog.Header class="md:p-0 p-4" showCloseButton={false}>
      <Dialog.Title class="flex items-center gap-2">
        <Settings class="h-5 w-5" />

        <span>{fTranslate('text_74a883a037bc')}</span>
      </Dialog.Title>

      <!-- the body is flush with the dialog, so the corner close needs its own inset -->
      <Dialog.Close
        class="absolute top-0 right-0 rounded-xs opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-hidden max-md:top-1 max-md:right-1 {cPanelCloseMobileClass}"
      >
        <X class="size-4" />

        <span class="sr-only">{fTranslate('text_7d9eb7acb13e')}</span>
      </Dialog.Close>
    </Dialog.Header>

    <SettingsChat {initialSection} onClose={() => (open = false)} onSectionChange={() => {}} />
  </Dialog.Content>
</Dialog.fRoot>
