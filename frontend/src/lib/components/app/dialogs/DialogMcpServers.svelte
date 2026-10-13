<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { McpLogo } from '$lib/components/app';
  import { SettingsMcpServers } from '$lib/components/app/settings';
  import * as Dialog from '$lib/components/ui/dialog';
  import { cDeviceStore } from '$lib/stores';

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
    class="max-md:h-[100dvh]! max-md:w-screen! max-md:max-w-none! max-md:rounded-none! max-md:border-0! md:h-[calc(100vh-4rem)]! md:max-h-240! md:w-[calc(100vw-4rem)]! md:max-w-360! flex flex-col p-4"
    onOpenAutoFocus={(pEvent) => cDeviceStore.isMobile && pEvent.preventDefault()}
  >
    <Dialog.Header>
      <Dialog.Title class="flex items-center gap-2">
        <McpLogo class="h-5 w-5" />

        <span>{fTranslate('text_aab669a72882')}</span>
      </Dialog.Title>
    </Dialog.Header>

    <SettingsMcpServers class="mt-4" />
  </Dialog.Content>
</Dialog.fRoot>
