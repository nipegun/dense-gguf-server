<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Pencil } from '@lucide/svelte';
  import * as AlertDialog from '$lib/components/ui/alert-dialog';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';

  interface Props {
    open: boolean;
    currentTitle: string;
    value: string;
    onConfirm: () => void;
    onCancel: () => void;
  }

  let {
    currentTitle,
    onCancel,
    onConfirm,
    open = $bindable(),
    value = $bindable('')
  }: Props = $props();

  let vInputRef = $state<HTMLInputElement | null>(null);

  const cCanSubmit = $derived(value.trim().length > 0 && value.trim() !== currentTitle.trim());

  $effect(() => {
    if (open) {
      value = currentTitle;
      queueMicrotask(() => {
        vInputRef?.focus();
        vInputRef?.select();
      });
    }
  });

  function fHandleOpenChange(pNewOpen: boolean) {
    if (!pNewOpen) {
      onCancel();
    }
  }

  function fHandleSubmit(pEvent: Event) {
    pEvent.preventDefault();

    if (!cCanSubmit) return;

    value = value.trim();
    onConfirm();
  }
</script>

<AlertDialog.fRoot bind:open onOpenChange={fHandleOpenChange}>
  <AlertDialog.Content>
    <AlertDialog.Header>
      <AlertDialog.Title class="flex items-center gap-2">
        <Pencil class="h-5 w-5" />
        {fTranslate('text_4f10b6dc1d44')}
      </AlertDialog.Title>

      <AlertDialog.Description>{fTranslate('text_715430ab8c99')}</AlertDialog.Description>
    </AlertDialog.Header>

    <form class="space-y-2 pt-2 pb-4" onsubmit={fHandleSubmit}>
      <label class="text-sm font-medium text-muted-foreground" for="conversation-rename-input">
        {fTranslate('text_5b5cd3e853ae')}
      </label>

      <Input
        bind:ref={vInputRef}
        bind:value
        autocomplete="off"
        autocorrect="off"
        id="conversation-rename-input"
        maxlength={200}
        placeholder={fTranslate('text_5b5cd3e853ae')}
        spellcheck={false}
      />
    </form>

    <AlertDialog.Footer>
      <AlertDialog.Cancel>{fTranslate('text_19766ed6ccb2')}</AlertDialog.Cancel>

      <Button disabled={!cCanSubmit} onclick={fHandleSubmit} type="button"
        >{fTranslate('text_1509f561f241')}</Button
      >
    </AlertDialog.Footer>
  </AlertDialog.Content>
</AlertDialog.fRoot>
