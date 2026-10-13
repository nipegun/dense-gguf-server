<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { McpServerForm } from '$lib/components/app/mcp';
  import { Button } from '$lib/components/ui/button';
  import { fParseHeadersToArray } from '$lib/utils';

  interface Props {
    serverId: string;
    serverUrl: string;
    serverUseProxy?: boolean;
    /** Current automatic label, prefilled so the user can customize it. */
    serverLabel?: string;
    onSave: (pUrl: string, pHeaders: string, pUseProxy: boolean, pName?: string) => void;
    onCancel: () => void;
  }

  let {
    onCancel,
    onSave,
    serverId,
    serverLabel = '',
    serverUrl,
    serverUseProxy = false
  }: Props = $props();

  let vEditUrl = $derived(serverUrl);
  let vEditName = $derived(serverLabel);
  let vEditHeaders = $state('');
  let vEditUseProxy = $derived(serverUseProxy);

  let vUrlError = $derived.by(() => {
    if (!vEditUrl.trim()) return 'URL is required';

    try {
      new URL(vEditUrl);

      return null;
    } catch {
      return 'Invalid URL format';
    }
  });

  let vHeaderPairsValid = $derived(
    fParseHeadersToArray(vEditHeaders).every((pP) => pP.key.trim() && pP.value.trim())
  );
  let vCanSave = $derived(!vUrlError && vHeaderPairsValid);

  function fHandleSave() {
    if (!vCanSave) return;

    // An unchanged prefill keeps following the automatic label; only an
    // actual edit becomes a persisted custom display name.
    const name = vEditName.trim() !== serverLabel.trim() ? vEditName.trim() : undefined;

    onSave(vEditUrl.trim(), vEditHeaders.trim(), vEditUseProxy, name);
  }

  function fHandleSubmit(pEvent: SubmitEvent) {
    pEvent.preventDefault();
    fHandleSave();
  }

  export function setInitialValues(pUrl: string, pHeaders: string, pUseProxy: boolean, pName = '') {
    vEditUrl = pUrl;
    vEditHeaders = pHeaders;
    vEditUseProxy = pUseProxy;
    vEditName = pName;
  }
</script>

<form class="contents" onsubmit={fHandleSubmit}>
  <div class="space-y-4">
    <p class="font-medium">{fTranslate('text_b86b2fbec250')}</p>

    <McpServerForm
      headers={vEditHeaders}
      id={serverId}
      name={vEditName}
      onHeadersChange={(pV) => (vEditHeaders = pV)}
      onNameChange={(pV) => (vEditName = pV)}
      onUrlChange={(pV) => (vEditUrl = pV)}
      onUseProxyChange={(pV) => (vEditUseProxy = pV)}
      url={vEditUrl}
      urlError={vEditUrl ? vUrlError : null}
      useProxy={vEditUseProxy}
    />

    <div class="flex items-center justify-end gap-2">
      <Button onclick={onCancel} size="sm" variant="secondary"
        >{fTranslate('text_19766ed6ccb2')}</Button
      >

      <Button disabled={!vCanSave} size="sm" type="submit">
        {serverUrl.trim() ? 'Update' : 'Add'}
      </Button>
    </div>
  </div>
</form>
