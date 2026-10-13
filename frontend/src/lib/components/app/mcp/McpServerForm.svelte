<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { KeyValuePairs } from '$lib/components/app';
  import { Input } from '$lib/components/ui/input';
  import { Switch } from '$lib/components/ui/switch';
  import { cCliFlags, HEADERS, cMcpServerUrlPlaceholder } from '$lib/constants';
  import { UrlProtocol } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';
  import type { KeyValuePair } from '$lib/types';
  import { fParseHeadersToArray, fSerializeHeaders } from '$lib/utils';

  interface Props {
    url: string;
    headers: string;
    name?: string;
    onNameChange?: (pName: string) => void;
    /** Shown in the empty display name field, e.g. the current automatic label. */
    namePlaceholder?: string;
    useProxy?: boolean;
    onUrlChange: (pUrl: string) => void;
    onHeadersChange: (pHeaders: string) => void;
    onUseProxyChange?: (pUseProxy: boolean) => void;
    urlError?: string | null;
    id?: string;
    /**
     * "Wants Authorization" is the user's *intent* to add a Bearer token
     * (separate from `hasAuthorization` which reflects what's already in
     * the headers). Bindable so a parent - e.g. the recommendation cards
     * on the "Add New Server" dialog - can flip the switch on when the
     * picked server ships a `needsAuthorization: true` flag.
     */
    wantsAuthorization?: boolean;
    /**
     * Marks the "Authorization" field as required. Locks the toggle so the
     * user can't dismiss it, and visually marks the field with a red
     * asterisk. The parent is expected to gate its submit affordance on
     * the bearer token actually being filled. Used by the "Add New Server"
     * dialog for recommendations whose `needsAuthorization` flag is true.
     */
    required?: boolean;
  }

  let {
    headers,
    id = 'server',
    name = '',
    namePlaceholder = 'Name reported by the server',
    onHeadersChange,
    onNameChange,
    onUrlChange,
    onUseProxyChange,
    required = false,
    url,
    urlError = null,
    useProxy = false,
    wantsAuthorization = $bindable(false)
  }: Props = $props();

  let vIsWebSocket = $derived(
    url.toLowerCase().startsWith(UrlProtocol.cWebsocket) ||
      url.toLowerCase().startsWith(UrlProtocol.cWebsocketSecure)
  );

  let aHeaderPairs = $derived<KeyValuePair[]>(fParseHeadersToArray(headers));

  // Heuristic: this dedicated UI only owns Authorization headers that already
  // carry a Bearer scheme. Anything else (e.g. Basic, raw tokens) stays in the
  // KV section so the user can still edit those values verbatim.
  const fMatchesAuthorizationKey = (pKey: string): boolean =>
    HEADERS.REDACTED.has(pKey.trim().toLowerCase());

  const fIsBearerScheme = (pValue: string): boolean =>
    pValue.trim().toLowerCase().startsWith(HEADERS.BEARER.toLowerCase());

  const fOwnedByBearerUi = (pP: KeyValuePair): boolean =>
    fMatchesAuthorizationKey(pP.key) && fIsBearerScheme(pP.value);

  let vHasAuthorization = $derived(aHeaderPairs.some(fOwnedByBearerUi));

  let vShowAuthorization = $derived(vHasAuthorization || wantsAuthorization);

  let vUrlInput: HTMLInputElement | null = $state(null);
  let vBearerInput: HTMLInputElement | null = $state(null);

  $effect(() => {
    vUrlInput?.focus();
  });

  $effect(() => {
    if (wantsAuthorization && vBearerInput) {
      vBearerInput.focus();
    }
  });

  let vBearerToken = $derived.by(() => {
    const cAuth = aHeaderPairs.find(fOwnedByBearerUi);

    if (!cAuth) return '';

    return cAuth.value.trim().slice(HEADERS.BEARER.length).trim();
  });

  $effect(() => {
    if (!headers.trim()) {
      wantsAuthorization = false;
    }
  });

  function fUpdateHeaderPairs(pNewPairs: KeyValuePair[]) {
    aHeaderPairs = pNewPairs;
    onHeadersChange(fSerializeHeaders(pNewPairs));
  }

  // The dedicated UI owns the Authorization slot end-to-end when the user
  // engages it: any prior Authorization row (Bearer or otherwise) is replaced
  // by exactly one { Authorization: "Bearer <token>" } entry. JSON's last-key
  // behavior would otherwise pick one arbitrarily, so we strip first.
  function fUpdateBearerToken(pToken: string) {
    const aFiltered = aHeaderPairs.filter((pP) => !fMatchesAuthorizationKey(pP.key));
    const cTrimmed = pToken.trim();

    if (cTrimmed) {
      aFiltered.push({ key: HEADERS.AUTHORIZATION, value: `${HEADERS.BEARER}${cTrimmed}` });
    }

    fUpdateHeaderPairs(aFiltered);
  }

  function fSetUseAuthorization(pChecked: boolean) {
    wantsAuthorization = pChecked;

    if (!pChecked) {
      // Only drop the entry this UI owns; a non-Bearer Authorization row
      // authored in the KV section must survive a toggle off untouched.
      const aFiltered = aHeaderPairs.filter((pP) => !fOwnedByBearerUi(pP));

      fUpdateHeaderPairs(aFiltered);
    }
  }
</script>

<div class="grid gap-2">
  <div class="mb-4">
    <label class="mb-2 block text-xs font-medium select-none" for="server-url-{id}">
      {fTranslate('text_22f5ebc3ad3f')} <span class="text-destructive">*</span>
    </label>

    <Input
      bind:ref={vUrlInput}
      class={urlError ? 'border-destructive' : ''}
      id="server-url-{id}"
      oninput={(pE) => onUrlChange(pE.currentTarget.value)}
      placeholder={cMcpServerUrlPlaceholder}
      type="url"
      value={url}
    />

    {#if urlError}
      <p class="mt-1.5 text-xs text-destructive">{urlError}</p>
    {/if}
  </div>

  <div class="mb-4">
    <label class="mb-2 block text-xs font-medium select-none" for="server-name-{id}">
      {fTranslate('text_2b7f6a84de91')}
    </label>

    <Input
      id="server-name-{id}"
      oninput={(pE) => onNameChange?.(pE.currentTarget.value)}
      placeholder={namePlaceholder}
      type="text"
      value={name}
    />
  </div>

  <label class="flex items-center gap-2 cursor-pointer select-none">
    <Switch
      checked={vShowAuthorization}
      disabled={required}
      id="use-authorization-{id}"
      onCheckedChange={fSetUseAuthorization}
    />

    <span class="text-xs text-muted-foreground">
      {fTranslate('text_ca5839e38a15')}{#if required}
        <span class="text-destructive">*</span>{/if}
    </span>
  </label>

  {#if vShowAuthorization}
    <div class="relative mt-2">
      <Input
        bind:ref={vBearerInput}
        autocomplete="off"
        class="pl-16"
        id="bearer-token-{id}"
        oninput={(pE) => fUpdateBearerToken(pE.currentTarget.value)}
        placeholder={fTranslate('text_4a1c9cb993d2')}
        type="password"
        value={vBearerToken}
      />

      <span
        class="pointer-events-none absolute inset-y-0 left-3 flex items-center text-sm font-medium text-foreground"
      >
        {fTranslate('text_710e0dbdd422')}
      </span>
    </div>
  {/if}

  <KeyValuePairs
    addButtonLabel="Add"
    class="mt-3"
    emptyMessage="No custom headers configured."
    keyPlaceholder="Header name"
    onPairsChange={(pPairs) => {
      const cAuth = aHeaderPairs.find(fOwnedByBearerUi);

      fUpdateHeaderPairs(cAuth ? [...pPairs, cAuth] : pPairs);
    }}
    pairs={aHeaderPairs.filter((pP) => !fOwnedByBearerUi(pP))}
    sectionLabel="Custom Headers"
    sectionLabelOptional
    valuePlaceholder="Value"
  />

  {#if !vIsWebSocket && onUseProxyChange}
    <label
      class={[
        'mt-3 flex items-start gap-2',
        cMcpStore.isProxyAvailable && 'cursor-pointer',
        !cMcpStore.isProxyAvailable && 'opacity-80'
      ]}
    >
      <Switch
        checked={useProxy}
        class="mt-1"
        disabled={!cMcpStore.isProxyAvailable}
        id="use-proxy-{id}"
        onCheckedChange={(pChecked) => onUseProxyChange?.(pChecked)}
      />

      <span>
        <span class="text-xs text-muted-foreground">{fTranslate('dgs_088f58211e4c')}</span>

        <br />

        {#if !cMcpStore.isProxyAvailable}
          <span class="inline-flex gap-0.75 text-xs text-muted-foreground/60"
            >{fTranslate('text_2f7f3e22ea61')}
            <pre>dgs-server</pre>
            {fTranslate('text_0695b563acde')}
            <pre>{cCliFlags.MCP_PROXY}</pre>
            {fTranslate('text_f9921b14d171')}</span
          >
        {/if}
      </span>
    </label>
  {/if}
</div>
