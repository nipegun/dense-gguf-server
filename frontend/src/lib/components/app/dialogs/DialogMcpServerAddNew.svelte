<script lang="ts">
  import { cUiSettingsStore } from '$lib/stores/ui-settings.svelte';
  import { fTranslate } from '$lib/i18n';
  import { SharedPreferencesService } from '$lib/services/shared-preferences.service';

  import { browser } from '$app/environment';
  import { McpServerCardCompact, McpServerForm } from '$lib/components/app/mcp';
  import { Button } from '$lib/components/ui/button';
  import * as Dialog from '$lib/components/ui/dialog';
  import {
    cDismissedRecommendedMcpServersLocalstorageKey,
    HEADERS,
    cMcpServerIdPrefix,
    aRecommendedMcpServers
  } from '$lib/constants';
  import { BooleanString, HealthCheckStatus } from '$lib/enums';
  import { cDeviceStore, cMcpStore } from '$lib/stores';
  import { fCanonicalizeServerUrl, fParseHeadersToArray, fUuid } from '$lib/utils';

  interface Props {
    open: boolean;
    onOpenChange?: (pOpen: boolean) => void;
  }

  let { onOpenChange, open = $bindable() }: Props = $props();

  let vNewServerUrl = $state('');
  let vNewServerName = $state('');
  let vNameAutoFilled = $state('');
  let vNameTouched = $state(false);

  let vPreviewRun = 0;

  function fHandleNameChange(pValue: string) {
    vNewServerName = pValue;
    vNameTouched = true;
  }
  let vNewServerHeaders = $state('');
  let vNewServerUseProxy = $state(false);

  let vNewServerWantsAuthorization = $state(false);

  let vSelectedRecommendationId = $derived.by(() => {
    const url = vNewServerUrl.trim();

    if (!url) return null;

    const cTargetCanonical = fCanonicalizeServerUrl(url);

    return (
      aRecommendedMcpServers.find((pRec) => fCanonicalizeServerUrl(pRec.url) === cTargetCanonical)
        ?.id ?? null
    );
  });
  let vSelectedRecommendation = $derived(
    vSelectedRecommendationId
      ? (aRecommendedMcpServers.find((pRec) => pRec.id === vSelectedRecommendationId) ?? null)
      : null
  );
  let vAuthRequired = $derived(vSelectedRecommendation?.needsAuthorization ?? false);

  let vBearerTokenFilled = $derived.by(() => {
    const aPairs = fParseHeadersToArray(vNewServerHeaders);
    const cBearerPrefix = HEADERS.BEARER.toLowerCase();
    const cBearer = aPairs.find(
      (pP) =>
        HEADERS.REDACTED.has(pP.key.trim().toLowerCase()) &&
        pP.value.trim().toLowerCase().startsWith(cBearerPrefix)
    );

    if (!cBearer) return false;

    return cBearer.value.trim().slice(cBearerPrefix.length).trim().length > 0;
  });

  let vNewServerUrlError = $derived.by(() => {
    if (!vNewServerUrl.trim()) return 'URL is required';

    try {
      new URL(vNewServerUrl);

      return null;
    } catch {
      return 'Invalid URL format';
    }
  });
  let vNewServerHeaderPairsValid = $derived(
    fParseHeadersToArray(vNewServerHeaders).every((pP) => pP.key.trim() && pP.value.trim())
  );
  let vCanSave = $derived(
    !vNewServerUrlError && vNewServerHeaderPairsValid && (!vAuthRequired || vBearerTokenFilled)
  );

  // The current shared settings document stores a boolean string.
  function fReadRecommendationsDismissed(): boolean {
    if (!browser) return false;

    const cRaw = SharedPreferencesService.mGetItem(cDismissedRecommendedMcpServersLocalstorageKey);

    if (!cRaw) return false;

    return cRaw === BooleanString.cTrue;
  }

  function fWriteRecommendationsDismissed(pDismissed: boolean) {
    vRecommendationsDismissed = pDismissed;

    if (browser) {
      SharedPreferencesService.mSetItem(
        cDismissedRecommendedMcpServersLocalstorageKey,
        pDismissed ? BooleanString.cTrue : BooleanString.cFalse
      );
    }
  }

  let vRecommendationsDismissed = $state<boolean>(fReadRecommendationsDismissed());

  $effect(() => {
    if (cUiSettingsStore.vIsLoaded) vRecommendationsDismissed = fReadRecommendationsDismissed();
  });

  // Read-only once a recommendation is picked: switch is disabled, so we keep
  // the Authorization field in sync with the requirement.
  $effect(() => {
    if (vAuthRequired) {
      vNewServerWantsAuthorization = true;
    }
  });

  // Debounced preview handshake: once the URL is valid and stable, fetch the
  // server-reported name to prefill the display name field. A manual edit
  // freezes the autofill for good, and failures stay silent.
  $effect(() => {
    const url = vNewServerUrl.trim();
    const headers = vNewServerHeaders.trim();
    const cUseProxy = vNewServerUseProxy;

    if (!open || vNewServerUrlError || !url) return;

    const cRun = ++vPreviewRun;
    // One throwaway id per run: concurrent previews (URL typed, then the
    // bearer token pasted) would poison each other's shared health state.
    const cPreviewId = `${cMcpServerIdPrefix}-preview-${cRun}`;
    const cTimer = setTimeout(async () => {
      await cMcpStore.mRunHealthCheck({
        enabled: false,
        headers: headers || undefined,
        id: cPreviewId,
        url,
        useProxy: cUseProxy
      });

      const cState = cMcpStore.mGetHealthCheckState(cPreviewId);

      cMcpStore.mClearHealthCheck(cPreviewId);

      if (cRun !== vPreviewRun) return;

      if (cState.status !== HealthCheckStatus.SUCCESS) return;

      const cAutoName = cState.serverInfo?.title || cState.serverInfo?.name || '';

      if (cAutoName && !vNameTouched) {
        vNewServerName = cAutoName;
        vNameAutoFilled = cAutoName;
      }
    }, 600);

    return () => clearTimeout(cTimer);
  });

  let vHasSelection = $derived(vSelectedRecommendationId !== null);

  let aUnconfiguredRecommendations = $derived.by(() => {
    const sConfiguredCanonicals = new Set(
      cMcpStore.mGetServers().map((pS) => fCanonicalizeServerUrl(pS.url))
    );

    return aRecommendedMcpServers.filter(
      (pRec) => !sConfiguredCanonicals.has(fCanonicalizeServerUrl(pRec.url))
    );
  });

  let aRecommendationsToShow = $derived(
    vRecommendationsDismissed ? [] : aUnconfiguredRecommendations
  );

  function fHandleRecommendationClick(pRecommendedId: string) {
    const cRecommendation = aRecommendedMcpServers.find((pRec) => pRec.id === pRecommendedId);

    if (!cRecommendation) return;

    vNewServerUrl = cRecommendation.url;
    vNewServerHeaders = '';
    vNewServerWantsAuthorization = cRecommendation.needsAuthorization ?? false;
  }

  function fHandleDismissAll() {
    fWriteRecommendationsDismissed(true);
  }

  function fHandleOpenChange(pValue: boolean) {
    if (!pValue) {
      vNewServerUrl = '';
      vNewServerName = '';
      vNameAutoFilled = '';
      vNameTouched = false;
      vPreviewRun++;
      vNewServerHeaders = '';
      vNewServerUseProxy = false;
      vNewServerWantsAuthorization = false;
    }

    open = pValue;
    onOpenChange?.(pValue);
  }

  function fSaveNewServer() {
    if (!vCanSave) return;

    const cNewServerId = fUuid() ?? `${cMcpServerIdPrefix}-${Date.now()}`;

    cMcpStore.mAddServer({
      // A name equal to the autofilled server-reported one is not a
      // customization: keep following the automatic label.
      displayName:
        vNewServerName.trim() && vNewServerName.trim() !== vNameAutoFilled.trim()
          ? vNewServerName.trim()
          : undefined,
      enabled: true,
      headers: vNewServerHeaders.trim() || undefined,
      id: cNewServerId,
      url: vNewServerUrl.trim(),
      useProxy: vNewServerUseProxy
    });

    fHandleOpenChange(false);
  }

  function fHandleSubmit(pEvent: SubmitEvent) {
    pEvent.preventDefault();
    fSaveNewServer();
  }
</script>

<Dialog.fRoot onOpenChange={fHandleOpenChange} {open}>
  <Dialog.Content
    class="max-w-2xl! p-4"
    onOpenAutoFocus={(pEvent) => cDeviceStore.isMobile && pEvent.preventDefault()}
  >
    <Dialog.Header>
      <Dialog.Title class="select-none">{fTranslate('text_143882ba828c')}</Dialog.Title>
    </Dialog.Header>

    {#if aRecommendationsToShow.length > 0}
      <div class="space-y-3 pt-2">
        <div class="flex items-center justify-between gap-3">
          <h3 class="text-sm font-medium">{fTranslate('text_17bdcbc59576')}</h3>

          <Button
            class="text-muted-foreground"
            onclick={fHandleDismissAll}
            size="sm"
            variant="ghost">{fTranslate('text_48845bff334a')}</Button
          >
        </div>

        <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {#each aRecommendationsToShow as vRecommendation (vRecommendation.id)}
            <McpServerCardCompact
              dimmed={vHasSelection && vSelectedRecommendationId !== vRecommendation.id}
              onClick={() => fHandleRecommendationClick(vRecommendation.id)}
              selected={vSelectedRecommendationId === vRecommendation.id}
              server={vRecommendation}
            />
          {/each}
        </div>
      </div>
    {/if}

    <form class="contents" onsubmit={fHandleSubmit}>
      <div class="space-y-4 py-4">
        <McpServerForm
          bind:wantsAuthorization={vNewServerWantsAuthorization}
          headers={vNewServerHeaders}
          id="new-server"
          name={vNewServerName}
          onHeadersChange={(pV) => (vNewServerHeaders = pV)}
          onNameChange={fHandleNameChange}
          onUrlChange={(pV) => (vNewServerUrl = pV)}
          onUseProxyChange={(pV) => (vNewServerUseProxy = pV)}
          required={vAuthRequired}
          url={vNewServerUrl}
          urlError={vNewServerUrl ? vNewServerUrlError : null}
          useProxy={vNewServerUseProxy}
        />
      </div>

      <Dialog.Footer>
        <Button onclick={() => fHandleOpenChange(false)} size="sm" variant="secondary">
          {fTranslate('text_19766ed6ccb2')}
        </Button>

        <Button
          aria-label={fTranslate('text_1509f561f241')}
          disabled={!vCanSave}
          size="sm"
          type="submit"
          variant="default"
        >
          {fTranslate('text_9fd728c66c9a')}
        </Button>
      </Dialog.Footer>
    </form>
  </Dialog.Content>
</Dialog.fRoot>
