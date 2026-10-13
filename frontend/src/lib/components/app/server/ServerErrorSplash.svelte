<script lang="ts">
  import { fApiUrl } from '$lib/api-url';

  import { fTranslate } from '$lib/i18n';
  import { AlertTriangle, CheckCircle, Key, RefreshCw, XCircle } from '@lucide/svelte';
  import { goto } from '$app/navigation';
  import { base } from '$app/paths';
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import Label from '$lib/components/ui/label/label.svelte';
  import { HEADERS, cIconClassDefault, cRoutes, cSettingsKeys } from '$lib/constants';
  import { KeyboardKey } from '$lib/enums';
  import { cServerStore, cSettingsStore } from '$lib/stores';
  import { fade, fly, scale } from 'svelte/transition';

  interface Props {
    class?: string;
    error: string;
    onRetry?: () => void;
    showRetry?: boolean;
    showTroubleshooting?: boolean;
  }

  let {
    class: className = '',
    error,
    onRetry,
    showRetry = true,
    showTroubleshooting = false
  }: Props = $props();

  let vIsServerLoading = $derived(cServerStore.loading);
  let vIsAccessDeniedError = $derived(
    error.toLowerCase().includes('access denied') ||
      error.toLowerCase().includes('invalid api key') ||
      error.toLowerCase().includes('unauthorized') ||
      error.toLowerCase().includes('401') ||
      error.toLowerCase().includes('403')
  );

  let vApiKeyInput = $state('');
  let vShowApiKeyInput = $state(false);
  let vApiKeyState = $state<'idle' | 'validating' | 'success' | 'error'>('idle');
  let vApiKeyError = $state('');

  function fHandleRetryConnection() {
    if (onRetry) {
      onRetry();
    } else {
      cServerStore.mFetch();
    }
  }

  function fHandleShowApiKeyInput() {
    vShowApiKeyInput = true;
    // Pre-fill with current API key if it exists
    const cCurrentConfig = cSettingsStore.vConfig;

    vApiKeyInput = cCurrentConfig.apiKey?.toString() || '';
  }

  async function fHandleSaveApiKey() {
    if (!vApiKeyInput.trim()) return;

    vApiKeyState = 'validating';
    vApiKeyError = '';

    try {
      // Update the API key in settings first
      cSettingsStore.mUpdateConfig(cSettingsKeys.API_KEY, vApiKeyInput.trim());

      // Test the API key by making a real request to the server
      const cResponse = await fetch(fApiUrl('/props'), {
        headers: {
          'Content-Type': 'application/json',
          [HEADERS.AUTHORIZATION]: `${HEADERS.BEARER}${vApiKeyInput.trim()}`
        }
      });

      if (cResponse.ok) {
        // API key is valid - User Story B
        vApiKeyState = 'success';

        // Show success state briefly, then navigate to home
        setTimeout(() => {
          goto(cRoutes.START);
        }, 1000);
      } else {
        // API key is invalid - User Story A
        vApiKeyState = 'error';

        if (cResponse.status === 401 || cResponse.status === 403) {
          vApiKeyError = 'Invalid API key - please check and try again';
        } else {
          vApiKeyError = `Authentication failed (${cResponse.status})`;
        }

        // Reset to idle state after showing error (don't reload UI)
        setTimeout(() => {
          vApiKeyState = 'idle';
        }, 3000);
      }
    } catch (error) {
      // Network or other errors - User Story A
      vApiKeyState = 'error';

      if (error instanceof Error) {
        if (error.message.includes('fetch')) {
          vApiKeyError = 'Cannot connect to server - check if server is running';
        } else {
          vApiKeyError = error.message;
        }
      } else {
        vApiKeyError = 'Connection error - please try again';
      }

      // Reset to idle state after showing error (don't reload UI)
      setTimeout(() => {
        vApiKeyState = 'idle';
      }, 3000);
    }
  }

  function fHandleApiKeyKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === KeyboardKey.cEnter) {
      fHandleSaveApiKey();
    }
  }
</script>

<div class="flex h-full items-center justify-center {className}">
  <div class="w-full max-w-md px-4 text-center">
    <div in:fade={{ duration: 300 }} class="mb-6">
      <div
        class="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10"
      >
        <AlertTriangle class="h-8 w-8 text-destructive" />
      </div>

      <h2 class="mb-2 text-xl font-semibold">{fTranslate('text_8d9809100ff0')}</h2>

      <p class="mb-4 text-sm text-muted-foreground">
        {error}
      </p>
    </div>

    {#if vIsAccessDeniedError && !vShowApiKeyInput}
      <div in:fly={{ delay: 200, duration: 300, y: 10 }} class="mb-4">
        <Button class="w-full" onclick={fHandleShowApiKeyInput} variant="outline">
          <Key class={cIconClassDefault} />
          {fTranslate('text_895f77e86e47')}
        </Button>
      </div>
    {/if}

    {#if vShowApiKeyInput}
      <div in:fly={{ delay: 200, duration: 300, y: 10 }} class="mb-4 space-y-3 text-left">
        <div class="space-y-2">
          <Label class="text-sm font-medium" for="api-key-input"
            >{fTranslate('text_23189d55f697')}</Label
          >

          <div class="relative">
            <Input
              bind:value={vApiKeyInput}
              autocomplete="new-password"
              class="w-full pr-10 {vApiKeyState === 'error'
                ? 'border-destructive'
                : vApiKeyState === 'success'
                  ? 'border-green-500'
                  : ''}"
              disabled={vApiKeyState === 'validating'}
              id="api-key-input"
              onkeydown={fHandleApiKeyKeydown}
              placeholder={fTranslate('text_cfc419f196c7')}
              type="password"
            />

            {#if vApiKeyState === 'validating'}
              <div class="absolute top-1/2 right-3 -translate-y-1/2">
                <RefreshCw class="{cIconClassDefault} animate-spin text-muted-foreground" />
              </div>
            {:else if vApiKeyState === 'success'}
              <div
                in:scale={{ duration: 200, start: 0.8 }}
                class="absolute top-1/2 right-3 -translate-y-1/2"
              >
                <CheckCircle class="{cIconClassDefault} text-green-500" />
              </div>
            {:else if vApiKeyState === 'error'}
              <div
                in:scale={{ duration: 200, start: 0.8 }}
                class="absolute top-1/2 right-3 -translate-y-1/2"
              >
                <XCircle class="{cIconClassDefault} text-destructive" />
              </div>
            {/if}
          </div>

          {#if vApiKeyError}
            <p in:fly={{ duration: 200, y: -10 }} class="text-sm text-destructive">
              {vApiKeyError}
            </p>
          {/if}

          {#if vApiKeyState === 'success'}
            <p in:fly={{ duration: 200, y: -10 }} class="text-sm text-green-600">
              {fTranslate('text_e40ab11d04df')}
            </p>
          {/if}
        </div>

        <div class="flex gap-2">
          <Button
            class="flex-1"
            disabled={!vApiKeyInput.trim() ||
              vApiKeyState === 'validating' ||
              vApiKeyState === 'success'}
            onclick={fHandleSaveApiKey}
          >
            {#if vApiKeyState === 'validating'}
              <RefreshCw class="{cIconClassDefault} animate-spin" />
              {fTranslate('text_b28b48d59a08')}{:else if vApiKeyState === 'success'}{fTranslate(
                'text_662efaf46c61'
              )}{:else}{fTranslate('text_49673e62edaf')}{/if}
          </Button>

          <Button
            class="flex-1"
            disabled={vApiKeyState === 'validating'}
            onclick={() => {
              vShowApiKeyInput = false;
              vApiKeyState = 'idle';
              vApiKeyError = '';
            }}
            variant="outline"
          >
            {fTranslate('text_19766ed6ccb2')}
          </Button>
        </div>
      </div>
    {/if}

    {#if showRetry}
      <div in:fly={{ delay: 200, duration: 300, y: 10 }}>
        <Button class="w-full" disabled={vIsServerLoading} onclick={fHandleRetryConnection}>
          {#if vIsServerLoading}
            <RefreshCw class="{cIconClassDefault} animate-spin" />

            {fTranslate('text_5f04ae9ed6a8')}{:else}
            <RefreshCw class={cIconClassDefault} />

            {fTranslate('text_35f09f1991fb')}{/if}
        </Button>
      </div>
    {/if}

    {#if showTroubleshooting}
      <div in:fly={{ delay: 400, duration: 300, y: 10 }} class="mt-4 text-left">
        <details class="text-sm">
          <summary class="cursor-pointer text-muted-foreground hover:text-foreground">
            {fTranslate('text_c3af076f92c5')}
          </summary>

          <div class="mt-2 space-y-3 text-xs text-muted-foreground">
            <div class="space-y-2">
              <p class="mb-4 font-medium">{fTranslate('dgs_b66a145e5f45')}</p>

              <div class="rounded bg-muted/50 px-2 py-1 font-mono text-xs">
                <p>{fTranslate('dgs_2905c3bc9574')}</p>
              </div>

              <p>{fTranslate('text_7175517a370b')}</p>

              <div class="rounded bg-muted/50 px-2 py-1 font-mono text-xs">
                <p class="mt-1">{fTranslate('dgs_144c9e16be70')}</p>
              </div>
            </div>

            <ul class="list-disc space-y-1 pl-4">
              <li>{fTranslate('text_4d326da4f4c4')}</li>

              <li>{fTranslate('text_f5cb1a5f2421')}</li>

              <li>{fTranslate('text_7b3d3f648257')}</li>
            </ul>
          </div>
        </details>
      </div>
    {/if}
  </div>
</div>
