<script lang="ts">
  import { AlertTriangle, Loader2, RefreshCw } from '@lucide/svelte';
  import * as Alert from '$lib/components/ui/alert';
  import { cIconClassDefault } from '$lib/constants';
  import { cServerStore } from '$lib/stores';

  let vHasError = $derived(!!cServerStore.error);
  let vIsLoadingModel = $derived(cServerStore.status === 503);
</script>

{#if vHasError || vIsLoadingModel}
  <div class="pointer-events-auto mx-auto mb-4 max-w-[48rem] px-1">
    <Alert.Root variant={vIsLoadingModel ? 'default' : 'destructive'}>
      {#if vIsLoadingModel}
        <Loader2 class="{cIconClassDefault} animate-spin" />
      {:else}
        <AlertTriangle class={cIconClassDefault} />
      {/if}

      <Alert.Title class="flex items-center justify-between">
        <span>{vIsLoadingModel ? 'Loading model' : 'Server unavailable'}</span>

        {#if !vIsLoadingModel}
          <button
            class="flex items-center gap-1.5 rounded-lg bg-destructive/20 px-2 py-1 text-xs font-medium hover:bg-destructive/30 disabled:opacity-50"
            disabled={cServerStore.loading}
            onclick={() => cServerStore.mFetch()}
          >
            <RefreshCw class="h-3 w-3 {cServerStore.loading ? 'animate-spin' : ''}" />
            {cServerStore.loading ? 'Retrying...' : 'Retry'}
          </button>
        {/if}
      </Alert.Title>

      {#if !vIsLoadingModel}
        <Alert.Description>{cServerStore.error}</Alert.Description>
      {/if}
    </Alert.Root>
  </div>
{/if}
