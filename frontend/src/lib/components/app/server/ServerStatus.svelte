<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { AlertTriangle, Server } from '@lucide/svelte';
  import { Badge } from '$lib/components/ui/badge';
  import { Button } from '$lib/components/ui/button';
  import { cIconClassDefault } from '$lib/constants';
  import { cModelsStore, cServerStore } from '$lib/stores';

  interface Props {
    class?: string;
    showActions?: boolean;
  }

  let { class: className = '', showActions = false }: Props = $props();

  let error = $derived(cServerStore.error);
  let loading = $derived(cServerStore.loading);
  let vModel = $derived(cModelsStore.singleModelName);
  let vServerData = $derived(cServerStore.props);

  function fGetStatusColor() {
    if (loading) return 'bg-yellow-500';

    if (error) return 'bg-red-500';

    if (vServerData) return 'bg-green-500';

    return 'bg-gray-500';
  }

  function fGetStatusText() {
    if (loading) return 'Connecting...';

    if (error) return 'Connection Error';

    if (vServerData) return 'Connected';

    return 'Unknown';
  }
</script>

<div class="flex items-center space-x-3 {className}">
  <div class="flex items-center space-x-2">
    <div class="h-2 w-2 rounded-full {fGetStatusColor()}"></div>

    <span class="text-sm text-muted-foreground">{fGetStatusText()}</span>
  </div>

  {#if vServerData && !error}
    <Badge class="text-xs" variant="outline">
      <Server class="mr-1 h-3 w-3" />

      {vModel || 'Unknown Model'}
    </Badge>

    {#if vServerData?.default_generation_settings?.n_ctx}
      <Badge class="text-xs" variant="secondary">
        {fTranslate('text_051765386b25')}
        {vServerData.default_generation_settings.n_ctx.toLocaleString()}
      </Badge>
    {/if}
  {/if}

  {#if showActions && error}
    <Button class="text-destructive" size="sm" variant="outline">
      <AlertTriangle class={cIconClassDefault} />

      {error}
    </Button>
  {/if}
</div>
