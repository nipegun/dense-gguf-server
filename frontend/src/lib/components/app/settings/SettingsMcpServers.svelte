<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Plus } from '@lucide/svelte';
  import { replaceState } from '$app/navigation';
  import { page } from '$app/state';
  import { McpServerCard, McpServerCardSkeleton } from '$lib/components/app';
  import { DialogMcpResourcesBrowser, DialogMcpServerAddNew } from '$lib/components/app/dialogs';
  import { Button } from '$lib/components/ui/button';
  import * as Empty from '$lib/components/ui/empty';
  import { HealthCheckStatus } from '$lib/enums';
  import { cMcpStore, cToolsStore } from '$lib/stores';
  import { onMount } from 'svelte';
  import { fade } from 'svelte/transition';

  interface Props {
    class?: string;
  }

  let { class: className }: Props = $props();

  let aServers = $derived(cMcpStore.mGetServers());

  let vIsAddingServer = $state(false);
  let vIsResourcesDialogOpen = $state(false);

  onMount(() => {
    if (page.url.searchParams.has('add')) {
      vIsAddingServer = true;

      const cNewUrl = new URL(page.url);

      cNewUrl.searchParams.delete('add');

      replaceState(cNewUrl, {});
    }
  });

  // Each card decides for itself whether to render based on its own
  // health-check state, so adding a server only flashes the new card
  // (not every other already-loaded card) until its health check resolves.
  // Disabled servers never receive a startup health check, so IDLE only
  // counts as pending when the server is enabled; otherwise the real card
  // renders and keeps the enable toggle reachable.
  function fIsServerPending(pServerId: string, pEnabled: boolean): boolean {
    const cStatus = cMcpStore.mGetHealthCheckState(pServerId).status;

    return (
      cStatus === HealthCheckStatus.cConnecting || (cStatus === HealthCheckStatus.cIdle && pEnabled)
    );
  }
</script>

<div in:fade={{ duration: 150 }} class="flex flex-col h-full">
  <DialogMcpServerAddNew bind:open={vIsAddingServer} />

  <DialogMcpResourcesBrowser bind:open={vIsResourcesDialogOpen} />

  {#if aServers.length === 0}
    <div class="flex flex-1 items-center justify-center pb-20 pt-10 my-auto">
      <Empty.Root class="max-w-md">
        <Empty.Header>
          <Empty.Media variant="icon">
            <Plus />
          </Empty.Media>

          <Empty.Title>{fTranslate('text_b2b642a46b33')}</Empty.Title>

          <Empty.Description>{fTranslate('text_341e549ff92d')}</Empty.Description>
        </Empty.Header>

        <Empty.Content>
          <Button onclick={() => (vIsAddingServer = true)} size="sm">
            <Plus />

            {fTranslate('text_554d104442c2')}
          </Button>
        </Empty.Content>
      </Empty.Root>
    </div>
  {:else}
    <div
      class="grid gap-4 {className}"
      style="grid-template-columns: repeat(auto-fill, minmax(min(25rem, calc(100dvw - 4rem)), 1fr));"
    >
      {#each aServers as vServer (vServer.id)}
        {#if fIsServerPending(vServer.id, vServer.enabled)}
          <McpServerCardSkeleton />
        {:else}
          <McpServerCard
            enabled={vServer.enabled}
            onBrowseResources={() => (vIsResourcesDialogOpen = true)}
            onDelete={() => cMcpStore.mRemoveServer(vServer.id)}
            onToggle={async () => {
              const cWasEnabled = vServer.enabled;

              cMcpStore.mUpdateServer(vServer.id, { enabled: !cWasEnabled });

              if (!cWasEnabled) {
                // Promote the connection so tools/prompts/resources become
                // available right away instead of waiting for the next chat-init.
                await cMcpStore.mRunHealthCheck(vServer, true);
                cToolsStore.mEnableAllToolsForServer(vServer.id);
              }
            }}
            onUpdate={(pUpdates) => cMcpStore.mUpdateServer(vServer.id, pUpdates)}
            server={vServer}
          />
        {/if}
      {/each}

      {#if !vIsAddingServer}
        <Empty.Root class="border">
          <Empty.Header>
            <Empty.Media variant="icon">
              <Plus />
            </Empty.Media>

            <Empty.Title>{fTranslate('text_857b19e08d14')}</Empty.Title>

            <Empty.Description>{fTranslate('text_341e549ff92d')}</Empty.Description>
          </Empty.Header>

          <Empty.Content>
            <Button onclick={() => (vIsAddingServer = true)} size="sm">
              <Plus />

              {fTranslate('text_554d104442c2')}
            </Button>
          </Empty.Content>
        </Empty.Root>
      {/if}
    </div>
  {/if}
</div>
