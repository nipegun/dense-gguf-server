<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import {
    McpConnectionLogs,
    McpServerCardActions,
    McpServerCardDeleteDialog,
    McpServerCardEditForm,
    McpServerCardHeader,
    McpServerCardToolsList,
    McpServerInfo
  } from '$lib/components/app/mcp';
  import * as Card from '$lib/components/ui/card';
  import { Skeleton } from '$lib/components/ui/skeleton';
  import { cIconClassDefault } from '$lib/constants';
  import { HealthCheckStatus } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';
  import type { HealthCheckState, MCPServerSettingsEntry } from '$lib/types';
  import { tick } from 'svelte';

  interface Props {
    server: MCPServerSettingsEntry;
    enabled?: boolean;
    onToggle: (pEnabled: boolean) => void;
    onUpdate: (pUpdates: Partial<MCPServerSettingsEntry>) => void;
    onDelete: () => void;
    onBrowseResources?: () => void;
  }

  let { enabled, onBrowseResources, onDelete, onToggle, onUpdate, server }: Props = $props();

  let vHealthState = $derived<HealthCheckState>(cMcpStore.mGetHealthCheckState(server.id));
  let vDisplayName = $derived(cMcpStore.mGetServerLabel(server));
  let vFaviconUrl = $derived(cMcpStore.mGetServerFavicon(server.id));
  let vIsIdle = $derived(vHealthState.status === HealthCheckStatus.cIdle);
  let vIsHealthChecking = $derived(vHealthState.status === HealthCheckStatus.cConnecting);
  let vIsConnected = $derived(vHealthState.status === HealthCheckStatus.SUCCESS);
  let vIsError = $derived(vHealthState.status === HealthCheckStatus.ERROR);
  // Disabled servers stay IDLE (no startup health check), so the body
  // skeleton only applies while a check is running or expected to run.
  let vShowSkeleton = $derived(vIsHealthChecking || (vIsIdle && server.enabled));
  let vErrorMessage = $derived(
    vHealthState.status === HealthCheckStatus.ERROR ? vHealthState.message : undefined
  );
  let aTools = $derived(
    vHealthState.status === HealthCheckStatus.SUCCESS ? vHealthState.tools : []
  );

  let aConnectionLogs = $derived(
    vHealthState.status === HealthCheckStatus.cConnecting ||
      vHealthState.status === HealthCheckStatus.SUCCESS ||
      vHealthState.status === HealthCheckStatus.ERROR
      ? vHealthState.logs
      : []
  );

  let vSuccessState = $derived(
    vHealthState.status === HealthCheckStatus.SUCCESS ? vHealthState : null
  );
  let vServerInfo = $derived(vSuccessState?.serverInfo);
  let vCapabilities = $derived(vSuccessState?.capabilities);
  let vTransportType = $derived(vSuccessState?.transportType);
  let vProtocolVersion = $derived(vSuccessState?.protocolVersion);
  let vConnectionTimeMs = $derived(vSuccessState?.connectionTimeMs);
  let vInstructions = $derived(vSuccessState?.instructions);

  let vIsEditing = $derived(!server.url.trim());
  let vShowDeleteDialog = $state(false);
  let vEditFormRef: McpServerCardEditForm | null = $state(null);

  function fHandleHealthCheck() {
    cMcpStore.mRunHealthCheck(server);
  }

  async function fStartEditing() {
    vIsEditing = true;
    await tick();
    vEditFormRef?.setInitialValues(
      server.url,
      server.headers || '',
      server.useProxy || false,
      vDisplayName
    );
  }

  function fCancelEditing() {
    if (server.url.trim()) {
      vIsEditing = false;
    } else {
      onDelete();
    }
  }

  function fSaveEditing(pUrl: string, pHeaders: string, pUseProxy: boolean, pName?: string) {
    onUpdate({
      // undefined = prefill untouched, keep any existing custom name;
      // empty string = field cleared, back to the automatic label
      displayName: pName === undefined ? server.displayName : pName.trim() || undefined,
      headers: pHeaders || undefined,
      url: pUrl,
      useProxy: pUseProxy
    });
    vIsEditing = false;

    if (server.enabled && pUrl) {
      setTimeout(
        () => cMcpStore.mRunHealthCheck({ ...server, url: pUrl, useProxy: pUseProxy }),
        100
      );
    }
  }

  function fHandleDeleteClick() {
    vShowDeleteDialog = true;
  }
</script>

<Card.Root class="!gap-3 bg-muted/30 p-4">
  {#if vIsEditing}
    <McpServerCardEditForm
      bind:this={vEditFormRef}
      onCancel={fCancelEditing}
      onSave={fSaveEditing}
      serverId={server.id}
      serverLabel={vDisplayName}
      serverUrl={server.url}
      serverUseProxy={server.useProxy}
    />
  {:else}
    <McpServerCardHeader
      capabilities={vCapabilities}
      disabled={vIsError}
      displayName={vDisplayName}
      enabled={enabled ?? server.enabled}
      faviconUrl={vFaviconUrl}
      {onBrowseResources}
      {onToggle}
      serverInfo={vServerInfo}
      transportType={vTransportType}
    />

    {#if vIsError && vErrorMessage}
      <p class="text-xs text-destructive">{vErrorMessage}</p>
    {/if}

    {#if vIsConnected && vServerInfo?.description}
      <p class="line-clamp-2 text-xs text-muted-foreground">
        {vServerInfo.description}
      </p>
    {/if}

    <div class="grid gap-3">
      {#if vShowSkeleton}
        <div class="space-y-2">
          <div class="flex items-center gap-2">
            <Skeleton class="{cIconClassDefault} rounded" />

            <Skeleton class="h-3 w-24" />
          </div>

          <div class="flex flex-wrap gap-1.5">
            <Skeleton class="h-5 w-16 rounded-full" />

            <Skeleton class="h-5 w-20 rounded-full" />

            <Skeleton class="h-5 w-14 rounded-full" />
          </div>
        </div>

        <div class="space-y-1.5">
          <div class="flex items-center gap-2">
            <Skeleton class="{cIconClassDefault} rounded" />

            <Skeleton class="h-3 w-32" />
          </div>
        </div>
      {:else}
        {#if vIsConnected && vInstructions}
          <McpServerInfo instructions={vInstructions} />
        {/if}

        {#if aTools.length > 0}
          <McpServerCardToolsList tools={aTools} />
        {/if}

        {#if aConnectionLogs.length > 0}
          <McpConnectionLogs connectionTimeMs={vConnectionTimeMs} logs={aConnectionLogs} />
        {/if}
      {/if}
    </div>

    <div class="mt-auto flex justify-between gap-4">
      {#if vShowSkeleton}
        <Skeleton class="h-3 w-28" />
      {:else if vProtocolVersion}
        <div class="flex flex-wrap items-center gap-1">
          <span class="text-[10px] text-muted-foreground">
            {fTranslate('text_cd6999408a95')}
            {vProtocolVersion}
          </span>
        </div>
      {/if}

      <div class="flex items-center gap-2">
        <McpServerCardActions
          isHealthChecking={vIsHealthChecking}
          onDelete={fHandleDeleteClick}
          onEdit={fStartEditing}
          onRefresh={fHandleHealthCheck}
        />
      </div>
    </div>
  {/if}
</Card.Root>

<McpServerCardDeleteDialog
  bind:open={vShowDeleteDialog}
  displayName={vDisplayName}
  onConfirm={onDelete}
  onOpenChange={(pOpen) => (vShowDeleteDialog = pOpen)}
/>
