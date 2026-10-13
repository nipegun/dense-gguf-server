<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import {
    fBuildResourceTree,
    fCountTreeResources,
    type ResourceTreeNode,
    fSortTreeChildren
  } from './mcp-resources-browser';
  import { Braces, ChevronDown, ChevronRight, FolderOpen, Loader2 } from '@lucide/svelte';
  import { McpServerIdentity } from '$lib/components/app/mcp';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import { cIconClassDefault } from '$lib/constants';
  import { cMcpStore } from '$lib/stores';
  import type { MCPResourceInfo, MCPResourceTemplateInfo, MCPServerResources } from '$lib/types';
  import { fGetDisplayName, fGetResourceIcon } from '$lib/utils';
  import { SvelteSet } from 'svelte/reactivity';

  interface Props {
    serverName: string;
    serverRes: MCPServerResources;
    isExpanded: boolean;
    selectedUris: Set<string>;
    selectedTemplateUri?: string | null;
    expandedFolders: SvelteSet<string>;
    onToggleServer: () => void;
    onToggleFolder: (pFolderId: string) => void;
    onSelect?: (pResource: MCPResourceInfo, pShiftKey?: boolean) => void;
    onToggle?: (pResource: MCPResourceInfo, pChecked: boolean) => void;
    onTemplateSelect?: (pTemplate: MCPResourceTemplateInfo) => void;
    searchQuery?: string;
  }

  let {
    expandedFolders,
    isExpanded,
    onSelect,
    onTemplateSelect,
    onToggle,
    onToggleFolder,
    onToggleServer,
    searchQuery = '',
    selectedTemplateUri,
    selectedUris,
    serverName,
    serverRes
  }: Props = $props();

  let vServerDisplayName = $derived(cMcpStore.mGetServerDisplayName(serverName));
  let vServerFaviconUrl = $derived(cMcpStore.mGetServerFavicon(serverName));

  const cHasResources = $derived(serverRes.resources.length > 0);
  const cHasTemplates = $derived(serverRes.templates.length > 0);
  const cHasContent = $derived(cHasResources || cHasTemplates);
  const cResourceTree = $derived(fBuildResourceTree(serverRes.resources, serverName, searchQuery));

  const aTemplateInfos = $derived<MCPResourceTemplateInfo[]>(
    serverRes.templates.map((pT) => ({
      annotations: pT.annotations,
      description: pT.description,
      icons: pT.icons,
      mimeType: pT.mimeType,
      name: pT.name,
      serverName,
      title: pT.title,
      uriTemplate: pT.uriTemplate
    }))
  );

  function fHandleResourceClick(pResource: MCPResourceInfo, pEvent: MouseEvent) {
    onSelect?.(pResource, pEvent.shiftKey);
  }

  function fHandleCheckboxChange(pResource: MCPResourceInfo, pChecked: boolean) {
    onToggle?.(pResource, pChecked);
  }

  function fIsResourceSelected(pResource: MCPResourceInfo): boolean {
    return selectedUris.has(pResource.uri);
  }
</script>

{#snippet renderTreeNode(node: ResourceTreeNode, pDepth: number, pParentPath: string)}
  {@const cIsFolder = !node.resource && node.children.size > 0}
  {@const cFolderId = `${serverName}:${pParentPath}/${node.name}`}
  {@const cIsFolderExpanded = expandedFolders.has(cFolderId)}

  {#if cIsFolder}
    {@const cFolderCount = fCountTreeResources(node)}
    <Collapsible.Root onOpenChange={() => onToggleFolder(cFolderId)} open={cIsFolderExpanded}>
      <Collapsible.Trigger
        class="flex w-full items-center gap-2 rounded px-2 py-1 text-sm hover:bg-muted/50"
      >
        {#if cIsFolderExpanded}
          <ChevronDown class="h-3 w-3" />
        {:else}
          <ChevronRight class="h-3 w-3" />
        {/if}

        <FolderOpen class="h-3.5 w-3.5 text-muted-foreground" />

        <span class="font-medium">{node.name}</span>

        <span class="text-xs text-muted-foreground">({cFolderCount})</span>
      </Collapsible.Trigger>

      <Collapsible.Content>
        <div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
          {#each fSortTreeChildren( [...node.children.values()] ) as vChild (vChild.resource?.uri || `${serverName}:${pParentPath}/${node.name}/${vChild.name}`)}
            {@render renderTreeNode(vChild, pDepth + 1, `${pParentPath}/${node.name}`)}
          {/each}
        </div>
      </Collapsible.Content>
    </Collapsible.Root>
  {:else if node.resource}
    {@const resource = node.resource}
    {@const ResourceIcon = fGetResourceIcon(resource.mimeType, resource.uri)}
    {@const cIsSelected = fIsResourceSelected(resource)}
    {@const cResourceDisplayName = resource.title || fGetDisplayName(node.name)}

    <div class="group flex w-full items-center gap-2">
      {#if onToggle}
        <Checkbox
          checked={cIsSelected}
          class={cIconClassDefault}
          onCheckedChange={(pChecked: boolean | 'indeterminate') =>
            fHandleCheckboxChange(resource, pChecked === true)}
        />
      {/if}

      <button
        class={[
          'flex flex-1 items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
          'hover:bg-muted/50',
          cIsSelected && 'bg-muted'
        ]}
        onclick={(pE: MouseEvent) => fHandleResourceClick(resource, pE)}
        title={cResourceDisplayName}
      >
        <ResourceIcon class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

        <span class="min-w-0 flex-1 truncate text-left">
          {cResourceDisplayName}
        </span>
      </button>
    </div>
  {/if}
{/snippet}

<Collapsible.Root onOpenChange={onToggleServer} open={isExpanded}>
  <Collapsible.Trigger
    class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50"
  >
    {#if isExpanded}
      <ChevronDown class="h-3.5 w-3.5" />
    {:else}
      <ChevronRight class="h-3.5 w-3.5" />
    {/if}

    <span class="inline-flex flex-col items-start gap-1 text-left">
      <div class="inline-flex min-w-0 items-center gap-1.5">
        <McpServerIdentity
          displayName={vServerDisplayName}
          faviconUrl={vServerFaviconUrl}
          iconClass={cIconClassDefault}
          showVersion={false}
        />
      </div>

      <span class="text-xs text-muted-foreground">
        ({serverRes.resources.length}
        {fTranslate('text_5de95319f174')}{serverRes.resources.length !== 1
          ? 's'
          : ''}{#if cHasTemplates}, {serverRes.templates.length}
          {fTranslate('text_5cde0f1298f4')}{serverRes.templates.length !== 1 ? 's' : ''}{/if})
      </span>
    </span>

    {#if serverRes.loading}
      <Loader2 class="ml-auto h-3 w-3 animate-spin text-muted-foreground" />
    {/if}
  </Collapsible.Trigger>

  <Collapsible.Content>
    <div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
      {#if serverRes.error}
        <div class="py-1 text-xs text-red-500">
          {fTranslate('text_617062906764')}
          {serverRes.error}
        </div>
      {:else if !cHasContent}
        <div class="py-1 text-xs text-muted-foreground">{fTranslate('text_76010760c629')}</div>
      {:else}
        {#if cHasResources}
          {#each fSortTreeChildren( [...cResourceTree.children.values()] ) as vChild (vChild.resource?.uri || `${serverName}:${vChild.name}`)}
            {@render renderTreeNode(vChild, 1, '')}
          {/each}
        {/if}

        {#if cHasTemplates && onTemplateSelect}
          {#if cHasResources}
            <div class="my-1 border-t border-border/30"></div>
          {/if}

          <div
            class="py-0.5 text-[11px] font-medium tracking-wide text-muted-foreground/70 uppercase"
          >
            {fTranslate('text_56b564b75c7f')}
          </div>

          {#each aTemplateInfos as vTemplate (vTemplate.uriTemplate)}
            <button
              class={[
                'flex w-full items-center gap-2 rounded px-2 py-1 text-left text-sm transition-colors',
                'hover:bg-muted/50',
                selectedTemplateUri === vTemplate.uriTemplate && 'bg-muted'
              ]}
              onclick={() => onTemplateSelect(vTemplate)}
              title={vTemplate.uriTemplate}
            >
              <Braces class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />

              <span class="min-w-0 flex-1 truncate text-left">
                {vTemplate.title || vTemplate.name}
              </span>
            </button>
          {/each}
        {/if}
      {/if}
    </div>
  </Collapsible.Content>
</Collapsible.Root>
