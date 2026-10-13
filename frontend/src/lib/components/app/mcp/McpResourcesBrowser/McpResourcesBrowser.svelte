<script lang="ts">
  import McpResourcesBrowserEmptyState from './McpResourcesBrowserEmptyState.svelte';
  import McpResourcesBrowserHeader from './McpResourcesBrowserHeader.svelte';
  import McpResourcesBrowserServerItem from './McpResourcesBrowserServerItem.svelte';
  import { cMcpStore } from '$lib/stores';
  import type { MCPResourceInfo, MCPResourceTemplateInfo, MCPServerResources } from '$lib/types';
  import { fParseResourcePath } from '$lib/utils';
  import { SvelteMap, SvelteSet } from 'svelte/reactivity';

  interface Props {
    onSelect?: (pResource: MCPResourceInfo, pShiftKey?: boolean) => void;
    onToggle?: (pResource: MCPResourceInfo, pChecked: boolean) => void;
    onTemplateSelect?: (pTemplate: MCPResourceTemplateInfo) => void;
    selectedUris?: Set<string>;
    selectedTemplateUri?: string | null;
    expandToUri?: string;
    class?: string;
  }

  let {
    class: className,
    expandToUri,
    onSelect,
    onTemplateSelect,
    onToggle,
    selectedTemplateUri,
    selectedUris = new Set()
  }: Props = $props();

  let vExpandedServers = new SvelteSet<string>();
  let vExpandedFolders = new SvelteSet<string>();
  let vSearchQuery = $state('');

  const dResources = $derived(cMcpStore.resources.serverResources);
  const cIsLoading = $derived(cMcpStore.resources.isLoading);

  const dFilteredResources = $derived.by(() => {
    if (!vSearchQuery.trim()) {
      return dResources;
    }

    const cQuery = vSearchQuery.toLowerCase();
    const cFiltered = new SvelteMap();

    for (const [vServerName, vServerRes] of dResources.entries()) {
      const aFilteredResources = vServerRes.resources.filter((pR) => {
        return (
          pR.title?.toLowerCase().includes(cQuery) ||
          pR.uri.toLowerCase().includes(cQuery) ||
          vServerName.toLowerCase().includes(cQuery)
        );
      });
      const aFilteredTemplates = vServerRes.templates.filter((pT) => {
        return (
          pT.name?.toLowerCase().includes(cQuery) ||
          pT.title?.toLowerCase().includes(cQuery) ||
          pT.uriTemplate.toLowerCase().includes(cQuery) ||
          vServerName.toLowerCase().includes(cQuery)
        );
      });

      if (aFilteredResources.length > 0 || aFilteredTemplates.length > 0 || cQuery.trim()) {
        cFiltered.set(vServerName, {
          ...vServerRes,
          resources: aFilteredResources,
          templates: aFilteredTemplates
        });
      }
    }

    return cFiltered;
  });

  $effect(() => {
    if (expandToUri && dResources.size > 0) {
      fAutoExpandToResource(expandToUri);
    }
  });

  function fAutoExpandToResource(pUri: string) {
    for (const [vServerName, vServerRes] of dResources.entries()) {
      const resource = vServerRes.resources.find((pR) => pR.uri === pUri);

      if (resource) {
        vExpandedServers.add(vServerName);

        const aPathParts = fParseResourcePath(pUri);

        if (aPathParts.length > 1) {
          let vCurrentPath = '';

          for (let vI = 0; vI < aPathParts.length - 1; vI++) {
            vCurrentPath = `${vCurrentPath}/${aPathParts[vI]}`;
            const cFolderId = `${vServerName}:${vCurrentPath}`;

            vExpandedFolders.add(cFolderId);
          }
        }

        break;
      }
    }
  }

  function fToggleServer(pServerName: string) {
    if (vExpandedServers.has(pServerName)) {
      vExpandedServers.delete(pServerName);
    } else {
      vExpandedServers.add(pServerName);
    }
  }

  function fToggleFolder(pFolderId: string) {
    if (vExpandedFolders.has(pFolderId)) {
      vExpandedFolders.delete(pFolderId);
    } else {
      vExpandedFolders.add(pFolderId);
    }
  }

  function fHandleRefresh() {
    cMcpStore.mFetchAllResources();
  }
</script>

<div class={['flex flex-col gap-2', className]}>
  <McpResourcesBrowserHeader
    isLoading={cIsLoading}
    onRefresh={fHandleRefresh}
    onSearch={(pQ) => (vSearchQuery = pQ)}
    searchQuery={vSearchQuery}
  />

  <div class="flex flex-col gap-1">
    {#if dFilteredResources.size === 0}
      <McpResourcesBrowserEmptyState isLoading={cIsLoading} />
    {:else}
      {#each [...dFilteredResources.entries()] as [vServerName, vServerRes] (vServerName)}
        <McpResourcesBrowserServerItem
          expandedFolders={vExpandedFolders}
          isExpanded={vExpandedServers.has(vServerName as string)}
          {onSelect}
          {onTemplateSelect}
          {onToggle}
          onToggleFolder={fToggleFolder}
          onToggleServer={() => fToggleServer(vServerName as string)}
          searchQuery={vSearchQuery}
          {selectedTemplateUri}
          {selectedUris}
          serverName={vServerName as string}
          serverRes={vServerRes as MCPServerResources}
        />
      {/each}
    {/if}
  </div>
</div>
