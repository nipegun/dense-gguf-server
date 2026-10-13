<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Braces, FolderOpen, Loader2, Plus, X } from '@lucide/svelte';
  import {
    McpResourcePreview,
    McpResourcesBrowser,
    McpResourceTemplateForm
  } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import * as Dialog from '$lib/components/ui/dialog';
  import { cIconClassDefault, cPanelCloseMobileClass } from '$lib/constants';
  import { cDeviceStore, cMcpStore } from '$lib/stores';
  import type { MCPResourceContent, MCPResourceInfo, MCPResourceTemplateInfo } from '$lib/types';
  import { fGetResourceDisplayName } from '$lib/utils';
  import { SvelteSet } from 'svelte/reactivity';
  import { toast } from 'svelte-sonner';

  interface Props {
    open?: boolean;
    onOpenChange?: (pOpen: boolean) => void;
    onAttach?: (pResource: MCPResourceInfo) => void;
    preSelectedUri?: string;
  }

  let { onAttach, onOpenChange, open = $bindable(false), preSelectedUri }: Props = $props();

  let vSelectedResources = new SvelteSet<string>();
  let vLastSelectedUri = $state<string | null>(null);
  let vIsAttaching = $state(false);

  let vSelectedTemplate = $state<MCPResourceTemplateInfo | null>(null);
  let vTemplatePreviewUri = $state<string | null>(null);
  let vTemplatePreviewContent = $state<MCPResourceContent[] | null>(null);
  let vTemplatePreviewLoading = $state(false);
  let vTemplatePreviewError = $state<string | null>(null);

  const cTotalCount = $derived(cMcpStore.resources.totalResourceCount);

  $effect(() => {
    if (open) {
      fLoadResources();

      if (preSelectedUri) {
        vSelectedResources.clear();
        vSelectedResources.add(preSelectedUri);
        vLastSelectedUri = preSelectedUri;
      }
    }
  });

  async function fLoadResources() {
    const cInitialized = await cMcpStore.mEnsureInitialized();

    if (cInitialized) {
      await cMcpStore.mFetchAllResources();
    }
  }

  function fHandleOpenChange(pNewOpen: boolean) {
    open = pNewOpen;
    onOpenChange?.(pNewOpen);

    if (!pNewOpen) {
      vSelectedResources.clear();
      vLastSelectedUri = null;
      fClearTemplateState();
    }
  }

  function fClearTemplateState() {
    vSelectedTemplate = null;
    vTemplatePreviewUri = null;
    vTemplatePreviewContent = null;
    vTemplatePreviewLoading = false;
    vTemplatePreviewError = null;
  }

  function fHandleTemplateSelect(pTemplate: MCPResourceTemplateInfo) {
    vSelectedResources.clear();
    vLastSelectedUri = null;

    if (
      vSelectedTemplate?.uriTemplate === pTemplate.uriTemplate &&
      vSelectedTemplate?.serverName === pTemplate.serverName
    ) {
      fClearTemplateState();

      return;
    }

    vSelectedTemplate = pTemplate;
    vTemplatePreviewUri = null;
    vTemplatePreviewContent = null;
    vTemplatePreviewLoading = false;
    vTemplatePreviewError = null;
  }

  async function fHandleTemplateResolve(pUri: string, pServerName: string) {
    vTemplatePreviewUri = pUri;
    vTemplatePreviewContent = null;
    vTemplatePreviewLoading = true;
    vTemplatePreviewError = null;

    try {
      const cContent = await cMcpStore.mReadResourceByUri(pServerName, pUri);

      if (cContent) {
        vTemplatePreviewContent = cContent;
      } else {
        vTemplatePreviewError = 'Failed to read resource';
      }
    } catch (error) {
      vTemplatePreviewError = error instanceof Error ? error.message : 'Unknown error';
    } finally {
      vTemplatePreviewLoading = false;
    }
  }

  function fHandleTemplateCancelForm() {
    fClearTemplateState();
  }

  async function fHandleAttachTemplateResource() {
    if (!vTemplatePreviewUri || !vSelectedTemplate || !vTemplatePreviewContent) return;

    vIsAttaching = true;

    try {
      const cKnownResource = cMcpStore.resources.mFindResourceByUri(vTemplatePreviewUri);

      if (cKnownResource) {
        if (!cMcpStore.resources.mIsAttached(cKnownResource.uri)) {
          await cMcpStore.mAttachResource(cKnownResource.uri);
        }

        toast.success(`Resource attached: ${cKnownResource.title || cKnownResource.name}`);
      } else {
        if (cMcpStore.resources.mIsAttached(vTemplatePreviewUri)) {
          toast.info(fTranslate('text_1aa9ca580c64'));
          fHandleOpenChange(false);

          return;
        }

        const cResourceInfo: MCPResourceInfo = {
          name: vTemplatePreviewUri.split('/').pop() || vTemplatePreviewUri,
          serverName: vSelectedTemplate.serverName,
          uri: vTemplatePreviewUri
        };
        const attachment = cMcpStore.resources.mAddAttachment(cResourceInfo);

        cMcpStore.resources.mUpdateAttachmentContent(attachment.id, vTemplatePreviewContent);

        toast.success(`Resource attached: ${cResourceInfo.name}`);
      }

      fHandleOpenChange(false);
    } catch (error) {
      console.error('Failed to attach template resource:', error);
    } finally {
      vIsAttaching = false;
    }
  }

  function fHandleResourceSelect(pResource: MCPResourceInfo, pShiftKey: boolean = false) {
    fClearTemplateState();

    if (pShiftKey && vLastSelectedUri) {
      const aAllResources = fGetAllResourcesFlatInTreeOrder();
      const cLastIndex = aAllResources.findIndex((pR) => pR.uri === vLastSelectedUri);
      const cCurrentIndex = aAllResources.findIndex((pR) => pR.uri === pResource.uri);

      if (cLastIndex !== -1 && cCurrentIndex !== -1) {
        const cStart = Math.min(cLastIndex, cCurrentIndex);
        const cEnd = Math.max(cLastIndex, cCurrentIndex);

        for (let vI = cStart; vI <= cEnd; vI++) {
          vSelectedResources.add(aAllResources[vI].uri);
        }
      }
    } else {
      vSelectedResources.clear();
      vSelectedResources.add(pResource.uri);
      vLastSelectedUri = pResource.uri;
    }
  }

  function fHandleResourceToggle(pResource: MCPResourceInfo, pChecked: boolean) {
    fClearTemplateState();

    if (pChecked) {
      vSelectedResources.add(pResource.uri);
    } else {
      vSelectedResources.delete(pResource.uri);
    }

    vLastSelectedUri = pResource.uri;
  }

  function fGetAllResourcesFlatInTreeOrder(): MCPResourceInfo[] {
    const aAllResources: MCPResourceInfo[] = [];
    const dResourcesMap = cMcpStore.resources.serverResources;

    for (const [vServerName, vServerRes] of dResourcesMap.entries()) {
      for (const resource of vServerRes.resources) {
        aAllResources.push({ ...resource, serverName: vServerName });
      }
    }

    return aAllResources.sort((pA, pB) => {
      const aName = fGetResourceDisplayName(pA);
      const cBName = fGetResourceDisplayName(pB);

      return aName.localeCompare(cBName);
    });
  }

  async function fHandleAttach() {
    if (vSelectedResources.size === 0) return;

    vIsAttaching = true;

    try {
      const aAllResources = fGetAllResourcesFlatInTreeOrder();
      const aResourcesToAttach = aAllResources.filter((pR) => vSelectedResources.has(pR.uri));

      for (const resource of aResourcesToAttach) {
        await cMcpStore.mAttachResource(resource.uri);
        onAttach?.(resource);
      }

      const cCount = aResourcesToAttach.length;

      toast.success(
        cCount === 1
          ? `Resource attached: ${aResourcesToAttach[0].name}`
          : `${cCount} resources attached`
      );

      fHandleOpenChange(false);
    } catch (error) {
      console.error('Failed to attach resources:', error);
    } finally {
      vIsAttaching = false;
    }
  }

  const cSelectedTemplateUri = $derived(vSelectedTemplate?.uriTemplate ?? null);

  const cHasTemplateResult = $derived(
    !!vSelectedTemplate && !!vTemplatePreviewContent && !!vTemplatePreviewUri
  );
</script>

<Dialog.fRoot onOpenChange={fHandleOpenChange} {open}>
  <Dialog.Content
    class="max-h-[80vh] md:max-w-4xl! w-full! overflow-hidden p-0"
    onOpenAutoFocus={(pEvent) => cDeviceStore.isMobile && pEvent.preventDefault()}
  >
    <Dialog.Header class="border-b border-border/30 px-4 py-3" showCloseButton={false}>
      <Dialog.Title class="flex items-center gap-2">
        <FolderOpen class="h-5 w-5" />

        <span>{fTranslate('text_77a814db2847')}</span>

        {#if cTotalCount > 0}
          <span class="text-sm font-normal text-muted-foreground">({cTotalCount})</span>
        {/if}
      </Dialog.Title>

      <Dialog.Description>
        {fTranslate('text_e49707e6f350')}
      </Dialog.Description>

      <!-- the body is flush with the dialog, so the corner close needs its own inset -->
      <Dialog.Close
        class="absolute top-0 right-0 rounded-xs opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:ring-2 focus:ring-ring focus:ring-offset-2 focus:outline-hidden max-md:top-1 max-md:right-1 {cPanelCloseMobileClass}"
      >
        <X class="size-4" />

        <span class="sr-only">{fTranslate('text_7d9eb7acb13e')}</span>
      </Dialog.Close>
    </Dialog.Header>

    <div class="flex h-[500px] min-w-0">
      <div class="w-72 shrink-0 overflow-y-auto border-r border-border/30 p-4">
        <McpResourcesBrowser
          expandToUri={preSelectedUri}
          onSelect={fHandleResourceSelect}
          onTemplateSelect={fHandleTemplateSelect}
          onToggle={fHandleResourceToggle}
          selectedTemplateUri={cSelectedTemplateUri}
          selectedUris={vSelectedResources}
        />
      </div>

      <div class="min-w-0 flex-1 overflow-auto p-4">
        {#if vSelectedTemplate && !vTemplatePreviewContent}
          <div class="flex h-full flex-col">
            <div class="mb-3 flex items-center gap-2">
              <Braces class="{cIconClassDefault} text-muted-foreground" />

              <span class="text-sm font-medium">
                {vSelectedTemplate.title || vSelectedTemplate.name}
              </span>
            </div>

            {#if vSelectedTemplate.description}
              <p class="mb-4 text-xs text-muted-foreground">
                {vSelectedTemplate.description}
              </p>
            {/if}

            <div class="mb-4 rounded-md border border-border/50 bg-muted/30 px-3 py-2">
              <p class="font-mono text-xs break-all text-muted-foreground">
                {vSelectedTemplate.uriTemplate}
              </p>
            </div>

            {#if vTemplatePreviewLoading}
              <div class="flex flex-1 items-center justify-center">
                <Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
              </div>
            {:else if vTemplatePreviewError}
              <div class="flex flex-1 flex-col items-center justify-center gap-2 text-red-500">
                <span class="text-sm">{vTemplatePreviewError}</span>

                <Button
                  onclick={() => {
                    vTemplatePreviewError = null;
                  }}
                  size="sm"
                  variant="outline"
                >
                  {fTranslate('text_d8b8392e2c54')}
                </Button>
              </div>
            {:else}
              <McpResourceTemplateForm
                onCancel={fHandleTemplateCancelForm}
                onResolve={fHandleTemplateResolve}
                template={vSelectedTemplate}
              />
            {/if}
          </div>
        {:else if cHasTemplateResult}
          <!-- Template resolved: show preview -->
          <McpResourcePreview
            preloadedContent={vTemplatePreviewContent}
            resource={{
              name: vTemplatePreviewUri?.split('/').pop() || (vTemplatePreviewUri ?? ''),
              serverName: vSelectedTemplate?.serverName || '',
              uri: vTemplatePreviewUri ?? ''
            }}
          />
        {:else if vSelectedResources.size === 1}
          {@const aAllResources = fGetAllResourcesFlatInTreeOrder()}
          {@const cSelectedResource = aAllResources.find((pR) => vSelectedResources.has(pR.uri))}

          <McpResourcePreview resource={cSelectedResource ?? null} />
        {:else if vSelectedResources.size > 1}
          <div class="flex flex-col gap-10">
            {#each fGetAllResourcesFlatInTreeOrder() as vResource (vResource.uri)}
              {#if vSelectedResources.has(vResource.uri)}
                <McpResourcePreview resource={vResource} />
              {/if}
            {/each}
          </div>
        {:else}
          <div class="flex h-full items-center justify-center text-sm text-muted-foreground">
            {fTranslate('text_a09ca1a10c95')}
          </div>
        {/if}
      </div>
    </div>

    <Dialog.Footer class="border-t border-border/30 px-6 py-4">
      <Button onclick={() => fHandleOpenChange(false)} variant="outline"
        >{fTranslate('text_19766ed6ccb2')}</Button
      >

      {#if cHasTemplateResult}
        <Button disabled={vIsAttaching} onclick={fHandleAttachTemplateResource}>
          {#if vIsAttaching}
            <Loader2 class="mr-2 {cIconClassDefault} animate-spin" />
          {:else}
            <Plus class="mr-2 {cIconClassDefault}" />
          {/if}

          {fTranslate('text_1a3db5f016d3')}
        </Button>
      {:else}
        <Button disabled={vSelectedResources.size === 0 || vIsAttaching} onclick={fHandleAttach}>
          {#if vIsAttaching}
            <Loader2 class="mr-2 {cIconClassDefault} animate-spin" />
          {:else}
            <Plus class="mr-2 {cIconClassDefault}" />
          {/if}

          {fTranslate('text_d406ade2958c')}
          {vSelectedResources.size > 0 ? `(${vSelectedResources.size})` : 'Resource'}
        </Button>
      {/if}
    </Dialog.Footer>
  </Dialog.Content>
</Dialog.fRoot>
