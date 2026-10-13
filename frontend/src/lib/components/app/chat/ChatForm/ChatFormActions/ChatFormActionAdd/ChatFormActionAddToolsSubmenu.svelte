<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Check, ChevronDown, ChevronRight, Info, Loader2, PencilRuler } from '@lucide/svelte';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cAttachmentMenuText, cCliFlags, cIconClassDefault } from '$lib/constants';
  import { fUseToolsPanel } from '$lib/hooks/use-tools-panel.svelte';
  import { cMcpStore, cToolsStore } from '$lib/stores';
  import type { ToolGroup } from '$lib/types';

  const cToolsPanel = fUseToolsPanel();
  const cHasMcpServersAvailable = $derived(cMcpStore.mGetServers().length > 0);
</script>

<DropdownMenu.fSub onOpenChange={(pOpen) => pOpen && cToolsPanel.mHandleOpen()}>
  <DropdownMenu.SubTrigger class="flex cursor-pointer items-center gap-2">
    <PencilRuler class={cIconClassDefault} />

    <span>{cAttachmentMenuText.TOOLS}</span>
  </DropdownMenu.SubTrigger>

  <DropdownMenu.SubContent class="w-72 p-0">
    {#if cToolsPanel.totalToolCount === 0}
      {#if cToolsStore.loading}
        <div class="px-3 py-4 text-center text-sm text-muted-foreground">
          <Loader2 class="mx-auto mb-1 {cIconClassDefault} animate-spin" />

          {fTranslate('text_efc190cd4ce7')}
        </div>
      {:else if cToolsStore.isToolsEndpointUnreachable}
        <div class="grid gap-2.5 px-3 py-4 text-sm text-muted-foreground">
          <span class="flex gap-2">
            <Info class="mt-0.5 {cIconClassDefault} shrink-0" />

            <span>
              {fTranslate('dgs_6c9bdac273ca')} <code>{cCliFlags.TOOLS}</code>
              {fTranslate('text_534e6d6e87c7')}

              <strong>{fTranslate('dgs_e3ac1ee0f8b3')}</strong>.
            </span>
          </span>

          <span class="flex gap-2">
            <Info class="mt-0.5 {cIconClassDefault} shrink-0" />

            <span>
              {cHasMcpServersAvailable ? 'Enable' : 'Add'}
              {fTranslate('text_35302548c7fd')}

              <strong>{fTranslate('text_ecd806d3c25e')}</strong>.
            </span>
          </span>
        </div>
      {:else if cToolsStore.error}
        <div class="px-3 py-4 text-center text-sm text-muted-foreground">
          {fTranslate('text_5ceb12c7da97')}
        </div>
      {:else if cToolsPanel.noToolsInfoMessage}
        <div class="flex gap-2 px-3 py-4 text-sm text-muted-foreground">
          <Info class="mt-0.5 {cIconClassDefault} shrink-0" />

          <span>{cToolsPanel.noToolsInfoMessage}</span>
        </div>
      {:else}
        <div class="px-3 py-4 text-center text-sm text-muted-foreground">
          {fTranslate('text_0ec9813e343b')}
        </div>
      {/if}
    {:else}
      <div class="max-h-80 overflow-y-auto p-2 pr-1">
        {#each cToolsPanel.categoryGroups as vGroup (vGroup.key)}
          {@render groupRow(vGroup)}
        {/each}

        {#each cToolsPanel.mcpGroups as vGroup (vGroup.key)}
          {@render groupRow(vGroup)}
        {/each}
      </div>
    {/if}
  </DropdownMenu.SubContent>
</DropdownMenu.fSub>

{#snippet groupRow(pGroup: ToolGroup)}
  {@const cIsExpanded = cToolsPanel.expandedGroups.has(pGroup.key)}
  {@const cCheckState = cToolsPanel.mGetGroupCheckState(pGroup)}
  {@const cFavicon = cToolsPanel.mGetFavicon(pGroup)}
  {@const cGroupDisabled = cToolsPanel.mIsGroupDisabled(pGroup)}

  <Collapsible.Root
    onOpenChange={() => cToolsPanel.mToggleGroupExpanded(pGroup.key)}
    open={cIsExpanded}
  >
    <div class="flex items-center gap-1 {cGroupDisabled ? 'pointer-events-none opacity-50' : ''}">
      <Collapsible.Trigger
        class="flex min-w-0 flex-1 items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50"
      >
        {#if cIsExpanded}
          <ChevronDown class="h-3.5 w-3.5 shrink-0" />
        {:else}
          <ChevronRight class="h-3.5 w-3.5 shrink-0" />
        {/if}

        <span class="inline-flex min-w-0 items-center gap-1.5 font-medium">
          {#if cFavicon}
            <img
              alt=""
              class="{cIconClassDefault} shrink-0 rounded-sm"
              onerror={(pE) => {
                (pE.currentTarget as HTMLImageElement).style.display = 'none';
              }}
              src={cFavicon}
            />
          {/if}

          <span class="truncate">{pGroup.label}</span>
        </span>

        <span class="ml-auto shrink-0 text-xs text-muted-foreground">
          {cToolsPanel.mGetEnabledToolCount(pGroup)}/{pGroup.tools.length}
        </span>
      </Collapsible.Trigger>

      <Tooltip.fRoot>
        <Tooltip.Trigger>
          {#snippet child({ props: dProps })}
            <Checkbox
              {...dProps}
              checked={cCheckState.checked}
              class="mr-2 {cIconClassDefault} shrink-0"
              indeterminate={cCheckState.indeterminate}
              onCheckedChange={() => cToolsPanel.mToggleGroupByKey(pGroup.key)}
            />
          {/snippet}
        </Tooltip.Trigger>

        <Tooltip.Content side="right">
          <p>
            {cCheckState.checked ? 'Disable' : 'Enable'}
            {pGroup.tools.length}
            {fTranslate('text_7c9bbe5ec9b3')}{pGroup.tools.length !== 1 ? 's' : ''}
          </p>
        </Tooltip.Content>
      </Tooltip.fRoot>
    </div>

    <Collapsible.Content>
      <div class="ml-4 flex flex-col gap-0.5 border-l border-border/50 pl-2">
        {#each pGroup.tools as vEntry (vEntry.key)}
          {@const cEnabled = cToolsPanel.mIsToolEnabled(vEntry)}
          {@const cParentDisabled = cToolsPanel.mIsToolParentDisabled(vEntry)}
          <button
            class="flex w-full items-center gap-2 rounded px-2 py-1.5 text-left text-sm transition-colors hover:bg-muted/50 {cParentDisabled
              ? 'opacity-50'
              : ''}"
            onclick={() => cToolsPanel.mToggleTool(vEntry)}
            type="button"
          >
            <span
              class="flex size-4 shrink-0 items-center justify-center rounded-[4px] border border-input data-[state=checked]:border-primary data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground"
              data-slot="checkbox"
              data-state={cEnabled ? 'checked' : 'unchecked'}
            >
              {#if cEnabled}
                <Check class="size-3.5" />
              {/if}
            </span>

            <span class="min-w-0 flex-1 truncate font-mono text-[12px]">
              {vEntry.definition.function.name}
            </span>
          </button>
        {/each}
      </div>
    </Collapsible.Content>
  </Collapsible.Root>
{/snippet}
