<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ChevronDown, ChevronRight } from '@lucide/svelte';
  import { McpServerIdentity, TruncatedText } from '$lib/components/app';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import { cIconClassDefault } from '$lib/constants';
  import { ToolSource } from '$lib/enums/tools.enums';
  import { cMcpStore, cPermissionsStore, cToolsStore } from '$lib/stores';
  import { fGetToolUi } from '$lib/utils';
  import { SvelteSet } from 'svelte/reactivity';

  let vExpandedGroups = new SvelteSet<string>();
  let aGroups = $derived(cToolsStore.toolGroups);

  function fToggleExpanded(pKey: string) {
    if (vExpandedGroups.has(pKey)) {
      vExpandedGroups.delete(pKey);
    } else {
      vExpandedGroups.add(pKey);
    }
  }
</script>

{#if aGroups.length === 0}
  <div class="py-8 text-center text-sm text-muted-foreground">
    {fTranslate('text_0ec9813e343b')}
  </div>
{:else}
  <div class="space-y-2">
    <p class="text-sm text-muted-foreground">
      {fTranslate('dgs_122d4cf28fc4')}
    </p>

    {#each aGroups as vGroup (vGroup.key)}
      {@const cIsExpanded = vExpandedGroups.has(vGroup.key)}
      <Collapsible.Root onOpenChange={() => fToggleExpanded(vGroup.key)} open={cIsExpanded}>
        <Collapsible.Trigger
          class="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-sm hover:bg-muted/50"
        >
          {#if cIsExpanded}
            <ChevronDown class="h-3.5 w-3.5 shrink-0" />
          {:else}
            <ChevronRight class="h-3.5 w-3.5 shrink-0" />
          {/if}

          {@const cIsCategoryEnabled =
            vGroup.source !== ToolSource.cMcp && cToolsStore.mIsCategoryEnabled(vGroup.source)}

          {#if vGroup.source !== ToolSource.cMcp}
            <Checkbox
              checked={cIsCategoryEnabled}
              onCheckedChange={() => cToolsStore.mToggleCategory(vGroup.source)}
              onclick={(pE) => pE.stopPropagation()}
            />
          {/if}

          {@const cFaviconUrl = vGroup.serverId
            ? cMcpStore.mGetServerFavicon(vGroup.serverId)
            : null}

          <span class="inline-flex min-w-0 items-center gap-1.5 font-medium">
            {#if vGroup.source === 'mcp'}
              <McpServerIdentity
                displayName={vGroup.label}
                faviconUrl={cFaviconUrl}
                iconClass={cIconClassDefault}
                iconRounded="rounded-sm"
                showVersion={false}
              />
            {:else}
              <TruncatedText class="font-medium" text={vGroup.label} />
            {/if}
          </span>

          <span class="ml-auto shrink-0 text-xs text-muted-foreground">
            {vGroup.tools.length}
            {fTranslate('text_7c9bbe5ec9b3')}{vGroup.tools.length !== 1 ? 's' : ''}
          </span>
        </Collapsible.Trigger>

        <Collapsible.Content>
          <div class="ml-4 border-l border-border/50 pl-2">
            <!-- Header row -->
            <div class="flex items-center gap-2 px-2 py-1 text-xs text-muted-foreground">
              <span class="min-w-0 flex-1">{fTranslate('text_2e53bdcd0740')}</span>

              <span class="w-16 shrink-0 text-center">{fTranslate('text_92c1cdfdf4cb')}</span>

              <span class="w-20 shrink-0 text-center">{fTranslate('text_977618bd8bc7')}</span>
            </div>

            {#each vGroup.tools as vEntry (vEntry.key)}
              {@const cToolName = vEntry.definition.function.name}
              {@const toolUi =
                vEntry.source === ToolSource.SERVER || vEntry.source === ToolSource.cBrowser
                  ? fGetToolUi(cToolName)
                  : null}
              {@const cDisplayLabel = toolUi?.label ?? cToolName}
              {@const IconComponent = toolUi?.icon ?? null}
              {@const cIsEnabled = cToolsStore.mIsToolEnabled(vEntry.key)}
              {@const cPermissionKey = vEntry.key}
              {@const cIsAlwaysAllowed = cPermissionsStore.mHasTool(cPermissionKey)}

              <div class="flex items-center gap-2 rounded px-2 py-1.5 text-sm hover:bg-muted/50">
                <span class="flex min-w-0 flex-1 items-center gap-1.5">
                  {#if IconComponent}
                    <IconComponent class={cIconClassDefault} />
                  {/if}

                  <TruncatedText class="min-w-0" showTooltip={true} text={cDisplayLabel} />
                </span>

                <div class="flex w-16 shrink-0 justify-center">
                  <Checkbox
                    checked={cIsEnabled}
                    class={cIconClassDefault}
                    onCheckedChange={() => cToolsStore.mToggleTool(vEntry.key)}
                  />
                </div>

                <div class="flex w-20 shrink-0 justify-center">
                  <Checkbox
                    checked={cIsAlwaysAllowed}
                    class={cIconClassDefault}
                    onCheckedChange={() => {
                      if (cIsAlwaysAllowed) {
                        cPermissionsStore.mRevokeTool(cPermissionKey);
                      } else {
                        cPermissionsStore.mAllowTool(cPermissionKey);
                      }
                    }}
                  />
                </div>
              </div>
            {/each}
          </div>
        </Collapsible.Content>
      </Collapsible.Root>
    {/each}
  </div>
{/if}
