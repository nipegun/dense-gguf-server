<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import McpLogo from './McpLogo.svelte';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cIconClassDefault, cMaxDisplayedMcpAvatars } from '$lib/constants';
  import { HealthCheckStatus, ToolSource } from '$lib/enums';
  import { cConversationsStore, cMcpStore } from '$lib/stores';

  interface Props {
    class?: string;
    onclick?: () => void;
  }

  let { class: className = '', onclick }: Props = $props();

  let aMcpServers = $derived(cMcpStore.mGetServers().filter((pS) => pS.enabled));
  // respect the active conversation's tool policy, not just global enablement
  let aEnabledMcpServersForChat = $derived(
    aMcpServers.filter(
      (pS) =>
        pS.url.trim() &&
        cConversationsStore.preferences.mIsCategoryEnabled(ToolSource.cMcp) &&
        cConversationsStore.preferences.mIsServerToolsEnabled(pS.id)
    )
  );
  let aHealthyEnabledMcpServers = $derived(
    aEnabledMcpServersForChat.filter((pS) => {
      const cHealthState = cMcpStore.mGetHealthCheckState(pS.id);

      return cHealthState.status !== HealthCheckStatus.ERROR;
    })
  );
  let vHasEnabledMcpServers = $derived(aEnabledMcpServersForChat.length > 0);
  let vExtraServersCount = $derived(
    Math.max(0, aHealthyEnabledMcpServers.length - cMaxDisplayedMcpAvatars)
  );
  let aMcpFavicons = $derived(
    aHealthyEnabledMcpServers
      .slice(0, cMaxDisplayedMcpAvatars)
      .map((pS) => ({
        id: pS.id,
        name: cMcpStore.mGetServerDisplayName(pS.id),
        url: cMcpStore.mGetServerFavicon(pS.id)
      }))
      .filter((pF) => pF.url !== null)
  );
</script>

{#if !vHasEnabledMcpServers}
  <button
    class={[
      'inline-flex cursor-pointer items-center gap-0.75 opacity-70 transition-opacity hover:opacity-100',
      className,
      'opacity-50 hover:opacity-100'
    ]}
    {onclick}
  >
    <Tooltip.fRoot>
      <Tooltip.Trigger>
        <McpLogo class={cIconClassDefault} />
      </Tooltip.Trigger>

      <Tooltip.Content>
        <p>{fTranslate('text_aab669a72882')}</p>
      </Tooltip.Content>
    </Tooltip.fRoot>
  </button>
{:else if aMcpFavicons.length > 0}
  <button class={['inline-flex items-center gap-0.75', className]} {onclick}>
    <div class="flex -space-x-1">
      {#each aMcpFavicons as vFavicon (vFavicon.id)}
        <Tooltip.fRoot>
          <Tooltip.Trigger>
            <div class="box-shadow-lg overflow-hidden rounded-full bg-muted ring-1 ring-muted">
              <img
                alt=""
                class={cIconClassDefault}
                onerror={(pE) => {
                  (pE.currentTarget as HTMLImageElement).style.display = 'none';
                }}
                src={vFavicon.url}
              />
            </div>
          </Tooltip.Trigger>

          <Tooltip.Content>
            <p>{vFavicon.name}</p>
          </Tooltip.Content>
        </Tooltip.fRoot>
      {/each}
    </div>

    {#if vExtraServersCount > 0}
      <span class="text-xs text-muted-foreground">+{vExtraServersCount}</span>
    {/if}
  </button>
{/if}
