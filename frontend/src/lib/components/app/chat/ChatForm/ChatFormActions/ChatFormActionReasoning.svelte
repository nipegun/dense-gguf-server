<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Check, Info, Lightbulb, LightbulbOff } from '@lucide/svelte';
  import { Button } from '$lib/components/ui/button';
  import * as Drawer from '$lib/components/ui/drawer';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cIconClassDefault } from '$lib/constants';
  import { ReasoningEffort } from '$lib/enums';
  import { fUseReasoningMenu } from '$lib/hooks/use-reasoning-menu.svelte';
  import { cDeviceStore } from '$lib/stores';
  import type { ReasoningEffortLevel } from '$lib/types';

  const reasoning = fUseReasoningMenu();

  let vIsOpen = $state(false);

  // "Default" is the resting state, where the bulb alone carries the meaning
  let vIsDefault = $derived(reasoning.currentEffort === ReasoningEffort.cDefault);
  // a phone picks the level in a drawer, where a thumb reaches it
  let vIsMobile = $derived(cDeviceStore.isMobile);

  const cMaxEffortHint = 'Maximum reasoning effort with extended context usage';

  function fSelect(pLevel: ReasoningEffortLevel): void {
    reasoning.mSelect(pLevel);
    vIsOpen = false;
  }
</script>

{#snippet triggerLabel()}
  <span class="flex items-center gap-0.75 {reasoning.isOff ? 'text-muted-foreground' : ''}">
    {#if reasoning.isOff}
      <LightbulbOff class="size-3 max-md:size-4 shrink-0" />
    {:else}
      <Lightbulb class="size-3 max-md:size-4 shrink-0" />
    {/if}

    {#if !vIsDefault}
      <span class="capitalize">{reasoning.currentEffort}</span>
    {/if}
  </span>
{/snippet}

{#snippet levelRow(pLevel: ReasoningEffortLevel)}
  {@const cTokenLabel = reasoning.mTokenLabel(pLevel)}

  <button
    aria-pressed={reasoning.mIsSelected(pLevel)}
    class="flex min-h-11 w-full cursor-pointer items-center gap-3 rounded-md px-3 text-left text-sm transition-colors hover:bg-accent"
    onclick={() => fSelect(pLevel)}
    type="button"
  >
    {#if reasoning.mIsSelected(pLevel)}
      <Check class="{cIconClassDefault} shrink-0 text-foreground" />
    {:else}
      <div class="{cIconClassDefault} shrink-0"></div>
    {/if}

    <span class="min-w-0 flex-1 truncate">{pLevel.label}</span>

    {#if cTokenLabel}
      <span class="shrink-0 text-[11px] text-muted-foreground opacity-60">{cTokenLabel}</span>
    {/if}

    {#if pLevel.hasInfo}
      <!-- no hover on a phone: the hint rides on the icon's own title there -->
      <Info class="h-3.5 w-3.5 shrink-0 text-muted-foreground" title={cMaxEffortHint} />
    {/if}
  </button>
{/snippet}

<!-- the level belongs to the chat, not to the model list -->
{#if vIsMobile}
  <Button
    aria-label={fTranslate('dgs_3236aeec43e5')}
    class="h-auto gap-1 rounded-sm bg-transparent! backdrop-blur-none! {vIsDefault
      ? 'px-1'
      : 'px-1.75!'} py-1 text-xs max-md:h-8"
    onclick={() => (vIsOpen = true)}
    variant="ghost"
  >
    {@render triggerLabel()}
  </Button>

  <Drawer.Root bind:open={vIsOpen}>
    <Drawer.Content>
      <Drawer.Header>
        <Drawer.Title>{fTranslate('dgs_3236aeec43e5')}</Drawer.Title>
      </Drawer.Header>

      <div class="flex flex-col px-2 pb-4">
        {#each reasoning.levels as vLevel (vLevel.value)}
          {@render levelRow(vLevel)}
        {/each}
      </div>
    </Drawer.Content>
  </Drawer.Root>
{:else}
  <DropdownMenu.fRoot>
    <DropdownMenu.Trigger>
      {#snippet child({ props: dProps })}
        <Button
          {...dProps}
          aria-label={fTranslate('dgs_3236aeec43e5')}
          class="h-auto gap-1 rounded-sm bg-transparent! backdrop-blur-none! {vIsDefault
            ? 'px-1'
            : 'px-1.75!'} py-1 text-xs hover:bg-transparent!"
          variant="ghost"
        >
          {@render triggerLabel()}
        </Button>
      {/snippet}
    </DropdownMenu.Trigger>

    <DropdownMenu.Content align="end" class="min-w-56">
      {#each reasoning.levels as vLevel (vLevel.value)}
        {@const cTokenLabel = reasoning.mTokenLabel(vLevel)}

        <DropdownMenu.Item class="gap-3" onSelect={() => reasoning.mSelect(vLevel)}>
          {#if reasoning.mIsSelected(vLevel)}
            <Check class="{cIconClassDefault} shrink-0 text-foreground" />
          {:else}
            <div class="{cIconClassDefault} shrink-0"></div>
          {/if}

          <span class="min-w-0 flex-1 truncate">{vLevel.label}</span>

          {#if cTokenLabel}
            <span class="shrink-0 text-[11px] text-muted-foreground opacity-60">{cTokenLabel}</span>
          {/if}

          {#if vLevel.hasInfo}
            <Tooltip.fRoot>
              <Tooltip.Trigger>
                <Info class="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              </Tooltip.Trigger>

              <Tooltip.Content side="right">
                <p>{cMaxEffortHint}</p>
              </Tooltip.Content>
            </Tooltip.fRoot>
          {/if}
        </DropdownMenu.Item>
      {/each}
    </DropdownMenu.Content>
  </DropdownMenu.fRoot>
{/if}
