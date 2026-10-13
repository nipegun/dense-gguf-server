<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Heart, HeartOff, Info } from '@lucide/svelte';
  import { ActionIcon } from '$lib/components/app';
  import { cDeviceStore, cModelsStore, cUiStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';

  interface Props {
    isFav: boolean;
    option: ModelOption;
    /** Selector rows reveal the actions on hover, table rows keep them visible. */
    revealOnHover?: boolean;
  }

  let { isFav, option, revealOnHover = true }: Props = $props();

  // the favorite heart stays visible, so the reveal applies to the other icons
  const cRevealClass =
    'pointer-events-none opacity-0 group-hover:pointer-events-auto group-hover:opacity-100 [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100';
</script>

<div
  class="flex items-center justify-center gap-1 max-md:gap-2.5"
  onclick={(pEvent) => pEvent.stopPropagation()}
  onkeydown={(pEvent) => pEvent.stopPropagation()}
  role="presentation"
>
  <span class={revealOnHover ? cRevealClass : ''}>
    <ActionIcon
      class="h-5 w-5 hover:text-foreground"
      icon={Info}
      iconSize="h-4 w-4"
      onclick={() =>
        // a phone has no manager: its information dialog takes the icon
        cDeviceStore.isMobile
          ? cUiStore.mOpenModelInformation(option)
          : cUiStore.mOpenModelsManager(option.id)}
      tooltip={fTranslate('dgs_c9a71acd1c40')}
      tooltipAsTitle
    />
  </span>

  {#if isFav}
    <span class="flex h-5 w-5 items-center justify-center">
      <span class="flex group-hover:hidden [@media(pointer:coarse)]:hidden">
        <ActionIcon
          class="h-5 w-5 text-rose-500"
          icon={Heart}
          iconSize="h-4 w-4"
          onclick={() => cModelsStore.mToggleFavorite(option.model)}
          tooltip={fTranslate('text_18720329ca30')}
          tooltipAsTitle
        />
      </span>

      <span class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
        <ActionIcon
          class="h-5 w-5 hover:text-foreground"
          icon={HeartOff}
          iconSize="h-4 w-4"
          onclick={() => cModelsStore.mToggleFavorite(option.model)}
          tooltip={fTranslate('text_18720329ca30')}
          tooltipAsTitle
        />
      </span>
    </span>
  {:else}
    <span class={revealOnHover ? cRevealClass : ''}>
      <ActionIcon
        class="h-5 w-5 hover:text-foreground"
        icon={Heart}
        iconSize="h-4 w-4"
        onclick={() => cModelsStore.mToggleFavorite(option.model)}
        tooltip={fTranslate('text_7f3c0782afd7')}
        tooltipAsTitle
      />
    </span>
  {/if}
</div>
