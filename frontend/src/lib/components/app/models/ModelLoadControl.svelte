<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ActionIcon } from '$lib/components/app';
  import { cIconClassDefault, cModelLoadIcons } from '$lib/constants';
  import { cModelsStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';

  interface Props {
    class?: string;
    isFailed?: boolean;
    isLoaded: boolean;
    isLoading?: boolean;
    isSleeping?: boolean;
    option: ModelOption;
    /** Table rows keep the load action visible, selector rows reveal it on hover. */
    revealOnHover?: boolean;
    /** Renders the state alone, for a caller that moves load and unload elsewhere. */
    showAction?: boolean;
  }

  let {
    class: className = '',
    isFailed = false,
    isLoaded,
    isLoading = false,
    isSleeping = false,
    option,
    revealOnHover = true,
    showAction = true
  }: Props = $props();
</script>

<div class={['flex w-5 shrink-0 items-center justify-center', className]}>
  {#if isLoading}
    <cModelLoadIcons.loading class="{cIconClassDefault} animate-spin text-muted-foreground" />
  {:else}
    <!-- the state dot is what the row shows at rest; the action takes its place on hover -->
    {#if showAction && revealOnHover}
      {#if isFailed}
        <cModelLoadIcons.failed
          class="h-3.5 w-3.5 text-red-500 group-hover:hidden [@media(pointer:coarse)]:hidden"
        />
      {:else}
        <span
          class="h-2 w-2 rounded-full group-hover:hidden [@media(pointer:coarse)]:hidden {isSleeping
            ? 'bg-orange-400'
            : isLoaded
              ? 'bg-green-500'
              : 'bg-muted-foreground/50'}"
        ></span>
      {/if}
    {:else}
      {#if isFailed}
        <cModelLoadIcons.failed class="h-3.5 w-3.5 text-red-500" />
      {:else}
        <span
          class="h-2 w-2 rounded-full {isSleeping
            ? 'bg-orange-400'
            : isLoaded
              ? 'bg-green-500'
              : 'bg-muted-foreground/50'}"
        ></span>
      {/if}
    {/if}

    <!-- the wrapper has to pick one display value: flex and hidden together
		     leave the winner to stylesheet order -->
    <div
      class={!showAction
        ? 'hidden'
        : revealOnHover
          ? 'hidden group-hover:flex [@media(pointer:coarse)]:flex'
          : 'flex'}
    >
      {#if isFailed}
        <ActionIcon
          class="h-5 w-5 text-red-500 hover:text-foreground"
          icon={cModelLoadIcons.retry}
          iconSize="h-4 w-4"
          onclick={() => cModelsStore.status.load(option.model)}
          stopPropagationOnClick
          tooltip={fTranslate('text_3890e4decfe9')}
          tooltipAsTitle
        />
      {:else if isLoaded || isSleeping}
        <ActionIcon
          class="h-5 w-5 hover:text-foreground"
          icon={cModelLoadIcons.unload}
          iconSize="h-4 w-4"
          onclick={() => cModelsStore.status.mUnload(option.model)}
          stopPropagationOnClick
          tooltip={fTranslate('text_c2ed103ced3e')}
          tooltipAsTitle
        />
      {:else}
        <ActionIcon
          class="h-5 w-5 hover:text-foreground"
          icon={cModelLoadIcons.load}
          iconSize="h-4 w-4"
          onclick={() => cModelsStore.status.load(option.model)}
          stopPropagationOnClick
          tooltip={fTranslate('text_889fad74ece5')}
          tooltipAsTitle
        />
      {/if}
    </div>
  {/if}
</div>
