<script lang="ts">
  import ModelAvatar from '../ModelAvatar.svelte';
  import ModelCapabilities from '../ModelCapabilities.svelte';
  import ModelContext from '../ModelContext.svelte';
  import ModelId from '../ModelId.svelte';
  import type { ModelQuantGroup } from './utils';
  import { fConfiguredContext } from './utils';
  import { ChevronDown, ChevronUp } from '@lucide/svelte';
  import { cModelRowGridClass, cModelRowTrailingCellClass } from '$lib/constants';
  import { KeyboardKey, ModelGroupKind } from '$lib/enums';
  import { cDeviceStore, cModelsStore } from '$lib/stores';

  interface Props {
    entry: ModelQuantGroup;
    expanded: boolean;
    onToggle: () => void;
    /** Left padding in px, from the nesting depth. */
    indent?: number;
  }

  let { entry, expanded, indent = 0, onToggle }: Props = $props();

  let vGroupLabel = $derived(
    entry.kind === ModelGroupKind.cVariants
      ? `${entry.quants.length} variants`
      : `${entry.quants.length} quants available`
  );
  let vAnyLoaded = $derived(
    entry.quants.some((pQuant) => cModelsStore.mIsModelRunning(pQuant.model))
  );
  // a repo row stands for its quants, so it reports what they agree on
  let vContextSource = $derived(entry.quants.find((pQuant) => pQuant.contextLength) ?? entry.base);
  let vMediaSource = $derived(entry.quants.find((pQuant) => pQuant.modalities) ?? entry.base);

  function fHandleKeydown(pEvent: KeyboardEvent): void {
    if (pEvent.key === KeyboardKey.cSpace) pEvent.preventDefault();

    if (pEvent.key === KeyboardKey.cEnter || pEvent.key === KeyboardKey.cSpace) onToggle();
  }
</script>

<div
  class={[
    cModelRowGridClass,
    'relative cursor-pointer rounded-md px-2 py-2.5 transition hover:bg-muted/40 max-md:px-3 max-md:py-3.5'
  ]}
  onclick={onToggle}
  onkeydown={fHandleKeydown}
  role="button"
  tabindex="0"
>
  <span class="flex min-w-0 items-center gap-3 max-md:pr-9" style="padding-left: {indent}px">
    <ModelAvatar option={entry.base} size="size-9" />

    <span class="min-w-0">
      <span class="flex min-w-0 items-center gap-1.25">
        <ModelId
          aliases={entry.base.aliases}
          class="min-w-0"
          hideCapabilities
          hideModalities
          hideQuantization
          modalities={vMediaSource.modalities}
          modelId={entry.base.model}
          tags={entry.base.tags}
          title={entry.base.model}
        />

        <ModelCapabilities hideModalities={cDeviceStore.isMobile} option={entry.base} />
      </span>

      <span class="block text-xs text-muted-foreground">{vGroupLabel}</span>
    </span>
  </span>

  <ModelContext
    class="justify-self-end max-md:hidden"
    configured={fConfiguredContext(vContextSource)}
    option={vContextSource}
  />

  <span class="justify-self-center max-md:hidden">
    <span
      class="block h-2.5 w-2.5 rounded-full {vAnyLoaded
        ? 'bg-emerald-500'
        : 'border border-muted-foreground/50'}"
    ></span>
  </span>

  <span class="flex justify-center {cModelRowTrailingCellClass}">
    {#if expanded}
      <ChevronUp class="h-3.5 w-3.5 text-muted-foreground" />
    {:else}
      <ChevronDown class="h-3.5 w-3.5 text-muted-foreground" />
    {/if}
  </span>
</div>
