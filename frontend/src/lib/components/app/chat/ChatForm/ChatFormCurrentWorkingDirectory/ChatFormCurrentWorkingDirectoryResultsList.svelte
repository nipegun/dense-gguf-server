<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Folder } from '@lucide/svelte';
  import { fCn } from '$lib/components/ui/utils';
  import { cUiDataAttrs } from '$lib/constants';
  import { fHighlightMatch } from '$lib/utils';
  import { fly } from 'svelte/transition';

  // Fly-in transition for the results list.
  const cFlyYPx = -4;
  const cFlyDurationMs = 100;

  interface Props {
    results: string[];
    hoveredIndex: number;
    isSearching: boolean;
    error: string | null;
    rawQuery: string;
    container?: HTMLDivElement | null;
    onCommit?: (pPath: string) => void;
    onHover?: (pIndex: number) => void;
  }

  let {
    container = $bindable(null),
    error,
    hoveredIndex,
    isSearching,
    onCommit,
    onHover,
    rawQuery,
    results
  }: Props = $props();
</script>

<div
  bind:this={container}
  transition:fly={{ duration: cFlyDurationMs, y: cFlyYPx }}
  class="max-h-48 overflow-y-auto py-2"
>
  {#if isSearching && results.length === 0}
    <div class="px-2 py-1.5 text-sm text-muted-foreground">{fTranslate('text_78c9d9f6ace0')}</div>
  {:else if error}
    <div class="px-2 py-1.5 text-sm text-destructive">{error}</div>
  {:else if results.length === 0}
    <div class="px-2 py-1.5 text-sm text-muted-foreground">{fTranslate('dgs_dcbb554e995b')}</div>
  {:else}
    {#each results as vPath, vIndex (vPath)}
      <button
        type="button"
        {...{ [cUiDataAttrs.RESULT_INDEX]: vIndex }}
        class={fCn(
          'relative flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground'
        )}
        data-highlighted={vIndex === hoveredIndex ? '' : undefined}
        onclick={() => onCommit?.(vPath)}
        onmouseenter={() => onHover?.(vIndex)}
      >
        <Folder class="size-4 shrink-0 text-muted-foreground" />

        <span class="min-w-0 flex-1 truncate font-mono text-left">
          {#each fHighlightMatch(vPath, rawQuery.trim()) as vSeg, vSegIndex (vSegIndex)}
            {#if vSeg.match}
              <mark class="rounded bg-yellow-200/60 px-0.5 text-foreground dark:bg-yellow-500/30"
                >{vSeg.text}</mark
              >
            {:else}
              {vSeg.text}
            {/if}
          {/each}
        </span>
      </button>
    {/each}
  {/if}
</div>
