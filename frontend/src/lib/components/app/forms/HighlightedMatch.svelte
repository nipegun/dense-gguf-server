<script lang="ts">
  import { fHighlightMatch } from '$lib/utils';

  interface Props {
    text: string;
    query: string;
    matchClass?: string;
  }

  let {
    matchClass = 'rounded bg-yellow-200/60 px-0.5 text-foreground dark:bg-yellow-500/30',
    query,
    text
  }: Props = $props();

  let aSegments = $derived(fHighlightMatch(text, query));
</script>

{#each aSegments as vSeg, vI (vI)}
  {#if vSeg.match}
    <mark class={matchClass}>{vSeg.text}</mark>
  {:else}
    {vSeg.text}
  {/if}
{/each}
