<script lang="ts">
  import ModelAvatar from '../ModelAvatar.svelte';
  import { ModelIcon } from '$lib/constants';
  import type { ModelOption } from '$lib/types/models';

  interface Props {
    class?: string;
    /** Selected model; its avatar decides the mark, the generic icon when absent. */
    option?: ModelOption | null;
  }

  let { class: className = 'h-3.5 w-3.5', option }: Props = $props();
</script>

{#if option}
  <ModelAvatar class={className} {option} showQuantBadge={false} showRepoOrgAvatar size={className}>
    {#snippet fallback()}
      <ModelIcon class={className} />
    {/snippet}
  </ModelAvatar>
{:else}
  <ModelIcon class={className} />
{/if}
