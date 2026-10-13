<script lang="ts">
  import MentionBadge from './MentionBadge.svelte';
  import { fSplitMentionSegments } from '$lib/utils';

  interface Props {
    content: string;
  }

  let { content }: Props = $props();

  let aSegments = $derived(fSplitMentionSegments(content));
</script>

<!-- Segments sit in a `whitespace-pre-wrap` parent, so the markup stays
     glued: any newline between the tags below would print as a space. -->
<!-- prettier-ignore -->
{#each aSegments as vSegment, vIndex (vIndex)}{#if vSegment.mention}<MentionBadge name={vSegment.mention.name} path={vSegment.mention.path} />{:else}{vSegment.text}{/if}{/each}
