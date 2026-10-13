<script lang="ts">
  import { cSettingsKeys } from '$lib/constants';
  import { cSettingsStore, cToolsStore } from '$lib/stores';
  import {
    fGetMentionBadgeIconPaths,
    fGetMentionBadgeLabel,
    cMentionBadgeClassname,
    cMentionBadgeIconClassname,
    cMentionBadgeSvgAttributes
  } from '$lib/utils';

  interface Props {
    name: string;
    path: string;
  }

  let { name, path }: Props = $props();

  let vShowFullPath = $derived(
    cSettingsStore.mGetConfig(cSettingsKeys.SHOW_FULL_PATH_IN_MENTIONS) as boolean
  );
  let vLabel = $derived(fGetMentionBadgeLabel(name, path, vShowFullPath, cToolsStore.serverHome));
</script>

<!-- The chip is a flex container, so template whitespace between its
     children collapses away and the icon keeps its `gap-1` spacing. -->
<span class={cMentionBadgeClassname} title={path}>
  <svg {...cMentionBadgeSvgAttributes} class={cMentionBadgeIconClassname}>
    {#each fGetMentionBadgeIconPaths(path) as vD (vD)}
      <path d={vD} />
    {/each}
  </svg>

  <span class="shrink-0 truncate">{vLabel}</span>
</span>
