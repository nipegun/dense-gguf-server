<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelDownloadProgressBar from '../ModelDownloadProgressBar.svelte';
  import ModelOrgAvatar from '../ModelOrgAvatar.svelte';
  import { Loader2, Pause, Play, X } from '@lucide/svelte';
  import { ActionIcon, ModelId } from '$lib/components/app';
  import { HuggingFaceService, ModelsService } from '$lib/services';
  import { cModelsStore } from '$lib/stores';
  import type { ModelDownloadEntry } from '$lib/types';

  interface Props {
    /** One entry from the status feed: an in-flight or paused download. */
    entry: ModelDownloadEntry;
    /** Ask the list to confirm cancelling; the list owns the single dialog. */
    onRequestCancel?: (pRepoWithTag: string) => void;
    /** Show the organization name in the repo id, as the other selector rows do. */
    showOrgName?: boolean;
  }

  let { entry, onRequestCancel, showOrgName = true }: Props = $props();

  let vPercent = $derived(
    entry.progress && entry.progress.totalBytes > 0
      ? Math.round((entry.progress.downloadedBytes / entry.progress.totalBytes) * 100)
      : null
  );

  let vActionText = $derived(entry.isPaused ? 'Resume downloading' : 'Pause downloading');

  // the base model org resolves lazily via HF when unknown
  let vOrgName = $derived(ModelsService.mParseModelId(entry.repoWithTag).orgName);
  let vFetchedBaseModelOrg = $state<string | null>(null);

  $effect(() => {
    const cRepoWithTag = entry.repoWithTag;

    vFetchedBaseModelOrg = null;

    // avatars hidden with the Hub metadata setting off: no lookups at all
    if (!HuggingFaceService.mIsEnabled()) return;

    let vCancelled = false;

    void HuggingFaceService.mGetBaseModel(cRepoWithTag)
      .then((pBase) => {
        if (!vCancelled && pBase?.org) vFetchedBaseModelOrg = pBase.org;
      })
      // best-effort lookup: offline or unknown repos keep the repo org
      .catch(() => {});

    return () => {
      vCancelled = true;
    };
  });
</script>

<!-- One in-flight download; same actions as the discover quant chips: the row
     itself pauses / resumes, the trailing X cancels (stops and discards the
     partial files). Both affordances fade in on hover, the slots are reserved
     so the list never reflows. -->
<div
  class="group relative flex items-center gap-2 rounded-sm p-2 text-left text-sm transition-colors hover:bg-accent/50"
>
  <button
    aria-label={vActionText}
    class="flex min-w-0 flex-1 cursor-pointer items-center gap-2 text-left"
    onclick={() => {
      if (entry.isPaused) {
        void cModelsStore.status
          .mDownloadModel(entry.repoWithTag)
          .catch((pError) => console.warn('Failed to resume the download:', pError));
      } else {
        void cModelsStore.status.mPauseDownload(entry.repoWithTag);
      }
    }}
    type="button"
  >
    {#if vOrgName}
      <ModelOrgAvatar
        org={vFetchedBaseModelOrg ?? vOrgName}
        quantOrg={vOrgName}
        quantPositionClass="-bottom-1 -right-1"
        quantSize="h-3 w-3"
        size="h-6 w-6"
      />
    {/if}

    <ModelId class="flex-1" hideOrgName={!showOrgName} modelId={entry.repoWithTag} showRawTooltip />

    {#if vPercent !== null}
      <span class="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">{vPercent}%</span>
    {:else if entry.isPaused}
      <span class="shrink-0 text-xs text-muted-foreground">{fTranslate('dgs_e159b06187d3')}</span>
    {/if}

    <!-- status action: spinner -> pause on hover while in flight, play on hover
				 when paused; opacity only for the spinner, the spin owns the transform -->
    <span class="relative inline-flex h-5 w-5 shrink-0 items-center justify-center">
      {#if entry.isPaused}
        <Play
          class="absolute h-4 w-4 scale-75 opacity-0 transition-[opacity,transform] duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:scale-100 group-hover:opacity-100 [@media(pointer:coarse)]:scale-100 [@media(pointer:coarse)]:opacity-100"
        />
      {:else}
        <Loader2
          class="absolute h-4 w-4 animate-spin text-muted-foreground transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-0 [@media(pointer:coarse)]:hidden"
        />

        <Pause
          class="absolute h-4 w-4 opacity-0 transition-opacity duration-150 ease-[cubic-bezier(0.23,1,0.32,1)] group-hover:opacity-100 [@media(pointer:coarse)]:opacity-100"
        />
      {/if}
    </span>
  </button>

  <!-- the row actions keep the sizing and reveal of the other selector rows -->
  <ActionIcon
    ariaLabel="Cancel downloading"
    class="pointer-events-none h-5 w-5 shrink-0 opacity-0 transition-opacity group-hover:pointer-events-auto group-hover:opacity-100 hover:text-destructive [@media(pointer:coarse)]:pointer-events-auto [@media(pointer:coarse)]:opacity-100"
    icon={X}
    iconSize="h-4 w-4"
    onclick={() => onRequestCancel?.(entry.repoWithTag)}
  />

  {#if entry.progress && entry.progress.totalBytes > 0}
    <ModelDownloadProgressBar
      downloadedBytes={entry.progress.downloadedBytes}
      overlay
      totalBytes={entry.progress.totalBytes}
    />
  {/if}
</div>
