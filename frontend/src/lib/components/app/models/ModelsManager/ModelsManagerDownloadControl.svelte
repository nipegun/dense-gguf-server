<script lang="ts">
  import { fDownloadProgressFor } from './utils';
  import { ActionIcon } from '$lib/components/app';
  import { cModelDownloadIcons } from '$lib/constants';
  import { ModelRowDownloadState } from '$lib/enums';
  import { cModelsStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';

  interface Props {
    class?: string;
    option: ModelOption;
    /** Download state the row stands for. */
    state: ModelRowDownloadState;
  }

  let { class: className = '', option, state }: Props = $props();

  let vIsPaused = $derived(state === ModelRowDownloadState.cPaused);
  let vProgress = $derived(fDownloadProgressFor(option.model));
  let vPercent = $derived(
    vProgress && vProgress.totalBytes > 0
      ? Math.round((vProgress.downloadedBytes / vProgress.totalBytes) * 100)
      : null
  );
</script>

<!-- the percent is what the row shows at rest; the action takes its place on hover,
     the way the load control does in the other sections -->
<div class={['flex items-center justify-center', className]}>
  <span
    class="text-xs text-muted-foreground tabular-nums group-hover:hidden [@media(pointer:coarse)]:hidden"
  >
    {vPercent !== null ? `${vPercent}%` : vIsPaused ? 'Paused' : ''}
  </span>

  <div class="hidden group-hover:flex [@media(pointer:coarse)]:flex">
    <ActionIcon
      class="h-5 w-5 hover:text-foreground"
      icon={vIsPaused ? cModelDownloadIcons.resume : cModelDownloadIcons.pause}
      iconSize="h-4 w-4"
      onclick={() =>
        void (vIsPaused
          ? cModelsStore.status.mDownloadModel(option.model)
          : cModelsStore.status.mPauseDownload(option.model))}
      stopPropagationOnClick
      tooltip={vIsPaused ? 'Resume downloading' : 'Pause downloading'}
      tooltipAsTitle
    />
  </div>
</div>
