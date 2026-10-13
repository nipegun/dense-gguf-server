<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelLoadControl from '../ModelLoadControl.svelte';
  import ModelsManagerDownloadControl from './ModelsManagerDownloadControl.svelte';
  import type { ModelRowDownloadState } from '$lib/enums';
  import { ServerModelStatus } from '$lib/enums';
  import { cModelsStore, cServerStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';

  interface Props {
    class?: string;
    /** Download state, when the row stands for a tracked download. */
    download?: ModelRowDownloadState | null;
    option: ModelOption;
  }

  let { class: className = '', download = null, option }: Props = $props();

  let status = $derived(cModelsStore.mGetModelStatus(option.model));
  let vIsOperationInProgress = $derived(cModelsStore.status.mIsOperationInProgress(option.model));
  let vIsLoaded = $derived(cModelsStore.mIsModelRunning(option.model));
  // a MODEL server serves the model it was started with and answers no load route
  let vIsRouter = $derived(cServerStore.isRouterMode);
</script>

{#if download}
  <ModelsManagerDownloadControl class="justify-self-center {className}" {option} state={download} />
{:else if vIsRouter}
  <ModelLoadControl
    class="justify-self-center {className}"
    isFailed={status === ServerModelStatus.FAILED}
    isLoaded={vIsLoaded}
    isLoading={status === ServerModelStatus.LOADING || vIsOperationInProgress}
    isSleeping={status === ServerModelStatus.cSleeping}
    {option}
  />
{:else}
  <span
    class="justify-self-center text-sm text-muted-foreground {className}"
    title={fTranslate('dgs_e74c1741e03f')}
  >
    -
  </span>
{/if}
