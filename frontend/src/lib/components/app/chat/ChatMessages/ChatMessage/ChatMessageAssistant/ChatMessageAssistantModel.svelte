<script lang="ts">
  import { ModelBadge, ModelsSelector } from '$lib/components/app';
  import { ServerModelStatus } from '$lib/enums';
  import { cModelsStore } from '$lib/stores';
  import { fCopyToClipboard } from '$lib/utils';

  interface Props {
    displayedModel: string | null;
    isRouter: boolean;
    isLoading: boolean;
    onRegenerate: (pModelOverride?: string) => void;
  }

  let { displayedModel, isLoading, isRouter, onRegenerate }: Props = $props();

  function fHandleCopyModel() {
    void fCopyToClipboard(displayedModel ?? '');
  }
</script>

{#if isRouter}
  <ModelsSelector
    currentModel={displayedModel}
    disabled={isLoading}
    onModelChange={async (pModelId: string, pModelName: string) => {
      const cStatus = cModelsStore.mGetModelStatus(pModelId);

      // the picker's own selection applies before this resolves, so a model
      // that is not loaded yet gets its load request in first
      if (cStatus !== ServerModelStatus.cLoaded) {
        await cModelsStore.status.load(pModelId);
      }

      onRegenerate(pModelName);

      return true;
    }}
  />
{:else}
  <ModelBadge model={displayedModel || undefined} onclick={fHandleCopyModel} />
{/if}
