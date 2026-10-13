<script lang="ts">
  import { MODEL_ID } from '$lib/constants';
  import { HuggingFaceService } from '$lib/services';
  import type { ModelOption } from '$lib/types/models';
  import { fNearViewport, fRepoOf } from '$lib/utils';
  import { fFormatContextLength } from '$lib/utils/formatters';

  interface Props {
    class?: string;
    /** Context the model is set to run with; paired with the supported one. */
    configured?: number | null;
    option: ModelOption;
  }

  let { class: className = '', configured = null, option }: Props = $props();

  // a listing that reports a context window (OpenRouter, Groq, HF) needs no lookup
  let vReported = $derived(option.contextLength ?? null);
  let vIsNearViewport = $state(false);
  let vFetched = $state<number | null>(null);

  $effect(() => {
    vFetched = null;

    if (vReported || !vIsNearViewport) return;

    // a local GGUF carries its trained context in the model metadata
    const cRepo = fRepoOf(option.model);

    // a local file path has no repo to read, and the Hub would answer 404
    if (!cRepo.includes(MODEL_ID.ORG_SEPARATOR)) return;

    let vCancelled = false;

    void HuggingFaceService.mGetDetails(cRepo)
      .then((pDetails) => {
        if (!vCancelled) vFetched = pDetails?.gguf?.context_length ?? null;
      })
      // best-effort: offline or a repo we cannot read keeps the dash
      .catch(() => {});

    return () => {
      vCancelled = true;
    };
  });

  let vContext = $derived(vReported ?? vFetched);
  let aValues = $derived(
    [configured, vContext].filter((pValue): pValue is number => typeof pValue === 'number')
  );
</script>

<span
  use:fNearViewport={() => (vIsNearViewport = true)}
  class={['text-sm text-muted-foreground', className]}
>
  {fFormatContextLength(aValues)}
</span>
