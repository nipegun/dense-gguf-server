<script lang="ts">
  import ModelCapabilityIcons from './ModelCapabilityIcons.svelte';
  import { MODEL_ID } from '$lib/constants';
  import { ModelCapability } from '$lib/enums';
  import { HuggingFaceService } from '$lib/services';
  import type { ModelOption } from '$lib/types/models';
  import {
    fDetectThinkingSupport,
    fDetectToolUseSupport,
    fNearViewport,
    fRepoOf
  } from '$lib/utils';

  interface Props {
    class?: string;
    /** Hide the modality icons, keeping the capability ones. */
    hideModalities?: boolean;
    option: ModelOption;
  }

  let { class: className = '', hideModalities = false, option }: Props = $props();

  // a listing that declares its capabilities (Odgs-compatible backends) needs no lookup
  let aDeclared = $derived(option.capabilities);
  let vIsNearViewport = $state(false);
  let vTemplate = $state('');

  $effect(() => {
    vTemplate = '';

    if (!vIsNearViewport) return;

    // only the chat template says whether tools and reasoning work, and the Hub
    // carries it for a local GGUF whether the model is loaded or not
    const cRepo = fRepoOf(option.model);

    if (!cRepo.includes(MODEL_ID.ORG_SEPARATOR)) return;

    let vCancelled = false;

    void HuggingFaceService.mGetDetails(cRepo)
      .then((pDetails) => {
        if (!vCancelled) vTemplate = pDetails?.gguf?.chat_template ?? '';
      })
      // best-effort: offline or a repo we cannot read leaves the icons off
      .catch(() => {});

    return () => {
      vCancelled = true;
    };
  });

  let vSupportsThinking = $derived(
    aDeclared.includes(ModelCapability.REASONING) || fDetectThinkingSupport(vTemplate)
  );
  let vSupportsToolUse = $derived(
    aDeclared.includes(ModelCapability.cToolUse) || fDetectToolUseSupport(vTemplate)
  );
</script>

<span use:fNearViewport={() => (vIsNearViewport = true)} class={className}>
  <ModelCapabilityIcons
    {hideModalities}
    modalities={option.modalities}
    supportsThinking={vSupportsThinking}
    supportsToolUse={vSupportsToolUse}
  />
</span>
