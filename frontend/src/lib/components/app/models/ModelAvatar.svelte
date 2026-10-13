<script lang="ts">
  import ModelOrgAvatar from './ModelOrgAvatar.svelte';
  import { cHfBaseModelTagRegex, cSettingsKeys } from '$lib/constants';
  import { HuggingFaceService, ModelsService } from '$lib/services';
  import { cSettingsStore } from '$lib/stores';
  import type { ModelOption } from '$lib/types/models';
  import { fNearViewport, fOrgOf } from '$lib/utils';
  import type { Snippet } from 'svelte';

  interface Props {
    class?: string;
    /** Rendered when the model has neither an org avatar nor a provider mark. */
    fallback?: Snippet;
    option: ModelOption;
    quantPositionClass?: string;
    quantSize?: string;
    /** Show the base model's org as the main image, the repo (quantizer) org as the
     *  corner badge. The base org costs one Hub request per repo. Left unset with
     *  {@link showRepoOrgAvatar}, the row follows the family grouping. */
    showBaseModelAvatar?: boolean;
    /** Show the repo's own org as the main image, skipping the base model.
     *  Used inside a heading that already carries the base org. */
    showRepoOrgAvatar?: boolean;
    /** Keep the quantizer badge off, e.g. when a row stands for a whole family. */
    showQuantBadge?: boolean;
    size?: string;
  }

  let {
    class: className = '',
    fallback,
    option,
    quantPositionClass = '-bottom-1 -right-1',
    quantSize = 'h-3 w-3',
    showBaseModelAvatar,
    showQuantBadge = true,
    showRepoOrgAvatar,
    size = 'size-5'
  }: Props = $props();

  let vParsedId = $derived(ModelsService.mParseModelId(option.model));
  let vOrgName = $derived(vParsedId.orgName);
  // A row names neither flag: a family heading already carries the base org, so a
  // grouped list shows the repo's own org, and an ungrouped one the base org with the
  // quantizer badge. A caller that names one of the two keeps control of the avatar.
  let vFollowsGrouping = $derived(
    showBaseModelAvatar === undefined && showRepoOrgAvatar === undefined
  );
  let vGroupedByFamily = $derived(
    cSettingsStore.vConfig[cSettingsKeys.GROUP_MODELS_BY_FAMILY] ?? false
  );
  let vBaseOrgMain = $derived(showBaseModelAvatar ?? (vFollowsGrouping && !vGroupedByFamily));
  let vRepoOrgMain = $derived(showRepoOrgAvatar ?? (vFollowsGrouping && vGroupedByFamily));
  // Avatars come from the Hub, so with the metadata setting off they are
  // hidden entirely and no base-model lookup runs.
  let vHubEnabled = $derived(HuggingFaceService.mIsEnabled());
  let vTagBaseModel = $derived(
    (option.tags ?? [])
      .find((pT) => cHfBaseModelTagRegex.test(pT))
      ?.match(cHfBaseModelTagRegex)?.[1] ?? null
  );
  // long lists mount hundreds of avatars at once, so the base org lookup waits
  // until the row is near the viewport
  let vFetchedBaseModelOrg = $state<string | null>(null);
  let vBaseModelOrg = $derived(fOrgOf(vTagBaseModel) || vFetchedBaseModelOrg);
  let vIsNearViewport = $state(false);

  $effect(() => {
    vFetchedBaseModelOrg = null;

    if (!vIsNearViewport || !vBaseOrgMain || !vOrgName || vTagBaseModel) return;

    if (!vHubEnabled) return;

    let vCancelled = false;

    void HuggingFaceService.mGetBaseModel(option.model)
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

{#if vOrgName && vHubEnabled}
  <span
    use:fNearViewport={() => (vIsNearViewport = true)}
    class={['inline-flex shrink-0', className]}
  >
    <ModelOrgAvatar
      class="mt-0"
      org={vRepoOrgMain ? vOrgName : (vBaseModelOrg ?? vOrgName)}
      quantOrg={vBaseOrgMain && showQuantBadge && !vRepoOrgMain ? vOrgName : undefined}
      {quantPositionClass}
      {quantSize}
      {size}
    />
  </span>
{:else}
  {@render fallback?.()}
{/if}
