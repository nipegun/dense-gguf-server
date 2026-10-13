<script lang="ts">
  import { cModelVariantBadgeClass } from '$lib/constants';
  import type { ModelSidecarBadge } from '$lib/types/models';
  import { fIsAuxSidecar } from '$lib/utils';
  import { SvelteSet } from 'svelte/reactivity';

  interface Props {
    /** Draft sidecars available for the model, each with its own quant. */
    draftSidecars?: ModelSidecarBadge[];
  }

  let { draftSidecars = [] }: Props = $props();

  // the router listing and the --model-draft args both report a model's drafts, so
  // the same sidecar can arrive twice
  let aBadges = $derived.by(() => {
    const cSeen = new SvelteSet<string>();

    return draftSidecars.filter((pBadge) => {
      if (fIsAuxSidecar(pBadge.kind)) return false;

      const cKey = `${pBadge.kind}:${pBadge.quant ?? ''}`;

      if (cSeen.has(cKey)) return false;

      cSeen.add(cKey);

      return true;
    });
  });
</script>

{#each aBadges as vBadge (vBadge.kind + (vBadge.quant ?? ''))}
  <span aria-hidden="true" class="text-muted-foreground">+</span>

  <span
    class={cModelVariantBadgeClass}
    title="{vBadge.kind.toUpperCase()} draft model{vBadge.quant ? ` at ${vBadge.quant}` : ''}"
  >
    {vBadge.kind}
  </span>
{/each}
