<script lang="ts">
  import { cModalityIcons, cModalityLabels } from '$lib/constants';
  import { ModelModality } from '$lib/enums';

  interface Props {
    modalities: ModelModality[];
    class?: string;
  }

  let { class: className = '', modalities }: Props = $props();

  const aShownModalities = [
    ModelModality.VISION,
    ModelModality.AUDIO,
    ModelModality.VIDEO
  ] as const;

  let aVisible = $derived(aShownModalities.filter((pModality) => modalities.includes(pModality)));
</script>

{#each aVisible as vModality (vModality)}
  {@const ModalityIcon = cModalityIcons[vModality]}
  <span
    class={[
      'inline-flex items-center gap-1 rounded-md bg-muted px-2 py-1 text-xs font-medium',
      className
    ]}
  >
    <ModalityIcon class="h-3 w-3" />

    {cModalityLabels[vModality]}
  </span>
{/each}
