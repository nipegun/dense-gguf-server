<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ScrollCarousel } from '$lib/components/app';
  import * as Select from '$lib/components/ui/select';
  import * as ToggleGroup from '$lib/components/ui/toggle-group';
  import {
    dCapabilityIcons,
    dCapabilityLabels,
    cFilterToggleItemClass,
    cFilterTriggerClass,
    dModalityFlagKeys,
    cModalityIcons,
    aModalityKeys,
    cModalityLabels,
    aModalityOrder,
    type ModalityKey
  } from '$lib/constants';
  import { ModelCapability } from '$lib/enums';
  import { cModelsStore } from '$lib/stores';
  import { SvelteSet } from 'svelte/reactivity';

  interface Props {
    /** Capabilities a model must have every one of. */
    capabilities?: ModelCapability[];
    /** Smallest context a model must support; 0 keeps every model. */
    contextLimit?: number;
    /** Modalities a model must support at least one of. */
    modalities?: ModalityKey[];
  }

  let {
    capabilities = $bindable<ModelCapability[]>([]),
    contextLimit = $bindable(0),
    modalities = $bindable<ModalityKey[]>([])
  }: Props = $props();

  const aContextSteps: { label: string; value: number }[] = [
    { label: fTranslate('dgs_6aa0434a3833'), value: 0 },
    { label: fTranslate('dgs_7d946727e343'), value: 8192 },
    { label: fTranslate('dgs_5b818adecd4c'), value: 32768 },
    { label: fTranslate('dgs_ebca10cba03a'), value: 131_072 },
    { label: fTranslate('dgs_9039f7d5c946'), value: 262_144 },
    { label: fTranslate('dgs_c15bb3e15c4c'), value: 1_048_576 }
  ];

  const aCapabilityToggles = [ModelCapability.cToolUse, ModelCapability.REASONING].map(
    (pValue) => ({
      icon: dCapabilityIcons[pValue],
      label: dCapabilityLabels[pValue],
      value: pValue
    })
  );

  // only the modalities the models at hand actually carry: a toggle for something
  // none of them supports could only ever empty the table
  let vDetectedModalities = $derived.by(() => {
    const cKeys = new SvelteSet<ModalityKey>();

    for (const cOption of cModelsStore.aModels) {
      for (const cKey of aModalityKeys) {
        if (cOption.modalities?.[cKey]) cKeys.add(cKey);
      }
    }

    return cKeys;
  });
  let aModalityToggles = $derived(
    aModalityOrder
      .filter((pModality) => vDetectedModalities.has(dModalityFlagKeys[pModality]))
      .map((pModality) => ({
        icon: cModalityIcons[pModality],
        label: cModalityLabels[pModality],
        value: dModalityFlagKeys[pModality]
      }))
  );

  // one group holds what a model can do and what it can accept
  let aToggles = $derived([...aCapabilityToggles, ...aModalityToggles]);
  const sCapabilityValues = new Set<string>(aCapabilityToggles.map((pEntry) => pEntry.value));
  // every flag value, so a choice made before a model disappeared still round-trips
  const sModalityValues = new Set<string>(aModalityKeys);

  // the group holds one flat list, so a change splits back into the two filters
  function fSetToggles(pValues: string[]): void {
    capabilities = pValues.filter((pValue): pValue is ModelCapability =>
      sCapabilityValues.has(pValue)
    );
    modalities = pValues.filter((pValue): pValue is ModalityKey => sModalityValues.has(pValue));
  }

  let vContextLabel = $derived(
    aContextSteps.find((pStep) => pStep.value === contextLimit)?.label ?? aContextSteps[0].label
  );
</script>

<!-- below md the carousel keeps a row of its own: a toolbar that also holds a call to
     action would otherwise squeeze the filters out of sight -->
<ScrollCarousel
  alwaysShowArrows
  class="min-w-0 flex-1 max-md:w-full max-md:flex-none"
  gapSize="2"
  innerClass="items-center"
>
  <Select.fRoot
    onValueChange={(pValue) => (contextLimit = Number(pValue))}
    type="single"
    value={String(contextLimit)}
  >
    <Select.Trigger class={cFilterTriggerClass} size="sm">
      <span class="text-muted-foreground">{fTranslate('dgs_e1de60d6ce24')}</span>

      {vContextLabel}
    </Select.Trigger>

    <Select.Content>
      {#each aContextSteps as vStep (vStep.value)}
        <Select.Item label={vStep.label} value={String(vStep.value)}>{vStep.label}</Select.Item>
      {/each}
    </Select.Content>
  </Select.fRoot>

  <ToggleGroup.Root
    class="border border-border/30 bg-muted/60 shadow-sm dark:border-border/20 dark:bg-muted/75"
    onValueChange={fSetToggles}
    type="multiple"
    value={[...capabilities, ...modalities]}
    variant="outline"
  >
    {#each aToggles as vToggle (vToggle.value)}
      <ToggleGroup.Item
        aria-label={vToggle.label}
        class={cFilterToggleItemClass}
        title={vToggle.label}
        value={vToggle.value}
      >
        <vToggle.icon class="h-3.5 w-3.5" />
      </ToggleGroup.Item>
    {/each}
  </ToggleGroup.Root>
</ScrollCarousel>
