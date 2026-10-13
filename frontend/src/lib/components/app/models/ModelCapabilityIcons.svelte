<script lang="ts">
  import * as Tooltip from '$lib/components/ui/tooltip';
  import {
    dCapabilityIcons,
    dCapabilityLabels,
    dModalityFlagKeys,
    cModalityIcons,
    cModalityLabels,
    aModalityOrder
  } from '$lib/constants';
  import { ModelCapability } from '$lib/enums';
  import type { ModelModalities } from '$lib/types/models';

  interface Props {
    modalities?: ModelModalities;
    supportsThinking?: boolean;
    supportsToolUse?: boolean;
    hideCapabilities?: boolean;
    hideModalities?: boolean;
    iconSize?: string;
    gapClass?: string;
  }

  let {
    gapClass = 'gap-1.25',
    hideCapabilities = false,
    hideModalities = false,
    iconSize = 'h-3 w-3',
    modalities,
    supportsThinking = false,
    supportsToolUse = false
  }: Props = $props();

  let aCapabilities = $derived([
    ...(supportsToolUse ? [ModelCapability.cToolUse] : []),
    ...(supportsThinking ? [ModelCapability.REASONING] : [])
  ]);

  let aShownModalities = $derived(
    aModalityOrder.filter((pModality) => modalities?.[dModalityFlagKeys[pModality]])
  );

  // an icon-less box still takes the gap where its icons would sit, so a caller with
  // nothing to show renders no box at all
  let vHasIcons = $derived(
    (!hideCapabilities && aCapabilities.length > 0) ||
      (aShownModalities.length > 0 && !hideModalities)
  );
</script>

{#if vHasIcons}
  <span class="inline-flex items-center {gapClass}">
    {#if !hideCapabilities}
      {#each aCapabilities as vCapability (vCapability)}
        {@const Icon = dCapabilityIcons[vCapability]}

        <Tooltip.fRoot>
          <Tooltip.Trigger>
            <Icon class="{iconSize} text-muted-foreground" />
          </Tooltip.Trigger>

          <Tooltip.Content>
            <p>{dCapabilityLabels[vCapability]}</p>
          </Tooltip.Content>
        </Tooltip.fRoot>
      {/each}
    {/if}

    {#if aShownModalities.length > 0 && !hideModalities}
      <span class="inline-flex items-center {gapClass} text-muted-foreground">
        {#each aShownModalities as vModality (vModality)}
          {@const Icon = cModalityIcons[vModality]}

          <Tooltip.fRoot>
            <Tooltip.Trigger>
              <Icon class={iconSize} />
            </Tooltip.Trigger>

            <Tooltip.Content>
              <p>{cModalityLabels[vModality]}</p>
            </Tooltip.Content>
          </Tooltip.fRoot>
        {/each}
      </span>
    {/if}
  </span>
{/if}
