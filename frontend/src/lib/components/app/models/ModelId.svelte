<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ModelCapabilityIcons from './ModelCapabilityIcons.svelte';
  import ModelDraftSidecars from './ModelDraftSidecars.svelte';
  import { Database, ScrollText } from '@lucide/svelte';
  import { TruncatedText } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cModelBadgeClass, cModelVariantBadgeClass } from '$lib/constants';
  import { HuggingFaceService } from '$lib/services';
  import { ModelsService } from '$lib/services/models.service';
  import { cSettingsStore } from '$lib/stores';
  import type { ModelModalities } from '$lib/types/models';
  import { type ModelSidecarBadge } from '$lib/types/models';
  import { fIsAuxSidecar } from '$lib/utils';
  import { fFormatParameters } from '$lib/utils';

  interface Props {
    modelId: string;
    hideOrgName?: boolean;
    hideName?: boolean;
    hideModalities?: boolean;
    hideCapabilities?: boolean;
    hideParameters?: boolean;
    showRaw?: boolean;
    showRawTooltip?: boolean;
    hideQuantization?: boolean;
    hideTags?: boolean;
    aliases?: string[];
    tags?: string[];
    /** Render the capability/modality/context icons on a second row. */
    iconsOnNewLine?: boolean;
    modalities?: ModelModalities;
    supportsThinking?: boolean;
    supportsToolUse?: boolean;
    /** Native title for the root element; keeps long lists light where a floating tooltip per row is too costly. */
    title?: string;
    /** Context length in tokens; renders a context icon when set. */
    contextLength?: number;
    /** Min/max GGUF file size (main + draft) across quants; renders a range when set. */
    sizeRange?: { min: number; max: number } | null;
    /** Draft sidecars available for the model, badged with their own quant. */
    draftSidecars?: ModelSidecarBadge[];
    /** Give the id a row of its own, with its badges and icons together under it. */
    stackId?: boolean;
    /** Allow badges to wrap onto new lines instead of truncating. */
    wrap?: boolean;
    class?: string;
  }

  let {
    aliases,
    class: className = '',
    contextLength,
    draftSidecars = [],
    hideCapabilities = false,
    hideModalities = false,
    hideName = false,
    hideOrgName = false,
    hideParameters = false,
    hideQuantization,
    hideTags,
    iconsOnNewLine = false,
    modalities,
    modelId,
    showRaw = undefined,
    showRawTooltip = false,
    sizeRange,
    stackId = false,
    supportsThinking = false,
    supportsToolUse = false,
    tags,
    title,
    wrap = false,
    ...rest
  }: Props = $props();

  const cBadgeClass = cModelBadgeClass;
  const cTagBadgeClass =
    'inline-flex w-fit shrink-0 items-center justify-center whitespace-nowrap rounded-md border border-border/50 px-1 py-0 text-[10px] font-mono text-foreground [a&]:hover:bg-accent [a&]:hover:text-accent-foreground';
  const cVariantBadgeClass = cModelVariantBadgeClass;

  /** Alias badges beyond this many collapse into a single `+x more` badge. */
  const cMaxAliasBadges = 2;

  let vParsed = $derived(ModelsService.mParseModelId(modelId));
  let vResolvedShowRaw = $derived(
    showRaw ?? (cSettingsStore.vConfig.showRawModelNames as boolean) ?? false
  );
  let vResolvedHideQuantization = $derived(
    hideQuantization ?? !cSettingsStore.vConfig.showModelQuantization
  );
  let vResolvedHideTags = $derived(hideTags ?? !cSettingsStore.vConfig.showModelTags);

  let aUniqueAliases = $derived([...new Set(aliases ?? [])]);
  let aUniqueTags = $derived([...new Set([...(vParsed.tags ?? []), ...(tags ?? [])])]);
  let aUniqueDraftSidecars = $derived(
    draftSidecars.filter((pBadge) => !fIsAuxSidecar(pBadge.kind))
  );

  let vPrimaryAlias = $derived(aUniqueAliases.length === 1 ? aUniqueAliases[0] : null);
  let vDisplayName = $derived(vPrimaryAlias ?? vParsed.modelName ?? modelId);

  let vHasBadges = $derived(
    vParsed.sidecar ||
      aUniqueDraftSidecars.length > 0 ||
      (vParsed.params && !hideParameters) ||
      (vParsed.quantization && !vResolvedHideQuantization) ||
      vPrimaryAlias ||
      aUniqueAliases.length > 1 ||
      (aUniqueTags.length > 0 && !vResolvedHideTags)
  );
</script>

{#if vResolvedShowRaw}
  <TruncatedText class="font-medium {className}" showTooltip={false} text={modelId} {...rest} />
{:else}
  {#snippet nameAndBadges()}
    {#if !hideName}
      <span class="min-w-0 truncate font-medium {stackId ? 'basis-full' : ''}">
        {#if !hideOrgName && vParsed.orgName}{vParsed.orgName}/{/if}{vDisplayName}
      </span>
    {/if}

    {#if vHasBadges}
      <!-- the badges keep their width, so a long id truncates around them instead of
			     a label being cut in half -->
      <span
        class="inline-flex min-w-0 items-center gap-1 overflow-hidden {wrap
          ? 'flex-wrap'
          : 'shrink-0'}"
      >
        {#if vParsed.sidecar}
          <span class={cVariantBadgeClass} title={`${vParsed.sidecar.toUpperCase()} draft model`}>
            {vParsed.sidecar}
          </span>
        {/if}

        {#if vParsed.params && !hideParameters}
          <span class={cBadgeClass}>
            {vParsed.params}{vParsed.activatedParams ? `-${vParsed.activatedParams}` : ''}
          </span>
        {/if}

        {#if vParsed.quantization && !vResolvedHideQuantization}
          <span class={cBadgeClass}>
            {vParsed.quantization}
          </span>
        {/if}

        <ModelDraftSidecars {draftSidecars} />

        {#if vPrimaryAlias}
          {#if vPrimaryAlias !== vParsed.modelName}
            <span class="{cBadgeClass} max-w-32 truncate" title={vParsed.modelName ?? modelId}>
              {vParsed.modelName ?? modelId}
            </span>
          {/if}
        {:else if aUniqueAliases.length > 1}
          {#each aUniqueAliases.slice(0, cMaxAliasBadges) as vAlias (vAlias)}
            <span class="{cBadgeClass} max-w-32 truncate" title={vAlias}>
              {vAlias}
            </span>
          {/each}

          {#if aUniqueAliases.length > cMaxAliasBadges}
            <span class={cBadgeClass} title={aUniqueAliases.slice(cMaxAliasBadges).join(', ')}>
              +{aUniqueAliases.length - cMaxAliasBadges}
              {fTranslate('text_187897ce0afc')}
            </span>
          {/if}
        {/if}

        {#if aUniqueTags.length > 0 && !vResolvedHideTags}
          {#each aUniqueTags as vTag (vTag)}
            <span class={cTagBadgeClass}>{vTag}</span>
          {/each}
        {/if}
      </span>
    {/if}
  {/snippet}

  <!-- badges, tags and icons do not shrink, so overflow-hidden bounds the id -->
  <span
    class="flex min-w-0 items-center gap-1.5 overflow-hidden {wrap
      ? 'flex-wrap'
      : ''} {iconsOnNewLine ? 'flex-col items-start' : ''} {className}"
    {title}
    {...rest}
  >
    <span
      class="flex min-w-0 items-center gap-1.5 max-md:gap-0.5 {wrap || stackId ? 'flex-wrap' : ''}"
    >
      {#if showRawTooltip}
        <Tooltip.fRoot>
          <Tooltip.Trigger class="flex min-w-0 items-center gap-1.5">
            {@render nameAndBadges()}
          </Tooltip.Trigger>

          <Tooltip.Content>
            <p>{modelId}</p>
          </Tooltip.Content>
        </Tooltip.fRoot>
      {:else}
        {@render nameAndBadges()}
      {/if}

      {#if !iconsOnNewLine}
        <ModelCapabilityIcons
          {hideCapabilities}
          {hideModalities}
          {modalities}
          {supportsThinking}
          {supportsToolUse}
        />
      {/if}
    </span>

    {#if iconsOnNewLine || contextLength || sizeRange}
      <span class="inline-flex items-center gap-1.5">
        {#if iconsOnNewLine}
          <ModelCapabilityIcons
            {hideCapabilities}
            {hideModalities}
            {modalities}
            {supportsThinking}
            {supportsToolUse}
          />
        {/if}

        {#if contextLength}
          <span class="inline-flex items-center gap-1 text-muted-foreground">
            <ScrollText class="h-3 w-3" />

            <span class="text-xs">{fFormatParameters(contextLength)}</span>
          </span>
        {/if}

        {#if sizeRange}
          <span class="inline-flex items-center gap-1 text-muted-foreground">
            <Database class="h-3 w-3" />

            <span class="text-xs"
              >{HuggingFaceService.mFormatSizeRange(sizeRange.min, sizeRange.max)}</span
            >
          </span>
        {/if}
      </span>
    {/if}
  </span>
{/if}
