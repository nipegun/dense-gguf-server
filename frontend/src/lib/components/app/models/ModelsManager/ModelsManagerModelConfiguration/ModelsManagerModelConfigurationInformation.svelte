<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import {
    ActionIconCopyToClipboard,
    BadgesModality,
    CollapsibleSection
  } from '$lib/components/app';
  import { Badge } from '$lib/components/ui/badge';
  import { HuggingFaceService } from '$lib/services';
  import { cModelsStore } from '$lib/stores';
  import type { ApiDgsServerProps } from '$lib/types/api';
  import type { HfModelDetailInfo } from '$lib/types/huggingface';
  import type { ModelOption } from '$lib/types/models';
  import { fFormatFileSize, fFormatNumber, fFormatParameters } from '$lib/utils/formatters';

  interface Props {
    /** Hub details, when the Hub is enabled and the server reports none. */
    hub?: HfModelDetailInfo | null;
    option: ModelOption;
    serverProps: ApiDgsServerProps | null | undefined;
  }

  let { hub = null, option, serverProps }: Props = $props();

  let dMeta = $derived(option.meta);
  let vGguf = $derived(hub?.gguf ?? null);
  let vQuant = $derived(option.parsedId?.quantization ?? null);
  // a GGUF carries the count in its metadata, a transformers repo in its SafeTensors index
  let vHubParameters = $derived(HuggingFaceService.mParameterCount(hub));
  let aDraftSidecars = $derived(option.draftSidecars ?? []);

  let aModalities = $derived.by(() => {
    void cModelsStore.props.vCacheVersion;

    return cModelsStore.props.mGetModelModalitiesArray(option.id);
  });

  let aRows = $derived([
    {
      isCopyable: true,
      isMono: true,
      label: fTranslate('text_a1039763f359'),
      value: serverProps?.model_path ?? null
    },
    {
      label: fTranslate('text_db8951096dc3'),
      // the server reports the context it runs with once the model is loaded; until
      // then the listing, or the Hub, says what the model can take
      value: serverProps
        ? `${fFormatNumber(serverProps.default_generation_settings?.n_ctx ?? 0)} tokens`
        : option.contextLength
          ? `${fFormatNumber(option.contextLength)} tokens`
          : vGguf?.context_length
            ? `${fFormatNumber(vGguf.context_length)} tokens`
            : null
    },
    {
      label: fTranslate('text_bcae6c15f396'),
      value: dMeta?.size ? fFormatFileSize(dMeta.size) : null
    },
    {
      label: fTranslate('text_e68b36b17cbd'),
      value: dMeta?.n_params
        ? fFormatParameters(dMeta.n_params)
        : (option.parsedId?.params ?? (vHubParameters ? fFormatParameters(vHubParameters) : null))
    },
    {
      label: fTranslate('text_b53b4922dde8'),
      value: dMeta?.n_embd ? fFormatNumber(dMeta.n_embd) : null
    },
    {
      label: fTranslate('text_86f9679cfa5b'),
      value: dMeta?.n_vocab ? `${fFormatNumber(dMeta.n_vocab)} tokens` : null
    },
    { isBadge: true, label: fTranslate('dgs_a3e4d9d8b7eb'), value: vQuant },
    {
      isBadge: true,
      label: fTranslate('dgs_cd74053c5481'),
      value: (dMeta?.architecture as string) ?? vGguf?.architecture ?? null
    },
    {
      isBadge: true,
      label: fTranslate('dgs_172ddfdf2469'),
      value:
        aDraftSidecars.length > 0
          ? aDraftSidecars.map((pBadge) => pBadge.kind.toUpperCase()).join(', ')
          : null
    },
    {
      isCopyable: true,
      label: fTranslate('dgs_6505b9e78c2f'),
      value: aDraftSidecars[0]?.repo ?? null
    },
    {
      isBadge: true,
      label: fTranslate('dgs_8018210b96d5'),
      value: aDraftSidecars[0]?.quant ?? null
    },
    {
      label: fTranslate('text_14f89dfcfa6e'),
      value: serverProps?.total_slots != null ? String(serverProps.total_slots) : null
    },
    { isMono: true, label: fTranslate('text_642ae0c77e01'), value: serverProps?.build_info ?? null }
  ] satisfies Array<{
    isBadge?: boolean;
    isCopyable?: boolean;
    isMono?: boolean;
    label: string;
    value: string | null;
  }>);

  // rows without a value say nothing; a loaded model fills the rest in from /props
  let aVisibleRows = $derived(aRows.filter((pRow) => pRow.value !== null));

  const cSectionTrigger = 'flex w-full cursor-pointer items-center gap-2 py-2 text-left';
</script>

<div class="space-y-0">
  <div class="flex items-center gap-3 border-b border-border/30 py-2.5">
    <span class="text-sm text-muted-foreground">{fTranslate('text_5e2c614c23f0')}</span>

    <span class="ml-auto flex min-w-0 flex-1 items-center gap-2">
      <span class="min-w-0 flex-1 overflow-x-auto font-mono text-xs whitespace-nowrap">
        {option.model}
      </span>

      <ActionIconCopyToClipboard
        ariaLabel="Copy model name to clipboard"
        canCopy
        text={option.model}
      />
    </span>
  </div>

  {#each aVisibleRows as vRow (vRow.label)}
    <div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
      <span class="text-sm text-muted-foreground">{vRow.label}</span>

      <span class="ml-auto flex min-w-0 items-center gap-2 {vRow.isCopyable ? 'flex-1' : ''}">
        {#if vRow.value === null}
          <span class="text-muted-foreground">-</span>
        {:else}
          <span
            class={[
              'min-w-0',
              // a copyable value scrolls instead of clipping, the icon stays put
              vRow.isCopyable ? 'flex-1 overflow-x-auto whitespace-nowrap' : 'truncate',
              vRow.isBadge ? '' : 'text-sm',
              vRow.isMono ? 'font-mono text-xs' : ''
            ]}
          >
            {#if vRow.isBadge}
              <Badge class="h-5 px-1.5 text-[10px]" variant="secondary">{vRow.value}</Badge>
            {:else}
              {vRow.value}
            {/if}
          </span>

          {#if vRow.isCopyable}
            <ActionIconCopyToClipboard
              ariaLabel="Copy value to clipboard"
              canCopy
              text={vRow.value}
            />
          {/if}
        {/if}
      </span>
    </div>
  {/each}

  {#if aVisibleRows.length === 0}
    <p class="py-2.5 text-sm text-muted-foreground">
      {fTranslate('dgs_1c8d16680e0f')}
    </p>
  {/if}

  {#if aModalities.length > 0}
    <div class="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
      <span class="text-sm text-muted-foreground">{fTranslate('text_870ae34ac3e0')}</span>

      <span class="ml-auto flex flex-wrap gap-1"><BadgesModality modalities={aModalities} /></span>
    </div>
  {/if}
</div>

{#if !serverProps}
  <p class="pt-2 text-xs text-muted-foreground">
    {fTranslate('dgs_04dc82127232')}
  </p>
{/if}

<div class="mt-4">
  <CollapsibleSection triggerClass={cSectionTrigger}>
    {#snippet trigger()}
      <span class="text-sm font-medium">{fTranslate('text_3e07c9137dce')}</span>
    {/snippet}

    <pre
      class="mt-1 rounded-md bg-muted/50 p-2 text-xs whitespace-pre-wrap">{serverProps?.chat_template ??
        vGguf?.chat_template ??
        'Shown once the model is loaded.'}</pre>
  </CollapsibleSection>
</div>
