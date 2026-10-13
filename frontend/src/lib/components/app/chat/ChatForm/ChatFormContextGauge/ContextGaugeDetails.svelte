<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ContextGaugeDetailRow from './ContextGaugeDetailRow.svelte';
  import { cStatsUnits } from '$lib/constants';

  interface Props {
    currentRead: number;
    currentFresh: number;
    currentCache: number;
    currentOutput: number;
    kvTotal: number;
    cumulativeRead: number;
    cumulativeOutput: number;
    cumulativeCacheTotal: number;
    averageTokensPerSecond: number | null;
    transientDetails: string[];
  }

  let {
    averageTokensPerSecond,
    cumulativeCacheTotal,
    cumulativeOutput,
    cumulativeRead,
    currentCache,
    currentFresh,
    currentOutput,
    currentRead,
    kvTotal,
    transientDetails
  }: Props = $props();

  const cHasCumulative = $derived(cumulativeRead > 0 || cumulativeOutput > 0);
  const cHasCurrent = $derived(currentRead > 0 || currentOutput > 0);
</script>

<div class="mt-3 border-t border-border/50 pt-4">
  <p class="text-xs text-muted-foreground max-md:text-[13px]">{fTranslate('text_f59cbe0da913')}</p>

  <div class="flex flex-col gap-4 pt-4 text-xs max-md:gap-5 max-md:text-[13px]">
    {#if cHasCumulative}
      <div>
        <h3
          class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2 max-md:text-xs"
        >
          {fTranslate('text_a1b285c2e617')}
        </h3>

        <div class="flex flex-col gap-2">
          {#if cumulativeRead > 0}
            <ContextGaugeDetailRow
              label={fTranslate('text_5c5444cd1b95')}
              subtitle={cumulativeCacheTotal > 0
                ? `${cumulativeCacheTotal.toLocaleString()} reused from KV cache`
                : undefined}
              value={`${cumulativeRead.toLocaleString()} tok`}
            />
          {/if}

          {#if cumulativeOutput > 0}
            <ContextGaugeDetailRow
              label={fTranslate('text_2f3bda40333c')}
              value={`${cumulativeOutput.toLocaleString()} tok`}
            />
          {/if}
        </div>
      </div>
    {/if}

    {#if cHasCurrent}
      <div>
        <h3
          class="text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70 mb-2 max-md:text-xs"
        >
          {fTranslate('text_e896ec9b0612')}
        </h3>

        <div class="flex flex-col gap-2">
          {#if currentRead > 0}
            <ContextGaugeDetailRow
              label={fTranslate('text_5c39123805ff')}
              subtitle={currentCache > 0
                ? `${currentFresh.toLocaleString()} fresh + ${currentCache.toLocaleString()} cached`
                : undefined}
              value={`${currentRead.toLocaleString()} tok`}
            />
          {/if}

          {#if currentOutput > 0}
            <ContextGaugeDetailRow
              label={fTranslate('text_827ec8d9f99d')}
              value={`${currentOutput.toLocaleString()} tok`}
            />
          {/if}

          <div class="pt-1 mt-0.5 border-t border-border/30">
            <div class="flex justify-between">
              <span class="text-muted-foreground">{fTranslate('text_4e453eb018ba')}</span>

              <span class="font-mono font-medium"
                >{kvTotal.toLocaleString()} {fTranslate('text_1a7674eb4ee7')}</span
              >
            </div>
          </div>
        </div>
      </div>
    {/if}

    {#if averageTokensPerSecond !== null}
      <div class="pt-1.5 mt-1 border-t border-border/30">
        <ContextGaugeDetailRow
          label={fTranslate('text_607e4c5f4272')}
          value={`${averageTokensPerSecond.toFixed(1)}${cStatsUnits.TOKENS_PER_SECOND}`}
        />
      </div>
    {/if}

    {#each transientDetails as vDetail (vDetail)}
      <div class="font-mono text-muted-foreground">{vDetail}</div>
    {/each}
  </div>
</div>
