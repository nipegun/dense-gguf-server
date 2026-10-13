<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { fColorLevelBgClass, fColorLevelTextClass } from './context-gauge';
  import ContextGaugeDetails from './ContextGaugeDetails.svelte';
  import ContextGaugeLoadModel from './ContextGaugeLoadModel.svelte';
  import {
    fGaugeCardEnter,
    fGaugeCardLeave,
    cGaugePopup,
    fGaugePopupClose
  } from './gauge-popup.svelte';
  import * as Drawer from '$lib/components/ui/drawer';
  import { fUseContextGauge } from '$lib/hooks/use-context-gauge.svelte';
  import { cDeviceStore } from '$lib/stores';
  import { fFormatParameters } from '$lib/utils/formatters';

  const cGauge = fUseContextGauge();

  // The gauge hook wraps a processing state instance that only follows the
  // live stream while its own monitoring flag is set, so the card instance
  // starts monitoring like the dial does.
  $effect(() => {
    cGauge.mStartMonitoring();
  });

  let vCardEl = $state<HTMLElement | null>(null);

  // Any press outside the card and outside the dial closes the card.
  // Presses on the dial are excluded because the dial handles its own
  // toggle; the listener only exists while the card is open. A phone shows
  // the panel in a drawer instead, which brings its own dismissal.
  $effect(() => {
    if (!cGaugePopup.open || cDeviceStore.isMobile) return;

    const fOnPointerDown = (pEvent: PointerEvent) => {
      const cTarget = pEvent.target;

      if (!(cTarget instanceof Node)) return;

      if (vCardEl?.contains(cTarget)) return;

      if (cTarget instanceof Element && cTarget.closest('[data-context-gauge-trigger]')) return;

      fGaugePopupClose();
    };

    document.addEventListener('pointerdown', fOnPointerDown, true);

    return () => document.removeEventListener('pointerdown', fOnPointerDown, true);
  });

  const cShowProgressBar = $derived(
    cGauge.contextTotal !== null &&
      cGauge.contextTotal > 0 &&
      (cGauge.activeModelId !== null || cGauge.isActiveModelLoaded)
  );
</script>

{#snippet usage()}
  <span class="font-mono text-muted-foreground">
    {fFormatParameters(cGauge.contextUsed)}
    / {cGauge.contextTotal !== null ? fFormatParameters(cGauge.contextTotal) : '-'}
  </span>
{/snippet}

{#snippet body()}
  {#if cGauge.activeModelId !== null && !cGauge.isActiveModelLoaded}
    <ContextGaugeLoadModel
      isLoading={cGauge.isActiveModelLoading}
      modelId={cGauge.activeModelId}
      onLoad={cGauge.mLoadModel}
    />
  {:else if cShowProgressBar}
    <div class="h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div
        class="h-full rounded-full transition-all duration-300 {fColorLevelBgClass(
          cGauge.colorLevel
        )}"
        style="width: {cGauge.contextPercent}%"
      ></div>
    </div>

    <div class="flex justify-between text-xs text-muted-foreground max-md:text-[13px]">
      <span>
        <span class={fColorLevelTextClass(cGauge.colorLevel)}>{cGauge.contextPercent}%</span>
        {fTranslate('text_f83916135509')}
      </span>

      <span>
        {fFormatParameters(cGauge.contextAvailable ?? 0)}
        {fTranslate('text_43e3ece73ff8')}
      </span>
    </div>
  {:else}
    <div class="text-xs text-muted-foreground">{fTranslate('text_2d1c5067fcc7')}</div>
  {/if}

  {#if cGauge.hasAnyUsage}
    <ContextGaugeDetails
      averageTokensPerSecond={cGauge.averageTokensPerSecond}
      cumulativeCacheTotal={cGauge.cumulativeCacheTotal}
      cumulativeOutput={cGauge.cumulativeOutput}
      cumulativeRead={cGauge.cumulativeRead}
      currentCache={cGauge.currentCache}
      currentFresh={cGauge.currentFresh}
      currentOutput={cGauge.currentOutput}
      currentRead={cGauge.currentRead}
      kvTotal={cGauge.kvTotal}
      transientDetails={cGauge.transientDetails}
    />
  {/if}
{/snippet}

{#if cDeviceStore.isMobile}
  <Drawer.Root bind:open={cGaugePopup.open}>
    <Drawer.Content>
      <Drawer.Header>
        <Drawer.Title>{fTranslate('text_a6e600a10fed')}</Drawer.Title>

        <Drawer.Description class="sr-only">
          {fTranslate('dgs_19e308df050d')}
        </Drawer.Description>
      </Drawer.Header>

      <div class="flex flex-col gap-2 px-4 pb-4 max-md:gap-3.5">
        {@render usage()}

        {@render body()}
      </div>
    </Drawer.Content>
  </Drawer.Root>
{:else if cGaugePopup.open}
  <div
    bind:this={vCardEl}
    class="absolute z-50 w-64 -translate-x-1/2 rounded-lg border border-border/50 bg-popover p-3 text-sm text-popover-foreground shadow-lg ring-1 ring-foreground/10"
    onpointerenter={fGaugeCardEnter}
    onpointerleave={fGaugeCardLeave}
    role="status"
    style="left: {cGaugePopup.centerX}px; bottom: {cGaugePopup.bottom}px"
  >
    <div class="flex flex-col gap-2">
      <div class="flex items-center gap-2">
        <span class="font-medium">{fTranslate('text_a6e600a10fed')}</span>

        <span class="text-muted-foreground">·</span>

        {@render usage()}
      </div>

      {@render body()}
    </div>
  </div>
{/if}
