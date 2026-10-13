<script lang="ts">
  import { ScrollCarousel } from '$lib/components/app';
  import { cIconClassDefault, cUiDataAttrs } from '$lib/constants';
  import { BooleanString } from '$lib/enums';
  import { fUseScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';
  import type { SettingsSection, SettingsSectionTitle } from '$lib/types';
  import { onMount, tick } from 'svelte';

  interface Props {
    sections: SettingsSection[];
    isActive: (pSection: SettingsSection) => boolean;
    onSectionChange?: (pSection: SettingsSectionTitle) => void;
  }

  let { isActive, onSectionChange, sections }: Props = $props();

  const cCarousel = fUseScrollCarousel();

  onMount(async () => {
    await tick();

    if (cCarousel.scrollContainer) {
      const cActiveTab = cCarousel.scrollContainer.querySelector(
        `[${cUiDataAttrs.ACTIVE}="${BooleanString.cTrue}"]`
      );

      if (cActiveTab instanceof HTMLElement) {
        cCarousel.scrollToCenter(cActiveTab);
      }
    }
  });

  export function updateCarousel() {
    setTimeout(cCarousel.updateScrollButtons, 100);
  }
</script>

<div class="flex flex-col bg-background md:hidden sticky top-13 z-50">
  <div class="border-b border-border/30">
    <ScrollCarousel alwaysShowArrows carousel={cCarousel} containerClass="py-2" innerClass="gap-2">
      {#each sections as vSection (vSection.title)}
        <button
          class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap transition-colors first:ml-4 last:mr-4 hover:bg-accent {isActive(
            vSection
          )
            ? 'bg-accent text-accent-foreground'
            : 'text-muted-foreground'}"
          {...{ [cUiDataAttrs.ACTIVE]: isActive(vSection) }}
          onclick={(pE: MouseEvent) => {
            onSectionChange?.(vSection.title);
            cCarousel.scrollToCenter(pE.currentTarget as HTMLElement);
          }}
        >
          <vSection.icon class="{cIconClassDefault} flex-shrink-0" />

          <span>{vSection.title}</span>
        </button>
      {/each}
    </ScrollCarousel>
  </div>
</div>
