<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import LanguageSelector from '../LanguageSelector.svelte';
  import { RefreshCw } from '@lucide/svelte';
  import {
    SettingsChatDesktopSidebar,
    SettingsChatFields,
    SettingsChatImportExportTab,
    SettingsChatMobileHeader,
    SettingsChatToolsTab,
    SettingsFooter
  } from '$lib/components/app/settings';
  import { Button } from '$lib/components/ui/button';
  import {
    aNumericFields,
    aPositiveIntegerFields,
    aSettingsChatSections,
    cSettingsSectionSlugs
  } from '$lib/constants';
  import { ColorMode } from '$lib/enums/ui.enums';
  import { cModelsStore, cServerStore, cSettingsStore } from '$lib/stores';
  import type { SettingsSection, SettingsSectionTitle } from '$lib/types';
  import { setMode } from 'mode-watcher';
  import { fade } from 'svelte/transition';
  interface Props {
    initialSection?: string;
    onSectionChange?: (pSection: SettingsSectionTitle) => void;
    onClose?: () => void;
  }

  let { initialSection, onClose, onSectionChange }: Props = $props();

  let vActiveSlug = $derived(initialSection ?? 'general');

  function fHandleSectionChange(pSection: SettingsSectionTitle) {
    const cFound = aSettingsChatSections.find((pS) => pS.title === pSection);

    if (cFound) {
      vActiveSlug = cFound.slug;
    }

    onSectionChange?.(pSection);
  }

  let vCurrentSection = $derived(
    aSettingsChatSections.find((pSection) => pSection.slug === vActiveSlug) ||
      aSettingsChatSections[0]
  );

  let vLocalConfig: SettingsConfigType = $state({ ...cSettingsStore.vConfig });

  let vMobileHeader: { updateCarousel: () => void } | undefined;

  let vFetchInitiated = false;

  $effect(() => {
    if (cServerStore.isRouterMode && vCurrentSection.fields?.length && !vFetchInitiated) {
      vFetchInitiated = true;

      void cModelsStore
        .mFetch()
        .then(() => cModelsStore.mFetchRouterModels())
        .then(() => cModelsStore.props.mFetchModalitiesForLoadedModels())
        .then(() => cModelsStore.mEnsureFirstModelSelected());
    }
  });

  function fHandleThemeChange(pNewTheme: string) {
    vLocalConfig.theme = pNewTheme;
    setMode(pNewTheme as ColorMode);
  }

  function fHandleConfigChange(pKey: string, pValue: string | boolean) {
    vLocalConfig[pKey] = pValue;
  }

  function fHandleReset() {
    vLocalConfig = { ...cSettingsStore.vConfig };
    setMode(vLocalConfig.theme as ColorMode);
    vMobileHeader?.updateCarousel();
  }

  function fHandleSave() {
    if (
      vLocalConfig.customJson &&
      typeof vLocalConfig.customJson === 'string' &&
      vLocalConfig.customJson.trim()
    ) {
      try {
        JSON.parse(vLocalConfig.customJson);
      } catch (error) {
        alert('Invalid JSON in custom parameters. Please check the format and try again.');
        console.error(error);

        return;
      }
    }

    const cProcessedConfig = { ...vLocalConfig };

    for (const cField of aNumericFields) {
      if (cProcessedConfig[cField] !== undefined && cProcessedConfig[cField] !== '') {
        const cNumValue = Number(cProcessedConfig[cField]);

        if (!isNaN(cNumValue)) {
          if ((aPositiveIntegerFields as readonly string[]).includes(cField)) {
            const cEntryByMinMax = aSettingsChatSections
              .flatMap((pSection) => pSection.fields ?? [])
              .find((pEntry) => pEntry.key === cField);
            const cLo = cEntryByMinMax?.min ?? 1;
            const cHi = cEntryByMinMax?.max ?? Number.POSITIVE_INFINITY;

            cProcessedConfig[cField] = Math.max(cLo, Math.min(cHi, Math.round(cNumValue)));
          } else {
            cProcessedConfig[cField] = cNumValue;
          }
        } else {
          alert(`Invalid numeric value for ${cField}. Please enter a valid number.`);

          return;
        }
      }
    }

    cSettingsStore.mUpdateMultipleConfig(cProcessedConfig);
    onClose?.();
  }

  export function reset() {
    vLocalConfig = { ...cSettingsStore.vConfig };
  }
</script>

<div in:fade={{ duration: 150 }} class="mx-auto flex h-full w-full flex-col">
  <div class="flex flex-1 flex-col md:flex-row md:gap-4">
    <SettingsChatDesktopSidebar
      isActive={(pSection: SettingsSection) => pSection.slug === vActiveSlug}
      onSectionChange={fHandleSectionChange}
      sections={aSettingsChatSections}
    />

    <SettingsChatMobileHeader
      bind:this={vMobileHeader}
      isActive={(pSection: SettingsSection) => pSection.slug === vActiveSlug}
      onSectionChange={fHandleSectionChange}
      sections={aSettingsChatSections}
    />

    <div class="mx-auto max-w-2xl px-4 flex-1 md:mt-4">
      <div class="space-y-6 pt-3">
        <div class="grid">
          {#if vCurrentSection.slug === cSettingsSectionSlugs.GENERAL}<LanguageSelector />{/if}
          {#if vCurrentSection.slug === cSettingsSectionSlugs.TOOLS}
            <SettingsChatToolsTab />
          {:else if vCurrentSection.slug === cSettingsSectionSlugs.IMPORT_EXPORT}
            <SettingsChatImportExportTab />
          {:else if vCurrentSection.fields}
            <div class="space-y-6">
              <SettingsChatFields
                fields={vCurrentSection.fields}
                localConfig={vLocalConfig}
                onConfigChange={fHandleConfigChange}
                onThemeChange={fHandleThemeChange}
              />

              {#if vCurrentSection.slug === cSettingsSectionSlugs.GENERAL}
                <div class="flex justify-end">
                  <Button onclick={() => window.location.reload()} variant="outline">
                    <RefreshCw class="h-3 w-3" />
                    {fTranslate('text_10f0b454208d')}
                  </Button>
                </div>
              {/if}
            </div>
          {/if}
        </div>

        <div class="mt-8 border-t border-border/30 pt-6">
          <p class="text-xs text-muted-foreground">{fTranslate('text_d1955162085a')}</p>
        </div>
      </div>

      <SettingsFooter onReset={fHandleReset} onSave={fHandleSave} />
    </div>
  </div>
</div>
