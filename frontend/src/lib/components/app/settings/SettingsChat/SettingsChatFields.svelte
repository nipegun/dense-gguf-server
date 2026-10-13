<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { FlaskConical, RotateCcw } from '@lucide/svelte';
  import { SettingsChatParameterSourceIndicator } from '$lib/components/app/settings';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { Input } from '$lib/components/ui/input';
  import Label from '$lib/components/ui/label/label.svelte';
  import * as RadioGroup from '$lib/components/ui/radio-group';
  import * as Select from '$lib/components/ui/select';
  import { Textarea } from '$lib/components/ui/textarea';
  import { cIconClassDefault, dSettingConfigInfo, cSettingsKeys } from '$lib/constants';
  import { SettingsFieldType } from '$lib/enums/settings.enums';
  import { cModelsStore, cServerStore, cSettingsStore } from '$lib/stores';
  import { fNormalizeFloatingPoint } from '$lib/utils/precision';
  import type { Component } from 'svelte';

  interface Props {
    fields: SettingsFieldConfig[];
    localConfig: SettingsConfigType;
    onConfigChange: (pKey: string, pValue: string | boolean) => void;
    onThemeChange?: (pTheme: string) => void;
  }

  let { fields, localConfig, onConfigChange, onThemeChange }: Props = $props();

  let dCurrentModelParams = $derived.by(() => {
    void cModelsStore.props.vCacheVersion;

    if (cServerStore.isRouterMode) {
      const cCurrentModelName = cModelsStore.vSelectedModelName;

      if (cCurrentModelName) {
        const cCurrentModelProps = cModelsStore.props.mGetModelProps(cCurrentModelName);

        return (cCurrentModelProps?.default_generation_settings?.params ?? {}) as Record<
          string,
          unknown
        >;
      }
    }

    return (cServerStore.defaultParams ?? {}) as Record<string, unknown>;
  });
</script>

{#each fields as vField (vField.key)}
  {#if !vField.dependsOn || Boolean(localConfig[vField.dependsOn])}
    <div class={vField.dependsOn ? 'space-y-2 pl-6' : 'space-y-2'}>
      {#if vField.type === SettingsFieldType.INPUT}
        {@const cCurrentValue = String(localConfig[vField.key] ?? '')}
        {@const cServerDefault = dCurrentModelParams[vField.key]}
        {@const cIsCustomRealTime = (() => {
          if (cServerDefault == null) return false;

          if (cCurrentValue === '') return false;

          const cNumericInput = parseFloat(cCurrentValue);
          const cNormalizedInput = !isNaN(cNumericInput)
            ? Math.round(cNumericInput * 1000000) / 1000000
            : cCurrentValue;
          const cNormalizedDefault =
            typeof cServerDefault === 'number'
              ? Math.round(cServerDefault * 1000000) / 1000000
              : cServerDefault;

          return cNormalizedInput !== cNormalizedDefault;
        })()}

        <div class="flex items-center gap-2">
          <Label class="flex items-center gap-1.5 text-sm font-medium" for={vField.key}>
            {vField.label}

            {#if vField.isExperimental}
              <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
            {/if}
          </Label>

          {#if cIsCustomRealTime}
            <SettingsChatParameterSourceIndicator />
          {/if}
        </div>

        <div class="relative w-full">
          <Input
            autocomplete={vField.isPrivate ? 'new-password' : undefined}
            id={vField.key}
            type={vField.isPrivate ? 'password' : vField.isPositiveInteger ? 'number' : 'text'}
            {...vField.isPositiveInteger
              ? {
                  min: String(vField.min ?? 1),
                  step: '1',
                  ...(vField.max != null ? { max: String(vField.max) } : {})
                }
              : {}}
            class="w-full {cIsCustomRealTime ? 'pr-8' : ''}"
            oninput={(pE) => onConfigChange(vField.key, pE.currentTarget.value)}
            placeholder={dCurrentModelParams[vField.key] != null
              ? `Default: ${fNormalizeFloatingPoint(dCurrentModelParams[vField.key])}`
              : (vField.placeholder ?? '')}
            value={cCurrentValue}
          />

          {#if cIsCustomRealTime}
            <button
              aria-label={fTranslate('text_bc5b45ae7b60')}
              class="absolute top-1/2 right-2 inline-flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded transition-colors hover:bg-muted"
              onclick={() => {
                cSettingsStore.mResetParameterToServerDefault(vField.key);
                onConfigChange(vField.key, '');
              }}
              title={fTranslate('text_bc5b45ae7b60')}
              type="button"
            >
              <RotateCcw class="h-3 w-3" />
            </button>
          {/if}
        </div>

        {#if vField.help || dSettingConfigInfo[vField.key]}
          <p class="mt-1 text-xs text-muted-foreground">
            {@html vField.help || dSettingConfigInfo[vField.key]}
          </p>
        {/if}
      {:else if vField.type === SettingsFieldType.cTextarea}
        {#if vField.label}
          <Label class="block flex items-center gap-1.5 text-sm font-medium" for={vField.key}>
            {vField.label}

            {#if vField.isExperimental}
              <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
            {/if}
          </Label>
        {/if}

        <Textarea
          class="min-h-[10rem] w-full md:max-w-3xl"
          id={vField.key}
          onchange={(pE) => onConfigChange(vField.key, pE.currentTarget.value)}
          placeholder=""
          value={String(localConfig[vField.key] ?? '')}
        />

        {#if vField.help || dSettingConfigInfo[vField.key]}
          <p class="mt-1 text-xs text-muted-foreground">
            {vField.help || dSettingConfigInfo[vField.key]}
          </p>
        {/if}

        {#if vField.key === cSettingsKeys.SYSTEM_MESSAGE}
          <div class="mt-3 flex items-center gap-2">
            <Checkbox
              checked={Boolean(localConfig.showSystemMessage ?? true)}
              id="showSystemMessage"
              onCheckedChange={(pChecked) =>
                onConfigChange(cSettingsKeys.SHOW_SYSTEM_MESSAGE, Boolean(pChecked))}
            />

            <Label class="cursor-pointer text-sm font-normal" for="showSystemMessage">
              {fTranslate('text_b1e1743d4be2')}
            </Label>
          </div>
        {/if}
      {:else if vField.type === SettingsFieldType.cSelect}
        {@const cSelectedOption = vField.options?.find(
          (pOpt: { value: string; label: string; icon?: Component }) =>
            pOpt.value === localConfig[vField.key]
        )}
        {@const cCurrentValue = localConfig[vField.key]}
        {@const cServerDefault = dCurrentModelParams[vField.key]}
        {@const cIsCustomRealTime = (() => {
          if (cServerDefault == null) return false;

          if (cCurrentValue === '' || cCurrentValue === undefined) return false;

          return cCurrentValue !== cServerDefault;
        })()}

        <div class="flex items-center gap-2">
          <Label class="flex items-center gap-1.5 text-sm font-medium" for={vField.key}>
            {vField.label}

            {#if vField.isExperimental}
              <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
            {/if}
          </Label>

          {#if cIsCustomRealTime}
            <SettingsChatParameterSourceIndicator />
          {/if}
        </div>

        <Select.fRoot
          onValueChange={(pValue) => {
            if (vField.key === cSettingsKeys.THEME && pValue && onThemeChange) {
              onThemeChange(pValue);
            } else {
              onConfigChange(vField.key, pValue);
            }
          }}
          type="single"
          value={cCurrentValue}
        >
          <div class="relative w-full md:w-auto">
            <Select.Trigger class="w-full">
              <div class="flex items-center gap-2">
                {#if cSelectedOption?.icon}
                  {@const IconComponent = cSelectedOption.icon}
                  <IconComponent class={cIconClassDefault} />
                {/if}

                {cSelectedOption?.label || `Select ${vField.label.toLowerCase()}`}
              </div>
            </Select.Trigger>

            {#if cIsCustomRealTime}
              <button
                aria-label={fTranslate('text_bc5b45ae7b60')}
                class="absolute top-1/2 right-8 inline-flex h-5 w-5 -translate-y-1/2 items-center justify-center rounded transition-colors hover:bg-muted"
                onclick={() => {
                  cSettingsStore.mResetParameterToServerDefault(vField.key);
                  onConfigChange(vField.key, '');
                }}
                title={fTranslate('text_bc5b45ae7b60')}
                type="button"
              >
                <RotateCcw class="h-3 w-3" />
              </button>
            {/if}
          </div>

          <Select.Content>
            {#if vField.options}
              {#each vField.options as vOption (vOption.value)}
                <Select.Item label={vOption.label} value={vOption.value}>
                  <div class="flex items-center gap-2">
                    {#if vOption.icon}
                      {@const IconComponent = vOption.icon}
                      <IconComponent class={cIconClassDefault} />
                    {/if}
                    {vOption.label}
                  </div>
                </Select.Item>
              {/each}
            {/if}
          </Select.Content>
        </Select.fRoot>

        {#if vField.help || dSettingConfigInfo[vField.key]}
          <p class="mt-1 text-xs text-muted-foreground">
            {vField.help || dSettingConfigInfo[vField.key]}
          </p>
        {/if}
      {:else if vField.type === SettingsFieldType.cRadio && vField.radioOptions}
        {@const cRadioOptions = vField.radioOptions}
        {@const cCurrentMode =
          cRadioOptions.find((pO: { key: string }) => Boolean(localConfig[pO.key]))?.value ??
          cRadioOptions[0].value}

        <Label class="flex items-center gap-1.5 text-sm font-medium mb-4">
          {vField.label}

          {#if vField.isExperimental}
            <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
          {/if}
        </Label>

        <RadioGroup.Root
          class="gap-4"
          onValueChange={(pValue) => {
            for (const cOpt of cRadioOptions) {
              onConfigChange(cOpt.key, cOpt.value === pValue);
            }
          }}
          value={cCurrentMode}
        >
          {#each cRadioOptions as vOpt (vOpt.value)}
            {@const cItemId = `${vField.key}-${vOpt.value}`}
            <div class="flex items-center gap-2">
              <RadioGroup.Item id={cItemId} value={vOpt.value} />

              <Label
                class="flex cursor-pointer items-center gap-1.5 text-sm font-normal"
                for={cItemId}
              >
                {vOpt.label}

                {#if vOpt.isExperimental}
                  <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
                {/if}
              </Label>
            </div>
          {/each}
        </RadioGroup.Root>

        {#if vField.help || dSettingConfigInfo[vField.key]}
          <p class="text-xs text-muted-foreground">
            {vField.help || dSettingConfigInfo[vField.key]}
          </p>
        {/if}
      {:else if vField.type === SettingsFieldType.cCheckbox}
        <div class="flex items-start space-x-3">
          <Checkbox
            checked={Boolean(localConfig[vField.key])}
            class="mt-1"
            id={vField.key}
            onCheckedChange={(pChecked) => onConfigChange(vField.key, pChecked)}
          />

          <div class="space-y-1">
            <label
              class="flex cursor-pointer items-center gap-1.5 pt-1 pb-0.5 text-sm leading-none font-medium"
              for={vField.key}
            >
              {vField.label}

              {#if vField.isExperimental}
                <FlaskConical class="h-3.5 w-3.5 text-muted-foreground" />
              {/if}
            </label>

            {#if vField.help || dSettingConfigInfo[vField.key]}
              <p class="text-xs text-muted-foreground">
                {vField.help || dSettingConfigInfo[vField.key]}
              </p>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  {/if}
{/each}
