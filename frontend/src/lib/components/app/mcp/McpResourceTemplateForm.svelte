<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { InputWithSuggestions } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import { cMinAutocompleteInputLength } from '$lib/constants';
  import { KeyboardKey } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';
  import type { MCPResourceTemplateInfo } from '$lib/types';
  import {
    fDebounce,
    fExpandTemplate,
    fExtractTemplateVariables,
    fIsTemplateComplete
  } from '$lib/utils';

  interface Props {
    template: MCPResourceTemplateInfo;
    onResolve: (pUri: string, pServerName: string) => void;
    onCancel: () => void;
  }

  let { onCancel, onResolve, template }: Props = $props();

  const aVariables = $derived(fExtractTemplateVariables(template.uriTemplate));

  let dValues = $state<Record<string, string>>({});
  let dSuggestions = $state<Record<string, string[]>>({});
  let dLoadingSuggestions = $state<Record<string, boolean>>({});
  let vActiveAutocomplete = $state<string | null>(null);
  let vAutocompleteIndex = $state(0);

  const cExpandedUri = $derived(fExpandTemplate(template.uriTemplate, dValues));
  const cIsComplete = $derived(fIsTemplateComplete(template.uriTemplate, dValues));

  const fFetchCompletions = fDebounce(async (pArgName: string, pValue: string) => {
    if (pValue.length < 1) {
      dSuggestions[pArgName] = [];

      return;
    }

    dLoadingSuggestions[pArgName] = true;

    try {
      const cResult = await cMcpStore.mGetResourceCompletions(
        template.serverName,
        template.uriTemplate,
        pArgName,
        pValue
      );

      if (cResult && cResult.values.length > 0) {
        const aFilteredValues = cResult.values.filter((pV) => pV.trim() !== '');

        if (aFilteredValues.length > 0) {
          dSuggestions[pArgName] = aFilteredValues;
          vActiveAutocomplete = pArgName;
          vAutocompleteIndex = 0;
        } else {
          dSuggestions[pArgName] = [];
        }
      } else {
        dSuggestions[pArgName] = [];
      }
    } catch (error) {
      console.error('[McpResourceTemplateForm] Failed to fetch completions:', error);
      dSuggestions[pArgName] = [];
    } finally {
      dLoadingSuggestions[pArgName] = false;
    }
  }, 200);

  function fHandleArgInput(pArgName: string, pValue: string) {
    dValues[pArgName] = pValue;
    fFetchCompletions(pArgName, pValue);
  }

  function fSelectSuggestion(pArgName: string, pValue: string) {
    dValues[pArgName] = pValue;
    dSuggestions[pArgName] = [];
    vActiveAutocomplete = null;
  }

  function fHandleArgKeydown(pEvent: KeyboardEvent, pArgName: string) {
    const aArgSuggestions = dSuggestions[pArgName] ?? [];

    if (pEvent.key === KeyboardKey.cEscape) {
      pEvent.preventDefault();
      pEvent.stopPropagation();

      if (aArgSuggestions.length > 0 && vActiveAutocomplete === pArgName) {
        dSuggestions[pArgName] = [];
        vActiveAutocomplete = null;
      } else {
        onCancel();
      }

      return;
    }

    if (aArgSuggestions.length === 0 || vActiveAutocomplete !== pArgName) return;

    if (pEvent.key === KeyboardKey.cArrowDown) {
      pEvent.preventDefault();
      vAutocompleteIndex = Math.min(vAutocompleteIndex + 1, aArgSuggestions.length - 1);
    } else if (pEvent.key === KeyboardKey.cArrowUp) {
      pEvent.preventDefault();
      vAutocompleteIndex = Math.max(vAutocompleteIndex - 1, 0);
    } else if (pEvent.key === KeyboardKey.cEnter && aArgSuggestions[vAutocompleteIndex]) {
      pEvent.preventDefault();
      pEvent.stopPropagation();
      fSelectSuggestion(pArgName, aArgSuggestions[vAutocompleteIndex]);
    }
  }

  function fHandleArgBlur(pArgName: string) {
    setTimeout(() => {
      if (vActiveAutocomplete === pArgName) {
        dSuggestions[pArgName] = [];
        vActiveAutocomplete = null;
      }
    }, 150);
  }

  function fHandleArgFocus(pArgName: string) {
    const cValue = dValues[pArgName] ?? '';

    if (cValue.length >= cMinAutocompleteInputLength) {
      fFetchCompletions(pArgName, cValue);
    }
  }

  function fHandleSubmit(pEvent: SubmitEvent) {
    pEvent.preventDefault();

    if (cIsComplete) {
      onResolve(cExpandedUri, template.serverName);
    }
  }
</script>

<form class="space-y-3" onsubmit={fHandleSubmit}>
  {#each aVariables as vVariable (vVariable.name)}
    <InputWithSuggestions
      autocompleteIndex={vActiveAutocomplete === vVariable.name ? vAutocompleteIndex : 0}
      isAutocompleteActive={vActiveAutocomplete === vVariable.name}
      isLoadingSuggestions={dLoadingSuggestions[vVariable.name] ?? false}
      name={vVariable.name}
      onBlur={() => fHandleArgBlur(vVariable.name)}
      onFocus={() => fHandleArgFocus(vVariable.name)}
      onInput={(pValue) => fHandleArgInput(vVariable.name, pValue)}
      onKeydown={(pE) => fHandleArgKeydown(pE, vVariable.name)}
      onSelectSuggestion={(pValue) => fSelectSuggestion(vVariable.name, pValue)}
      suggestions={dSuggestions[vVariable.name] ?? []}
      value={dValues[vVariable.name] ?? ''}
    />
  {/each}

  {#if cIsComplete}
    <div class="rounded-md bg-muted/50 px-3 py-2">
      <p class="text-xs text-muted-foreground">{fTranslate('text_7509c7981d15')}</p>

      <p class="mt-0.5 font-mono text-xs break-all">{cExpandedUri}</p>
    </div>
  {/if}

  <div class="flex justify-end gap-2 pt-1">
    <Button onclick={onCancel} size="sm" type="button" variant="secondary"
      >{fTranslate('text_19766ed6ccb2')}</Button
    >

    <Button disabled={!cIsComplete} size="sm" type="submit"
      >{fTranslate('text_65f6015df981')}</Button
    >
  </div>
</form>
