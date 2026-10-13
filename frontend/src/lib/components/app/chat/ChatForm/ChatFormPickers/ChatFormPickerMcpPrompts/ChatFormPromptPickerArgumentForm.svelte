<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ChatFormPromptPickerArgumentInput from './ChatFormPromptPickerArgumentInput.svelte';
  import { Button } from '$lib/components/ui/button';
  import type { MCPPromptInfo } from '$lib/types';

  interface Props {
    prompt: MCPPromptInfo;
    promptArgs: Record<string, string>;
    suggestions: Record<string, string[]>;
    loadingSuggestions: Record<string, boolean>;
    activeAutocomplete: string | null;
    autocompleteIndex: number;
    promptError: string | null;
    onArgInput: (pArgName: string, pValue: string) => void;
    onArgKeydown: (pEvent: KeyboardEvent, pArgName: string) => void;
    onArgBlur: (pArgName: string) => void;
    onArgFocus: (pArgName: string) => void;
    onSelectSuggestion: (pArgName: string, pValue: string) => void;
    onSubmit: (pEvent: SubmitEvent) => void;
    onCancel: () => void;
  }

  let {
    activeAutocomplete,
    autocompleteIndex,
    loadingSuggestions,
    onArgBlur,
    onArgFocus,
    onArgInput,
    onArgKeydown,
    onCancel,
    onSelectSuggestion,
    onSubmit,
    prompt,
    promptArgs,
    promptError,
    suggestions
  }: Props = $props();
</script>

<form class="space-y-3 pt-4" onsubmit={onSubmit}>
  {#each prompt.arguments ?? [] as vArg (vArg.name)}
    <ChatFormPromptPickerArgumentInput
      argument={vArg}
      autocompleteIndex={activeAutocomplete === vArg.name ? autocompleteIndex : 0}
      isAutocompleteActive={activeAutocomplete === vArg.name}
      isLoadingSuggestions={loadingSuggestions[vArg.name] ?? false}
      onBlur={() => onArgBlur(vArg.name)}
      onFocus={() => onArgFocus(vArg.name)}
      onInput={(pValue) => onArgInput(vArg.name, pValue)}
      onKeydown={(pE) => onArgKeydown(pE, vArg.name)}
      onSelectSuggestion={(pValue) => onSelectSuggestion(vArg.name, pValue)}
      suggestions={suggestions[vArg.name] ?? []}
      value={promptArgs[vArg.name] ?? ''}
    />
  {/each}

  {#if promptError}
    <div
      class="flex items-start gap-2 rounded-lg border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive"
      role="alert"
    >
      <span class="shrink-0">⚠</span>

      <span>{promptError}</span>
    </div>
  {/if}

  <div class="mt-8 flex justify-end gap-2">
    <Button onclick={onCancel} size="sm" type="button" variant="secondary"
      >{fTranslate('text_19766ed6ccb2')}</Button
    >

    <Button size="sm" type="submit">{fTranslate('text_0ce2c63c2535')}</Button>
  </div>
</form>
