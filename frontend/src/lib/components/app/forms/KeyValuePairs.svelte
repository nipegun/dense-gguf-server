<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Plus, Trash2 } from '@lucide/svelte';
  import { Input } from '$lib/components/ui/input';
  import { cKeyValuePairKeyMaxLength, cKeyValuePairValueMaxLength } from '$lib/constants';
  import type { KeyValuePair } from '$lib/types';
  import {
    fAutoResizeTextarea,
    fSanitizeKeyValuePairKey,
    fSanitizeKeyValuePairValue
  } from '$lib/utils';
  import { tick } from 'svelte';

  interface Props {
    class?: string;
    pairs: KeyValuePair[];
    onPairsChange: (pPairs: KeyValuePair[]) => void;
    keyPlaceholder?: string;
    valuePlaceholder?: string;
    addButtonLabel?: string;
    emptyMessage?: string;
    sectionLabel?: string;
    sectionLabelOptional?: boolean;
  }

  let {
    addButtonLabel = 'Add',
    class: className = '',
    emptyMessage = 'No items configured.',
    keyPlaceholder = 'Key',
    onPairsChange,
    pairs,
    sectionLabel,
    sectionLabelOptional = true,
    valuePlaceholder = 'Value'
  }: Props = $props();

  // Pre-allocate the ref array so `bind:ref={keyInputRefs[index]}` never reads `undefined`
  // for in-range indices; the $effect below keeps it in sync when `pairs` grows.
  // svelte-ignore state_referenced_locally
  let aKeyInputRefs: (HTMLInputElement | null)[] = $state(pairs.map(() => null));

  async function fAddPair() {
    // Capture the target index before mutating so deletions earlier in the
    // list can't make keyInputRefs.length drift past the newly-appended row.
    const cNewIndex = pairs.length;

    onPairsChange([...pairs, { key: '', value: '' }]);
    await tick();
    aKeyInputRefs[cNewIndex]?.focus();
  }

  function fRemovePair(pIndex: number) {
    onPairsChange(pairs.filter((p, pI) => pI !== pIndex));
  }

  function fUpdatePairKey(pIndex: number, pRawKey: string) {
    const cKey = fSanitizeKeyValuePairKey(pRawKey);
    const aNewPairs = [...pairs];

    aNewPairs[pIndex] = { ...aNewPairs[pIndex], key: cKey };
    onPairsChange(aNewPairs);
  }

  function fTrimPairKey(pIndex: number, pKey: string) {
    const cTrimmed = pKey.trim();

    if (cTrimmed === pKey) return;

    const aNewPairs = [...pairs];

    aNewPairs[pIndex] = { ...aNewPairs[pIndex], key: cTrimmed };
    onPairsChange(aNewPairs);
  }

  function fUpdatePairValue(pIndex: number, pRawValue: string) {
    const cValue = fSanitizeKeyValuePairValue(pRawValue);
    const aNewPairs = [...pairs];

    aNewPairs[pIndex] = { ...aNewPairs[pIndex], value: cValue };
    onPairsChange(aNewPairs);
  }

  function fTrimPairValue(pIndex: number, pValue: string) {
    const cTrimmed = pValue.trim();

    if (cTrimmed === pValue) return;

    const aNewPairs = [...pairs];

    aNewPairs[pIndex] = { ...aNewPairs[pIndex], value: cTrimmed };
    onPairsChange(aNewPairs);
  }

  // Keep keyInputRefs aligned with pairs length so bind:ref never sees `undefined`.
  // $effect.pre runs during traversal in tree order, before the {#each} block re-renders,
  // so newly-appended items always have a defined slot when their binding is set up.
  $effect.pre(() => {
    while (aKeyInputRefs.length < pairs.length) {
      aKeyInputRefs.push(null);
    }
  });
</script>

<div class={className}>
  <div class="mb-2 flex items-center justify-between">
    {#if sectionLabel}
      <span class="text-xs font-medium select-none">
        {sectionLabel}
        {#if sectionLabelOptional}
          <span class="text-muted-foreground">{fTranslate('text_0059798b7f70')}</span>
        {/if}
      </span>
    {/if}

    <button
      class="inline-flex cursor-pointer items-center gap-1 rounded-md px-1.5 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
      onclick={fAddPair}
      type="button"
    >
      <Plus class="h-3 w-3" />
      {addButtonLabel}
    </button>
  </div>

  {#if pairs.length > 0}
    <div class="space-y-3">
      {#each pairs as vPair, vIndex (vIndex)}
        <div class="flex items-start gap-2">
          <Input
            bind:ref={aKeyInputRefs[vIndex]}
            class="flex-1"
            maxlength={cKeyValuePairKeyMaxLength}
            onblur={(pE) => fTrimPairKey(vIndex, pE.currentTarget.value)}
            oninput={(pE) => fUpdatePairKey(vIndex, pE.currentTarget.value)}
            placeholder={keyPlaceholder}
            type="text"
            value={vPair.key}
          />

          <textarea
            use:fAutoResizeTextarea
            class="flex-1 resize-none rounded-md border border-input bg-transparent px-3 py-2 text-sm leading-5 placeholder:text-muted-foreground focus-visible:ring-1 focus-visible:ring-ring focus-visible:outline-none"
            maxlength={cKeyValuePairValueMaxLength}
            onblur={(pE) => fTrimPairValue(vIndex, pE.currentTarget.value)}
            oninput={(pE) => {
              fUpdatePairValue(vIndex, pE.currentTarget.value);
              fAutoResizeTextarea(pE.currentTarget);
            }}
            placeholder={valuePlaceholder}
            rows="1"
            value={vPair.value}
          ></textarea>

          <button
            aria-label={fTranslate('text_5a89edf2867d')}
            class="mt-1.5 shrink-0 cursor-pointer rounded-md p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
            onclick={() => fRemovePair(vIndex)}
            type="button"
          >
            <Trash2 class="h-3.5 w-3.5" />
          </button>
        </div>
      {/each}
    </div>
  {:else}
    <p class="select-none text-xs text-muted-foreground">{emptyMessage}</p>
  {/if}
</div>
