<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import SearchInput from '$lib/components/app/forms/SearchInput.svelte';
  import { Button } from '$lib/components/ui/button';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { ScrollArea } from '$lib/components/ui/scroll-area';
  import { cUiDataAttrs } from '$lib/constants';
  import { fUseMarqueeSelection } from '$lib/hooks/use-marquee-selection.svelte';
  import { SvelteSet } from 'svelte/reactivity';

  interface Props {
    conversations: DatabaseConversation[];
    messageCountMap?: Map<string, number>;
    mode: 'export' | 'import';
    onCancel: () => void;
    onConfirm: (pSelectedConversations: DatabaseConversation[]) => void;
    isOpen?: boolean;
  }

  let {
    conversations,
    isOpen = true,
    messageCountMap = new Map(),
    mode,
    onCancel,
    onConfirm
  }: Props = $props();

  let vSearchQuery = $state('');
  let vSelectedIds = $state.raw<SvelteSet<string>>(fGetInitialSelectedIds());

  function fGetInitialSelectedIds(): SvelteSet<string> {
    return new SvelteSet(conversations.map((pC) => pC.id));
  }

  let aFilteredConversations = $derived(
    conversations.filter((pConv) => {
      const name = pConv.name || 'Untitled conversation';

      return name.toLowerCase().includes(vSearchQuery.toLowerCase());
    })
  );

  let aOrderedIds = $derived(aFilteredConversations.map((pC) => pC.id));

  let vAllSelected = $derived(
    aFilteredConversations.length > 0 &&
      aFilteredConversations.every((pConv) => vSelectedIds.has(pConv.id))
  );

  let vSomeSelected = $derived(
    aFilteredConversations.some((pConv) => vSelectedIds.has(pConv.id)) && !vAllSelected
  );

  const cMarquee = fUseMarqueeSelection({
    enabled: () => isOpen,
    orderedIds: () => aOrderedIds,
    selectedIds: () => vSelectedIds
  });

  function fToggleAll() {
    const cNewSet = new SvelteSet(vSelectedIds);

    if (vAllSelected) {
      aFilteredConversations.forEach((pConv) => cNewSet.delete(pConv.id));
    } else {
      aFilteredConversations.forEach((pConv) => cNewSet.add(pConv.id));
    }

    vSelectedIds = cNewSet;
  }

  function fHandleConfirm() {
    const aSelected = conversations.filter((pConv) => vSelectedIds.has(pConv.id));

    onConfirm(aSelected);
  }

  function fHandleCancel() {
    vSelectedIds = fGetInitialSelectedIds();
    vSearchQuery = '';
    cMarquee.reset();

    onCancel();
  }

  export function reset() {
    vSelectedIds = fGetInitialSelectedIds();
    vSearchQuery = '';
    cMarquee.reset();
  }
</script>

<div class="space-y-4">
  <SearchInput bind:value={vSearchQuery} placeholder={fTranslate('text_e90dd0173105')} />

  <div class="flex items-center justify-between text-sm text-muted-foreground">
    <span>
      {vSelectedIds.size}
      {fTranslate('text_28391d3bc64e')}
      {conversations.length}
      {fTranslate('text_d7cbbb688b2e')}
      {#if vSearchQuery}
        ({aFilteredConversations.length} {fTranslate('text_db6ec1bfdac0')}{/if}
    </span>
  </div>

  <div class="overflow-hidden rounded-md border">
    <ScrollArea class="h-100">
      <table class="w-full">
        <thead class="sticky top-0 z-10 bg-muted">
          <tr class="border-b">
            <th class="w-12 p-3 text-left">
              <Checkbox
                checked={vAllSelected}
                indeterminate={vSomeSelected}
                onCheckedChange={fToggleAll}
              />
            </th>

            <th class="p-3 text-left text-sm font-medium">{fTranslate('text_ca6722cd00cf')}</th>

            <th class="w-32 p-3 text-left text-sm font-medium">{fTranslate('text_04d7b4833927')}</th
            >
          </tr>
        </thead>

        <tbody>
          {#if aFilteredConversations.length === 0}
            <tr>
              <td class="p-8 text-center text-sm text-muted-foreground" colspan="3">
                {#if vSearchQuery}{fTranslate('text_82bf5deb2051')}{vSearchQuery}"
                {:else}{fTranslate('text_5a484c8b8b23')}{/if}
              </td>
            </tr>
          {:else}
            {#each aFilteredConversations as vConv (vConv.id)}
              {@const cChecked = vSelectedIds.has(vConv.id)}
              <tr
                class="cursor-pointer border-b transition-colors hover:bg-muted/50 {cChecked
                  ? 'bg-muted/75'
                  : ''}"
                {...{ [cUiDataAttrs.CONVERSATION_ROW]: vConv.id }}
                onclick={(pEvent) => cMarquee.rowClick(vConv.id, pEvent.shiftKey)}
                onmousedown={(pEvent) => cMarquee.rowMouseDown(vConv.id, pEvent)}
              >
                <td class="p-3">
                  <Checkbox
                    checked={cChecked}
                    onclick={(pEvent) => {
                      pEvent.preventDefault();
                      pEvent.stopPropagation();
                      cMarquee.rowClick(vConv.id, pEvent.shiftKey);
                    }}
                  />
                </td>

                <td class="p-3 text-sm">
                  <div class="max-w-68 truncate" title={vConv.name || 'Untitled conversation'}>
                    {vConv.name || 'Untitled conversation'}
                  </div>
                </td>

                <td class="p-3 text-sm text-muted-foreground">
                  {messageCountMap.get(vConv.id) ?? 0}
                </td>
              </tr>
            {/each}
          {/if}
        </tbody>
      </table>
    </ScrollArea>
  </div>

  <div class="flex justify-end gap-2">
    <Button onclick={fHandleCancel} variant="outline">{fTranslate('text_19766ed6ccb2')}</Button>

    <Button disabled={vSelectedIds.size === 0} onclick={fHandleConfirm}>
      {mode === 'export' ? 'Export' : 'Import'} ({vSelectedIds.size})
    </Button>
  </div>
</div>
