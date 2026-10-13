<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Loader2, RefreshCw } from '@lucide/svelte';
  import { SearchInput } from '$lib/components/app/forms';
  import { Button } from '$lib/components/ui/button';
  import { cIconClassDefault } from '$lib/constants';

  interface Props {
    isLoading: boolean;
    onRefresh: () => void;
    onSearch?: (pQuery: string) => void;
    searchQuery?: string;
  }

  let { isLoading, onRefresh, onSearch, searchQuery = '' }: Props = $props();
</script>

<div class="flex flex-col gap-2">
  <div class="mb-2 flex items-center gap-4">
    <SearchInput
      onInput={(pValue) => onSearch?.(pValue)}
      placeholder={fTranslate('text_76625c6090df')}
      value={searchQuery}
    />

    <Button
      class="h-8 w-8 p-0"
      disabled={isLoading}
      onclick={onRefresh}
      size="sm"
      title={fTranslate('text_28a1a41378e7')}
      variant="ghost"
    >
      {#if isLoading}
        <Loader2 class="{cIconClassDefault} animate-spin" />
      {:else}
        <RefreshCw class={cIconClassDefault} />
      {/if}
    </Button>
  </div>

  <h3 class="text-sm font-medium">{fTranslate('text_036bcef84f47')}</h3>
</div>
