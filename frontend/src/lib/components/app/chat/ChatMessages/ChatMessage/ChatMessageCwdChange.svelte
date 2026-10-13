<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Folder, FolderX } from '@lucide/svelte';
  import type { DatabaseMessage } from '$lib/types';
  import { fParseCwdMessage } from '$lib/utils';

  interface Props {
    class?: string;
    message: DatabaseMessage;
  }

  let { class: className = '', message }: Props = $props();

  // Parse the synthetic message content in the UI so the row reuses the
  // exact same text the model saw, including any guidance suffix.
  let vInfo = $derived(fParseCwdMessage(message.content));
</script>

{#if vInfo}
  <div class="text-muted-foreground flex items-center gap-2 py-1.5 {className}">
    {#if vInfo.path === null}
      <FolderX class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />

      <span class="text-foreground/80 text-sm font-medium">{fTranslate('dgs_027ca2f2a76c')}</span>
    {:else}
      <Folder class="text-muted-foreground/60 h-3.5 w-3.5 shrink-0" />

      <span class="text-foreground/80 text-sm font-medium">{fTranslate('dgs_7ef99151dc34')} </span>

      <span class="font-mono text-foreground/90 text-sm break-all" title={vInfo.path}>
        {vInfo.display}
      </span>
    {/if}
  </div>
{/if}
