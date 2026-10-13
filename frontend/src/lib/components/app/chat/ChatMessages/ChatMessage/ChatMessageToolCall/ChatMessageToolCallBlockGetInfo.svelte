<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import ToolCallBlock from './ToolCallBlock.svelte';
  import { XCircle } from '@lucide/svelte';
  import { cToolsStore } from '$lib/stores';
  import type { AgenticSection } from '$lib/types';
  import { fAbbreviateHome } from '$lib/utils';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    onToggle?: () => void;
  }

  let { isStreaming, onToggle, open, section }: Props = $props();

  type GetInfoMeta = {
    os?: string;
    cwd?: string;
    errorMessage?: string;
  };

  function fParseGetInfoMeta(pToolResultString: string | undefined): GetInfoMeta {
    if (!pToolResultString) return {};

    try {
      const cParsed: unknown = JSON.parse(pToolResultString);

      if (cParsed && typeof cParsed === 'object' && !Array.isArray(cParsed)) {
        const dObj = cParsed as Record<string, unknown>;

        if (typeof dObj.error === 'string') return { errorMessage: dObj.error };

        return {
          cwd: typeof dObj.cwd === 'string' ? dObj.cwd : undefined,
          os: typeof dObj.os === 'string' ? dObj.os : undefined
        };
      }
    } catch {
      // not JSON - nothing to show
    }

    return {};
  }

  const cInfoMeta = $derived(fParseGetInfoMeta(section.toolResult));
  const cHome = $derived(cToolsStore.serverHome);
  const cCwdDisplay = $derived(fAbbreviateHome(cInfoMeta.cwd ?? '', cHome));
</script>

<ToolCallBlock
  {isStreaming}
  meta={cInfoMeta}
  {onToggle}
  {open}
  {section}
  spinIconWhenActive
  title={fTranslate('dgs_0bc0f77ade2d')}
>
  {#snippet children(meta, _ctx)}
    {#if meta?.errorMessage}
      <div
        class="flex items-start gap-2 rounded bg-red-500/10 p-2 text-xs text-red-600 italic dark:text-red-400"
      >
        <XCircle class="mt-0.5 h-3 w-3 shrink-0" />

        <span>{meta.errorMessage}</span>
      </div>
    {:else if cInfoMeta.os || cInfoMeta.cwd}
      <table class="w-full table-fixed border-collapse text-sm">
        <colgroup>
          <col class="w-12" />

          <col />
        </colgroup>

        <tbody class="divide-y divide-border/50">
          {#if cInfoMeta.os}
            <tr>
              <th
                class="py-1 pr-3 text-left align-baseline text-[11px] font-medium tracking-wide text-muted-foreground/60 uppercase"
                scope="row"
              >
                {fTranslate('dgs_840a8dcfeae9')}
              </th>

              <td class="py-1 align-baseline">
                <div class="min-w-0 overflow-x-auto font-mono text-foreground/90">
                  {cInfoMeta.os}
                </div>
              </td>
            </tr>
          {/if}

          {#if cInfoMeta.cwd}
            <tr>
              <th
                class="py-1 pr-3 text-left align-baseline text-[11px] font-medium tracking-wide text-muted-foreground/60 uppercase"
                scope="row"
              >
                {fTranslate('dgs_6f2f7e6c6558')}
              </th>

              <td class="py-1 align-baseline">
                <div
                  class="min-w-0 overflow-x-auto font-mono text-foreground/90"
                  title={cInfoMeta.cwd}
                >
                  {cCwdDisplay}
                </div>
              </td>
            </tr>
          {/if}
        </tbody>
      </table>
    {:else if section.toolResult}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {section.toolResult}
      </div>
    {:else}
      <div class="rounded bg-muted/20 p-2 text-xs text-muted-foreground/70 italic">
        {fTranslate('dgs_65c6ad00415c')}
      </div>
    {/if}
  {/snippet}
</ToolCallBlock>
