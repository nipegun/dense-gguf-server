<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Folder, X } from '@lucide/svelte';
  import { ActionIcon } from '$lib/components/app/actions';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cSetWorkingDirectoryLabel } from '$lib/constants';
  import { fAbbreviateWorkingDir } from '$lib/utils';

  interface Props {
    directory?: string | null;
    homeBase?: string | null;
    disabled?: boolean;
    showTooltip?: boolean;
    onClear?: (pEvent?: MouseEvent) => void;
  }

  let {
    directory = null,
    disabled = false,
    homeBase = null,
    onClear,
    showTooltip = false
  }: Props = $props();

  const cDisplayLabel = $derived(
    directory ? fAbbreviateWorkingDir(directory, homeBase) : cSetWorkingDirectoryLabel
  );
  // Full path surface: hover the abbreviated label to recall the exact directory.
  const cDisplayLabelTitle = $derived(directory ?? '');
</script>

<span
  class:text-foreground={directory}
  class="text-muted-foreground inline-flex items-center gap-1 text-xs group"
>
  <div class="flex min-w-0 items-center gap-1 cursor-pointer">
    <Folder class="w-3.5 h-3.5" />

    {#if showTooltip && cDisplayLabelTitle}
      <Tooltip.fRoot>
        <Tooltip.Trigger>
          {#snippet child({ props: dProps })}
            <span {...dProps} class="max-w-64 truncate">{cDisplayLabel}</span>
          {/snippet}
        </Tooltip.Trigger>

        <Tooltip.Content>
          <p>{cDisplayLabelTitle}</p>
        </Tooltip.Content>
      </Tooltip.fRoot>
    {:else}
      <span class="max-w-64 truncate">{cDisplayLabel}</span>
    {/if}
  </div>

  {#if directory}
    <div
      class="w-0 overflow-hidden opacity-0 transition-[width,opacity] duration-200 ease-out group-hover:w-auto group-hover:opacity-100"
    >
      <ActionIcon
        ariaLabel="Reset working directory"
        class="!h-4 !w-4 shrink-0 text-muted-foreground hover:text-foreground"
        {disabled}
        icon={X}
        iconSize="h-3 w-3"
        onclick={onClear}
        stopPropagationOnClick
        tooltip={fTranslate('dgs_f1fe0277a1bc')}
      />
    </div>
  {/if}
</span>
