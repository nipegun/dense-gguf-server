<script lang="ts">
  import {
    ChevronDown,
    ChevronRight,
    File,
    Image,
    MessageSquare,
    Mic,
    PencilRuler,
    Video
  } from '@lucide/svelte';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import * as Collapsible from '$lib/components/ui/collapsible';
  import * as Drawer from '$lib/components/ui/drawer';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import {
    aAttachmentFileItems,
    cAttachmentMenuText,
    dAttachmentModalityText,
    cIconClassDefault
  } from '$lib/constants';
  import { fGetChatFormActionsContext } from '$lib/contexts';
  import { AttachmentAction, AttachmentItemEnabledWhen } from '$lib/enums/attachment.enums';
  import { fUseAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';
  import { fUseToolsPanel } from '$lib/hooks/use-tools-panel.svelte';
  import type { ToolGroup } from '$lib/types';
  import type { Snippet } from 'svelte';

  interface Props {
    class?: string;
    trigger: Snippet<[{ disabled: boolean; onclick?: () => void }]>;
  }

  let { class: className = '', trigger }: Props = $props();

  const cChatFormActions = fGetChatFormActionsContext();

  let vDrawerOpen = $state(false);
  let vToolsExpanded = $state(false);

  const cAttachmentMenu = fUseAttachmentMenu(
    () => ({
      hasAudioModality: cChatFormActions.hasAudioModality,
      hasVideoModality: cChatFormActions.hasVideoModality,
      hasVisionModality: cChatFormActions.hasVisionModality
    }),
    () => ({
      onFileUpload: cChatFormActions.onFileUpload,
      onSystemPromptClick: cChatFormActions.onSystemPromptClick
    }),
    () => {
      vDrawerOpen = false;
    }
  );

  const dFileModalityIcons: Record<string, { icon: typeof Image; label: string }> = {
    [AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY]: {
      icon: Mic,
      label: dAttachmentModalityText[AttachmentItemEnabledWhen.HAS_AUDIO_MODALITY]
    },
    [AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY]: {
      icon: Video,
      label: dAttachmentModalityText[AttachmentItemEnabledWhen.HAS_VIDEO_MODALITY]
    },
    [AttachmentItemEnabledWhen.HAS_VISION_MODALITY]: {
      icon: Image,
      label: dAttachmentModalityText[AttachmentItemEnabledWhen.HAS_VISION_MODALITY]
    }
  };

  const aSupportedModalities = $derived.by(() =>
    aAttachmentFileItems
      .filter((pItem) => cAttachmentMenu.mIsItemEnabled(pItem.enabledWhen))
      .map((pItem) => dFileModalityIcons[pItem.enabledWhen ?? ''])
      .filter((pModality) => pModality !== undefined)
  );

  const cToolsPanel = fUseToolsPanel();

  const cItemClass =
    'flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm transition-colors hover:bg-accent active:bg-accent disabled:cursor-not-allowed disabled:opacity-50';

  const cItemRowClass =
    'flex w-full items-center justify-between gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent';
</script>

<div class="flex items-center gap-1 {className}">
  <Drawer.Root bind:open={vDrawerOpen}>
    {@render trigger({ disabled: cChatFormActions.disabled, onclick: () => (vDrawerOpen = true) })}

    <Drawer.Content class="gap-0 overflow-y-auto">
      <Drawer.Header>
        <Drawer.Title>{cAttachmentMenuText.ADD_TO_CHAT}</Drawer.Title>

        <Drawer.Description class="sr-only">
          {cAttachmentMenuText.DESCRIPTION}
        </Drawer.Description>
      </Drawer.Header>

      <div class="flex flex-col gap-1 px-1.5 pb-2">
        <button
          class={cItemClass}
          onclick={() => cAttachmentMenu.callbacks[AttachmentAction.FILE_UPLOAD]()}
          type="button"
        >
          <File class="{cIconClassDefault} shrink-0" />

          <span class="flex min-w-0 items-center gap-2">
            <span>{cAttachmentMenuText.ADD_FILES}</span>

            {#if aSupportedModalities.length > 0}
              <span class="flex items-center gap-0.75 text-muted-foreground">
                {#each aSupportedModalities as vModality (vModality.label)}
                  <Tooltip.fRoot>
                    <Tooltip.Trigger>
                      <vModality.icon class="size-2.75" />
                    </Tooltip.Trigger>

                    <Tooltip.Content>
                      <p>{vModality.label}</p>
                    </Tooltip.Content>
                  </Tooltip.fRoot>
                {/each}
              </span>
            {/if}
          </span>
        </button>

        <button
          class={cItemClass}
          onclick={() => cAttachmentMenu.callbacks[AttachmentAction.cSystemPromptClick]()}
          type="button"
        >
          <MessageSquare class="{cIconClassDefault} shrink-0" />

          <span>{cAttachmentMenuText.SYSTEM_MESSAGE}</span>
        </button>

        {#if cToolsPanel.totalToolCount > 0}
          <Collapsible.Root
            onOpenChange={(pOpen) => (vToolsExpanded = pOpen)}
            open={vToolsExpanded}
          >
            <Collapsible.Trigger class={cItemClass}>
              {#if vToolsExpanded}
                <ChevronDown class="{cIconClassDefault} shrink-0" />
              {:else}
                <ChevronRight class="{cIconClassDefault} shrink-0" />
              {/if}

              <PencilRuler class="inline {cIconClassDefault} shrink-0" />

              <span class="flex-1">{cAttachmentMenuText.TOOLS}</span>
            </Collapsible.Trigger>

            <Collapsible.Content>
              <div class="flex flex-col gap-0.5 pl-4">
                {#each cToolsPanel.categoryGroups as vGroup (vGroup.key)}
                  {@render groupRow(vGroup)}
                {/each}

                {#each cToolsPanel.mcpGroups as vGroup (vGroup.key)}
                  {@render groupRow(vGroup)}
                {/each}
              </div>
            </Collapsible.Content>
          </Collapsible.Root>
        {/if}
      </div>
    </Drawer.Content>
  </Drawer.Root>
</div>

{#snippet groupRow(pGroup: ToolGroup)}
  {@const cCheckState = cToolsPanel.mGetGroupCheckState(pGroup)}
  {@const cEnabledCount = cToolsPanel.mGetEnabledToolCount(pGroup)}
  {@const cFavicon = cToolsPanel.mGetFavicon(pGroup)}
  {@const cGroupDisabled = cToolsPanel.mIsGroupDisabled(pGroup)}

  <button
    class="{cItemRowClass} {cGroupDisabled ? 'pointer-events-none opacity-50' : ''}"
    onclick={() => cToolsPanel.mToggleGroupByKey(pGroup.key)}
    type="button"
  >
    {#if cFavicon}
      <img
        alt=""
        class="{cIconClassDefault} shrink-0 rounded-sm"
        onerror={(pE) => {
          (pE.currentTarget as HTMLImageElement).style.display = 'none';
        }}
        src={cFavicon}
      />
    {/if}

    <span class="min-w-0 flex-1 truncate text-sm font-medium">{pGroup.label}</span>

    <span class="shrink-0 text-xs text-muted-foreground">
      {cEnabledCount}/{pGroup.tools.length}
    </span>

    <Checkbox
      checked={cCheckState.checked}
      class="{cIconClassDefault} shrink-0"
      indeterminate={cCheckState.indeterminate}
      onCheckedChange={() => cToolsPanel.mToggleGroupByKey(pGroup.key)}
      onclick={(pE) => pE.stopPropagation()}
    />
  </button>
{/snippet}
