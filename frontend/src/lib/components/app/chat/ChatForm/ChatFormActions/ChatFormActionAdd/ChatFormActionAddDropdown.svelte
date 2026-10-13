<script lang="ts">
  import { File, Image, MessageSquare, Mic, Plus, Video } from '@lucide/svelte';
  import { ChatFormActionAddToolsSubmenu } from '$lib/components/app';
  import { fButtonVariants } from '$lib/components/ui/button';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { fCn } from '$lib/components/ui/utils';
  import {
    aAttachmentFileItems,
    cAttachmentMenuText,
    dAttachmentModalityText,
    cAttachmentTooltipText,
    cIconClassDefault
  } from '$lib/constants';
  import { fGetChatFormActionsContext } from '$lib/contexts';
  import { AttachmentAction, AttachmentItemEnabledWhen } from '$lib/enums';
  import { fUseAttachmentMenu } from '$lib/hooks/use-attachment-menu.svelte';

  interface Props {
    class?: string;
  }

  let { class: className = '' }: Props = $props();

  const cChatFormActions = fGetChatFormActionsContext();

  let vDropdownOpen = $state(false);
  // The system message action moves focus to the message editor, so the menu
  // must not restore focus to the trigger on close
  let vSuppressCloseAutoFocus = false;

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
      vDropdownOpen = false;
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
</script>

<div class="flex items-center gap-1 {className}">
  <DropdownMenu.fRoot bind:open={vDropdownOpen}>
    <!-- ignoreNonKeyboardFocus prevents the tooltip from flashing when the
		     menu closes and focus returns to the trigger -->
    <Tooltip.fRoot ignoreNonKeyboardFocus>
      <Tooltip.Trigger>
        {#snippet child({ props: dProps })}
          <DropdownMenu.Trigger
            {...dProps}
            class={fCn(
              // ghost brings no fill of its own for the muted composer surface to fight over
              fButtonVariants({ variant: 'ghost' }),
              'file-upload-button h-8 w-8 cursor-pointer rounded-full bg-background p-0 shadow-sm dark:bg-muted-foreground/15'
            )}
            disabled={cChatFormActions.disabled}
          >
            <span class="sr-only">{cAttachmentTooltipText}</span>

            <Plus class={cIconClassDefault} />
          </DropdownMenu.Trigger>
        {/snippet}
      </Tooltip.Trigger>

      <Tooltip.Content>
        <p>{cAttachmentTooltipText}</p>
      </Tooltip.Content>
    </Tooltip.fRoot>

    <DropdownMenu.Content
      align="start"
      class="w-52"
      onCloseAutoFocus={(pE) => {
        if (vSuppressCloseAutoFocus) {
          vSuppressCloseAutoFocus = false;
          pE.preventDefault();
        }
      }}
    >
      <DropdownMenu.Item
        class="flex cursor-pointer items-center gap-2"
        onclick={() => cAttachmentMenu.callbacks[AttachmentAction.FILE_UPLOAD]()}
      >
        <File class={cIconClassDefault} />

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
      </DropdownMenu.Item>

      <DropdownMenu.Item
        class="flex cursor-pointer items-center gap-2"
        onclick={() => {
          vSuppressCloseAutoFocus = true;
          cChatFormActions.onSystemPromptClick?.();
        }}
      >
        <MessageSquare class={cIconClassDefault} />

        <span>{cAttachmentMenuText.SYSTEM_MESSAGE}</span>
      </DropdownMenu.Item>

      <ChatFormActionAddToolsSubmenu />
    </DropdownMenu.Content>
  </DropdownMenu.fRoot>
</div>
