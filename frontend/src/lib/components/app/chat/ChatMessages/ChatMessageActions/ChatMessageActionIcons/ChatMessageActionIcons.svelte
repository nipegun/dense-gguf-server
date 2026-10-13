<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ArrowRight, Copy, Edit, GitBranch, RefreshCw, Trash2 } from '@lucide/svelte';
  import {
    ActionIcon,
    ChatMessageActionIconsBranchingControls,
    DialogConfirmation
  } from '$lib/components/app';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import Input from '$lib/components/ui/input/input.svelte';
  import Label from '$lib/components/ui/label/label.svelte';
  import { Switch } from '$lib/components/ui/switch';
  import { fGetChatMessageActionsContext, fGetChatMessageEditContext } from '$lib/contexts';
  import { MessageRole } from '$lib/enums';
  import { cConversationsStore } from '$lib/stores';

  interface Props {
    role: MessageRole.cUser | MessageRole.cAssistant;
    justify: 'start' | 'end';
    actionsPosition: 'left' | 'right';
    onRegenerate?: () => void;
    onContinue?: () => void;
    showRawOutputSwitch?: boolean;
    rawOutputEnabled?: boolean;
    onRawOutputToggle?: (pEnabled: boolean) => void;
  }

  let {
    actionsPosition,
    justify,
    onContinue,
    onRawOutputToggle,
    onRegenerate,
    rawOutputEnabled = false,
    role,
    showRawOutputSwitch = false
  }: Props = $props();

  const cMessageActions = fGetChatMessageActionsContext();
  const cEditCtx = fGetChatMessageEditContext();

  let vShowForkDialog = $state(false);
  let vForkName = $state('');
  let vForkIncludeAttachments = $state(true);

  function fHandleConfirmDelete() {
    cMessageActions.confirmDelete();
    cMessageActions.setShowDeleteDialog(false);
  }

  function fHandleOpenForkDialog() {
    const cConv = cConversationsStore.vActiveConversation;

    vForkName = `Fork of ${cConv?.name ?? 'Conversation'}`;
    vForkIncludeAttachments = true;
    vShowForkDialog = true;
  }

  function fHandleConfirmFork() {
    cMessageActions.forkConversation?.({
      includeAttachments: vForkIncludeAttachments,
      name: vForkName.trim()
    });
    vShowForkDialog = false;
  }
</script>

<div class="relative {justify === 'start' ? 'mt-2' : ''} flex h-6 items-center justify-between">
  <div
    class="{actionsPosition === 'left'
      ? 'left-0'
      : 'right-0'} flex items-center gap-2 opacity-100 transition-opacity"
  >
    {#if cMessageActions.siblingInfo && cMessageActions.siblingInfo.totalSiblings > 1}
      <ChatMessageActionIconsBranchingControls />
    {/if}

    <div
      class="pointer-events-auto inset-0 flex items-center gap-1 opacity-100 transition-all duration-150"
    >
      <ActionIcon
        icon={Copy}
        onclick={cMessageActions.copy}
        tooltip={fTranslate('text_e21f935f11d7')}
      />

      <ActionIcon
        icon={Edit}
        onclick={cEditCtx.startEdit}
        tooltip={fTranslate('text_464c4ffd019e')}
      />

      {#if role === MessageRole.cAssistant && onRegenerate}
        <ActionIcon
          icon={RefreshCw}
          onclick={() => onRegenerate()}
          tooltip={fTranslate('text_1651031bf58d')}
        />
      {/if}

      {#if role === MessageRole.cAssistant && onContinue}
        <ActionIcon
          icon={ArrowRight}
          onclick={onContinue}
          tooltip={fTranslate('text_31fbef162594')}
        />
      {/if}

      {#if cMessageActions.forkConversation}
        <ActionIcon
          icon={GitBranch}
          onclick={fHandleOpenForkDialog}
          tooltip={fTranslate('text_0e5f7f6732e0')}
        />
      {/if}

      <ActionIcon
        icon={Trash2}
        onclick={cMessageActions.requestDelete}
        tooltip={fTranslate('text_e2d0a54968ea')}
      />
    </div>
  </div>

  {#if showRawOutputSwitch}
    <div class="flex items-center gap-2">
      <span class="text-xs text-muted-foreground">{fTranslate('text_31d5cb1cf038')}</span>

      <Switch
        checked={rawOutputEnabled}
        onCheckedChange={(pChecked) => onRawOutputToggle?.(pChecked)}
      />
    </div>
  {/if}
</div>

<DialogConfirmation
  cancelText="Cancel"
  confirmText={cMessageActions.deletionInfo && cMessageActions.deletionInfo.totalCount > 1
    ? `Delete ${cMessageActions.deletionInfo.totalCount} Messages`
    : 'Delete'}
  description={cMessageActions.deletionInfo && cMessageActions.deletionInfo.totalCount > 1
    ? `This will delete ${cMessageActions.deletionInfo.totalCount} messages including: ${cMessageActions.deletionInfo.userMessages} user message${cMessageActions.deletionInfo.userMessages > 1 ? 's' : ''} and ${cMessageActions.deletionInfo.assistantMessages} assistant response${cMessageActions.deletionInfo.assistantMessages > 1 ? 's' : ''}. All messages in this branch and their responses will be permanently removed. This action cannot be undone.`
    : 'Are you sure you want to delete this message? This action cannot be undone.'}
  icon={Trash2}
  onCancel={() => cMessageActions.setShowDeleteDialog(false)}
  onConfirm={fHandleConfirmDelete}
  open={cMessageActions.showDeleteDialog}
  title={fTranslate('text_43e94c3cb95e')}
  variant="destructive"
/>

<DialogConfirmation
  bind:open={vShowForkDialog}
  cancelText="Cancel"
  confirmText="Fork"
  description={fTranslate('text_5af90254c54e')}
  icon={GitBranch}
  onCancel={() => (vShowForkDialog = false)}
  onConfirm={fHandleConfirmFork}
  title={fTranslate('text_84b04588e14c')}
>
  <div class="flex flex-col gap-4 py-2">
    <div class="flex flex-col gap-2">
      <Label for="fork-name">{fTranslate('text_7e8cd2056da7')}</Label>

      <Input
        bind:value={vForkName}
        class="text-foreground"
        id="fork-name"
        placeholder={fTranslate('text_1be3523505c4')}
        type="text"
      />
    </div>

    <div class="flex items-center gap-2">
      <Checkbox
        checked={vForkIncludeAttachments}
        id="fork-attachments"
        onCheckedChange={(pChecked) => {
          vForkIncludeAttachments = pChecked === true;
        }}
      />

      <Label class="cursor-pointer text-sm font-normal" for="fork-attachments">
        {fTranslate('text_c159fa23ee8e')}
      </Label>
    </div>
  </div>
</DialogConfirmation>
