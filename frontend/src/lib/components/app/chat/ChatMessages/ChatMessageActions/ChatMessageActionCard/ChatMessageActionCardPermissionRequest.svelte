<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ChevronDown, ShieldQuestion } from '@lucide/svelte';
  import { ChatMessageActionCard } from '$lib/components/app';
  import { Button, fButtonVariants } from '$lib/components/ui/button';
  import * as ButtonGroup from '$lib/components/ui/button-group';
  import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
  import { fCn } from '$lib/components/ui/utils';
  import { cToolServerLabels } from '$lib/constants';
  import { ToolPermissionDecision, ToolSource } from '$lib/enums';
  import { cToolsStore } from '$lib/stores';

  interface Props {
    toolName: string;
    serverLabel: string;
    onDecision: (pDecision: ToolPermissionDecision) => void;
  }

  let { onDecision, serverLabel, toolName }: Props = $props();
</script>

<ChatMessageActionCard icon={ShieldQuestion}>
  {#snippet message()}{fTranslate('text_44f5e65bfcaf')}
    <span class="font-semibold">{toolName}</span>{#if serverLabel} {fTranslate('text_75857a458999')}
      <span class="font-semibold">{serverLabel}</span>{/if}?
  {/snippet}

  {#snippet actions()}
    <DropdownMenu.fRoot>
      <ButtonGroup.Root class="overflow-hidden rounded-md shadow-sm">
        <Button
          class="!rounded-r-none !shadow-none"
          onclick={() => onDecision(ToolPermissionDecision.cOnce)}
          size="sm"
          variant="secondary"
        >
          {fTranslate('text_168511d24d9e')}
        </Button>

        <ButtonGroup.Separator />

        <DropdownMenu.Trigger
          aria-label={fTranslate('text_5680c021458b')}
          class={fCn(
            fButtonVariants({ size: 'sm', variant: 'secondary' }),
            'inline-flex cursor-pointer items-center !rounded-l-none !shadow-none !px-2'
          )}
        >
          <ChevronDown class="h-3.5 w-3.5" />
        </DropdownMenu.Trigger>
      </ButtonGroup.Root>

      <DropdownMenu.Content align="start" class="min-w-[8rem]">
        <DropdownMenu.Item onclick={() => onDecision(ToolPermissionDecision.cAlways)}>
          {fTranslate('text_977618bd8bc7')}
          <pre>{toolName}</pre>
          {fTranslate('text_7c9bbe5ec9b3')}
        </DropdownMenu.Item>

        {#if serverLabel}
          <DropdownMenu.Item onclick={() => onDecision(ToolPermissionDecision.cAlwaysServer)}>
            {fTranslate('text_c3b44cb1bd9a')}
            {serverLabel}
          </DropdownMenu.Item>
        {:else}
          {@const cSource = cToolsStore.mGetToolSource(toolName)}
          {@const cProviderName =
            cSource === ToolSource.SERVER
              ? cToolServerLabels[ToolSource.SERVER]
              : cSource === ToolSource.cCustom
                ? cToolServerLabels[ToolSource.cCustom]
                : 'MCP Tools'}
          <DropdownMenu.Item onclick={() => onDecision(ToolPermissionDecision.cAlwaysServer)}>
            {fTranslate('text_41b73824de28')}
            {cProviderName}
          </DropdownMenu.Item>
        {/if}
      </DropdownMenu.Content>
    </DropdownMenu.fRoot>

    <Button
      onclick={() => onDecision(ToolPermissionDecision.cDeny)}
      size="sm"
      variant="destructive"
    >
      {fTranslate('text_05a2d7332eb9')}
    </Button>
  {/snippet}
</ChatMessageActionCard>
