<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { AlertCircle, Loader2 } from '@lucide/svelte';
  import { X } from '@lucide/svelte';
  import { ActionIcon } from '$lib/components/app';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { cMcpStore } from '$lib/stores';
  import type { MCPResourceAttachment } from '$lib/types';
  import { fGetResourceDisplayName, fGetResourceIcon } from '$lib/utils';

  interface Props {
    attachment: MCPResourceAttachment;
    class?: string;
    onclick?: () => void;
    onRemove?: (pAttachmentId: string) => void;
  }

  let { attachment, class: className, onclick, onRemove }: Props = $props();

  const ResourceIcon = $derived(
    fGetResourceIcon(attachment.resource.mimeType, attachment.resource.uri)
  );
  const cServerName = $derived(cMcpStore.mGetServerDisplayName(attachment.resource.serverName));
  const cFavicon = $derived(cMcpStore.mGetServerFavicon(attachment.resource.serverName));

  function fGetStatusClass(pAttachment: MCPResourceAttachment): string {
    if (pAttachment.error) return 'border-red-500/50 bg-red-500/10';

    if (pAttachment.loading) return 'border-border/50 bg-muted/30';

    return 'border-border/50 bg-muted/30';
  }
</script>

<Tooltip.fRoot>
  <Tooltip.Trigger>
    <button
      class={[
        'flex flex-shrink-0 items-center gap-1.5 rounded-md border px-2 py-0.75 text-sm transition-colors',
        fGetStatusClass(attachment),
        onclick && 'cursor-pointer hover:bg-muted/50',
        className
      ]}
      disabled={!onclick}
      {onclick}
      type="button"
    >
      {#if attachment.loading}
        <Loader2 class="h-3 w-3 animate-spin text-muted-foreground" />
      {:else if attachment.error}
        <AlertCircle class="h-3 w-3 text-red-500" />
      {:else}
        <ResourceIcon class="h-3 w-3 text-muted-foreground" />
      {/if}

      <span class="max-w-[150px] truncate text-xs">
        {fGetResourceDisplayName(attachment.resource)}
      </span>

      {#if onRemove}
        <ActionIcon
          class="-my-2 -mr-1.5 bg-transparent"
          icon={X}
          iconSize="h-2 w-2"
          onclick={() => onRemove?.(attachment.id)}
          stopPropagationOnClick
          tooltip={fTranslate('text_c3812fc4acb8')}
        />
      {/if}
    </button>
  </Tooltip.Trigger>

  <Tooltip.Content>
    <div class="flex items-center gap-1 text-xs">
      {#if cFavicon}
        <img
          alt={attachment.resource.serverName}
          class="h-3 w-3 shrink-0 rounded-sm"
          onerror={(pE) => {
            (pE.currentTarget as HTMLImageElement).style.display = 'none';
          }}
          src={cFavicon}
        />
      {/if}

      <span class="truncate">
        {cServerName}
      </span>
    </div>
  </Tooltip.Content>
</Tooltip.fRoot>
