<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { ArrowUp, Edit, Trash2 } from '@lucide/svelte';
  import { ActionIcon, ChatMessageEditForm, ChatMessageUserBubble } from '$lib/components/app';
  import { fUseChatMessageEditContext } from '$lib/hooks/use-chat-message-edit-context.svelte';

  interface Props {
    class?: string;
    content: string;
    extras?: DatabaseMessageExtra[];
    onSendImmediately: () => void;
    onEdit: (pNewContent: string, pExtras?: DatabaseMessageExtra[]) => void;
    onDelete: () => void;
  }

  let {
    class: className = '',
    content,
    extras = [],
    onDelete,
    onEdit,
    onSendImmediately
  }: Props = $props();

  const cEditCtx = fUseChatMessageEditContext({
    getContent: () => content,
    getExtras: () => extras,
    onSave: (pContent, pExtras) => onEdit(pContent, pExtras)
  });
</script>

<div
  aria-label={fTranslate('text_99c5e7655065')}
  class="group flex flex-col items-end gap-3 transition-opacity hover:opacity-80 md:gap-2 {className} sticky bottom-32"
  role="group"
>
  {#if cEditCtx.isEditing}
    <ChatMessageEditForm />
  {:else}
    <ChatMessageUserBubble
      attachments={extras}
      cardBgClass="dark:bg-primary/8"
      {content}
      maxHeightStyle="overflow-wrap: anywhere; word-break: break-word;"
      textColorClass="text-muted-foreground"
    />

    <div class="max-w-[80%]">
      <div class="relative flex h-6 items-center justify-between">
        <div class="right-0 flex items-center gap-2 opacity-100 transition-opacity">
          <div
            class="pointer-events-auto inset-0 flex items-center gap-1 opacity-0 transition-all duration-150 group-hover:opacity-100"
          >
            <ActionIcon
              icon={Edit}
              onclick={cEditCtx.handleEdit}
              tooltip={fTranslate('text_464c4ffd019e')}
            />

            <ActionIcon
              icon={Trash2}
              onclick={onDelete}
              tooltip={fTranslate('text_e2d0a54968ea')}
            />

            <ActionIcon
              icon={ArrowUp}
              onclick={onSendImmediately}
              tooltip={fTranslate('text_d330d7c6066f')}
            />
          </div>
        </div>
      </div>
    </div>
  {/if}
</div>
