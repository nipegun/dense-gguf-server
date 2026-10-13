<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import {
    ChatMessageActionIcons,
    ChatMessageEditForm,
    ChatMessageMcpPromptContent
  } from '$lib/components/app';
  import { fGetChatMessageEditContext } from '$lib/contexts';
  import { McpPromptVariant, MessageRole } from '$lib/enums';
  import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';

  interface Props {
    class?: string;
    message: DatabaseMessage;
    mcpPrompt: DatabaseMessageExtraMcpPrompt;
  }

  let { class: className = '', mcpPrompt, message }: Props = $props();

  // Get edit context
  const cEditCtx = fGetChatMessageEditContext();
</script>

<div
  aria-label={fTranslate('text_0c58f271260a')}
  class="group flex flex-col items-end gap-3 md:gap-2 {className}"
  role="group"
>
  {#if cEditCtx.isEditing}
    <ChatMessageEditForm />
  {:else}
    <ChatMessageMcpPromptContent
      class="w-full max-w-[80%]"
      prompt={mcpPrompt}
      variant={McpPromptVariant.MESSAGE}
    />

    {#if message.timestamp}
      <div class="max-w-[80%]">
        <ChatMessageActionIcons actionsPosition="right" justify="end" role={MessageRole.cUser} />
      </div>
    {/if}
  {/if}
</div>
