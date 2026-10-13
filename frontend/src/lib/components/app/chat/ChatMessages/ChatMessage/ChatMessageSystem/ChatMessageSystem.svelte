<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Check, X } from '@lucide/svelte';
  import { ChatMessageActionIcons, MarkdownContent } from '$lib/components/app';
  import { Button } from '$lib/components/ui/button';
  import { Card } from '$lib/components/ui/card';
  import { cInputClasses } from '$lib/constants';
  import { fGetChatMessageEditContext } from '$lib/contexts';
  import { KeyboardKey, MessageRole } from '$lib/enums';
  import { cSettingsStore } from '$lib/stores';
  import { fAutoResizeTextarea, fIsIMEComposing } from '$lib/utils';

  interface Props {
    class?: string;
    message: DatabaseMessage;
    textareaElement?: HTMLTextAreaElement;
  }

  let { class: className = '', message, textareaElement = $bindable() }: Props = $props();

  const cEditCtx = fGetChatMessageEditContext();

  function fHandleEditKeydown(pEvent: KeyboardEvent) {
    if (pEvent.key === KeyboardKey.cEnter && !pEvent.shiftKey && !fIsIMEComposing(pEvent)) {
      pEvent.preventDefault();

      cEditCtx.save();
    } else if (pEvent.key === KeyboardKey.cEscape) {
      pEvent.preventDefault();

      cEditCtx.cancel();
    }
  }

  let vIsMultiline = $state(false);
  let vMessageElement: HTMLElement | undefined = $state();
  let vIsExpanded = $state(false);
  let vContentHeight = $state(0);

  const cMaxHeight = 200; // pixels
  const cCurrentConfig = cSettingsStore.vConfig;

  let vShowExpandButton = $derived(vContentHeight > cMaxHeight);

  $effect(() => {
    if (!vMessageElement || !message.content.trim()) return;

    if (message.content.includes('\n')) {
      vIsMultiline = true;
    }

    const cResizeObserver = new ResizeObserver((entries) => {
      for (const cEntry of entries) {
        const cElement = cEntry.target as HTMLElement;
        const cEstimatedSingleLineHeight = 24;

        vIsMultiline = cElement.offsetHeight > cEstimatedSingleLineHeight * 1.5;
        vContentHeight = cElement.scrollHeight;
      }
    });

    cResizeObserver.observe(vMessageElement);

    return () => {
      cResizeObserver.disconnect();
    };
  });
  $effect(() => {
    if (cEditCtx.isEditing && textareaElement) {
      fAutoResizeTextarea(textareaElement);
    }
  });

  function fToggleExpand() {
    vIsExpanded = !vIsExpanded;
  }
</script>

<div
  aria-label={fTranslate('text_d81498025ca2')}
  class="group flex flex-col items-end gap-3 md:gap-2 {className}"
  role="group"
>
  {#if cEditCtx.isEditing}
    <div class="w-full max-w-[80%]">
      <textarea
        bind:this={textareaElement}
        class="min-h-[60px] w-full resize-none rounded-2xl px-3 py-2 text-sm {cInputClasses}"
        oninput={(pE) => {
          fAutoResizeTextarea(pE.currentTarget);
          cEditCtx.setContent(pE.currentTarget.value);
        }}
        onkeydown={fHandleEditKeydown}
        placeholder={fTranslate('text_be77b39fa7b6')}
        style="max-height: var(--max-message-height);"
        value={cEditCtx.editedContent}
      ></textarea>

      <div class="mt-2 flex justify-end gap-2">
        <Button class="h-8 px-3" onclick={cEditCtx.cancel} size="sm" variant="outline">
          <X class="mr-1 h-3 w-3" />

          {fTranslate('text_19766ed6ccb2')}
        </Button>

        <Button
          class="h-8 px-3"
          disabled={!cEditCtx.editedContent.trim()}
          onclick={cEditCtx.save}
          size="sm"
        >
          <Check class="mr-1 h-3 w-3" />

          {fTranslate('text_1509f561f241')}
        </Button>
      </div>
    </div>
  {:else}
    {#if message.content.trim()}
      <div class="relative max-w-[80%]">
        <button
          class="group/expand w-full text-left {!vIsExpanded && vShowExpandButton
            ? 'cursor-pointer'
            : 'cursor-auto'}"
          onclick={vShowExpandButton && !vIsExpanded ? fToggleExpand : undefined}
          type="button"
        >
          <Card
            class="overflow-y-auto rounded-[1.125rem] !border-2 !border-dashed !border-border/50 bg-muted px-3.75 py-1.5 data-[multiline]:py-2.5"
            data-multiline={vIsMultiline ? '' : undefined}
            style="border: 2px dashed hsl(var(--border)); max-height: var(--max-message-height); overflow-wrap: anywhere; word-break: break-word;"
          >
            <div
              class="relative transition-all duration-300 {vIsExpanded
                ? 'cursor-text select-text'
                : 'select-none'}"
              style={!vIsExpanded && vShowExpandButton
                ? `max-height: ${cMaxHeight}px;`
                : 'max-height: none;'}
            >
              {#if cCurrentConfig.renderUserContentAsMarkdown}
                <div bind:this={vMessageElement} class={vIsExpanded ? 'cursor-text' : ''}>
                  <MarkdownContent class="markdown-system-content" content={message.content} />
                </div>
              {:else}
                <span
                  bind:this={vMessageElement}
                  class="text-md whitespace-pre-wrap {vIsExpanded ? 'cursor-text' : ''}"
                >
                  {message.content}
                </span>
              {/if}

              {#if !vIsExpanded && vShowExpandButton}
                <div
                  class="pointer-events-none absolute right-0 bottom-0 left-0 h-48 bg-gradient-to-t from-muted to-transparent"
                ></div>

                <div
                  class="pointer-events-none absolute right-0 bottom-4 left-0 flex justify-center opacity-0 transition-opacity group-hover/expand:opacity-100"
                >
                  <Button
                    class="rounded-full px-4 py-1.5 text-xs shadow-md"
                    size="sm"
                    variant="outline"
                  >
                    {fTranslate('text_f802cbe3dd9d')}
                  </Button>
                </div>
              {/if}
            </div>

            {#if vIsExpanded && vShowExpandButton}
              <div class="mb-2 flex justify-center">
                <Button
                  class="rounded-full px-4 py-1.5 text-xs"
                  onclick={(pE) => {
                    pE.stopPropagation();
                    fToggleExpand();
                  }}
                  size="sm"
                  variant="outline"
                >
                  {fTranslate('text_e104bef7a660')}
                </Button>
              </div>
            {/if}
          </Card>
        </button>
      </div>
    {/if}

    {#if message.timestamp}
      <div class="max-w-[80%]">
        <ChatMessageActionIcons actionsPosition="right" justify="end" role={MessageRole.cUser} />
      </div>
    {/if}
  {/if}
</div>
