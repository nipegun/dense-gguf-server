<script lang="ts">
  import { fTranslate } from '$lib/i18n';
  import { Loader2, Square, SquarePen, X } from '@lucide/svelte';
  import * as Tooltip from '$lib/components/ui/tooltip';
  import { fCn } from '$lib/components/ui/utils';
  import { cIconClassSm, cIconClassXs, cRoutes, cUiDataAttrs } from '$lib/constants';
  import { RouterService } from '$lib/services/router.service';

  interface Tab {
    id: string;
    isNewChat: boolean;
    name: string;
  }

  interface Props {
    tab: Tab;
    isActive?: boolean;
    isLoading?: boolean;
    onActivate?: (pId: string) => void;
    onClose?: (pId: string) => void;
    onStop?: (pId: string, pEvent: MouseEvent) => void;
    onAuxClick?: (pId: string, pEvent: MouseEvent) => void;
  }

  let {
    isActive = false,
    isLoading = false,
    onActivate,
    onAuxClick,
    onClose,
    onStop,
    tab
  }: Props = $props();

  let vContentOpacity = $derived(isActive ? '' : 'opacity-45 group-hover:opacity-75');

  let vHref = $derived(tab.isNewChat ? cRoutes.START : RouterService.mChat(tab.id));

  function fHandleActivate(pEvent: MouseEvent) {
    // let cmd/ctrl/middle-click fall through so the browser keeps its own
    // behavior (open in a new window); route the plain click ourselves so the
    // new-chat sentinel and history behave exactly like programmatic nav
    if (pEvent.metaKey || pEvent.ctrlKey || pEvent.button === 1) return;

    pEvent.preventDefault();
    onActivate?.(tab.id);
  }

  // stop/close sit on top of the tab link; swallow their clicks so they do
  // not also navigate
  function fHandleActionClick(pEvent: MouseEvent, pAction: () => void) {
    pEvent.preventDefault();
    pEvent.stopPropagation();
    pAction();
  }
</script>

<!-- the tab link covers the whole item; stop/close sit on top as siblings so
     interactive elements are never nested inside the anchor -->
<div
  {...{ [cUiDataAttrs.ACTIVE_TAB]: isActive ? 'true' : undefined }}
  class={fCn(
    'relative flex h-8 max-w-52 min-w-0 shrink-0 items-center gap-1 rounded-lg pr-1 text-sm whitespace-nowrap border backdrop-blur-xl first:ml-2',
    isLoading ? 'pl-1' : 'pl-3',
    isActive
      ? 'bg-muted/60 border-border/10 shadow-sm text-accent-foreground hover:bg-primary/15'
      : 'border-transparent hover:bg-primary/10 hover:border-border/10 hover:shadow-sm'
  )}
>
  <a
    aria-current={isActive ? 'page' : undefined}
    aria-label={tab.name}
    class="absolute inset-0 z-0 rounded-lg"
    href={vHref}
    onauxclick={(pE) => onAuxClick?.(tab.id, pE)}
    onclick={fHandleActivate}
  ></a>

  {#if isLoading}
    <Tooltip.fRoot>
      <Tooltip.Trigger>
        {#snippet child({ props: dProps })}
          <button
            {...dProps}
            aria-label={fTranslate('text_d4cb298c2663')}
            class="stop-button relative z-10 flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-colors hover:text-foreground"
            onclick={(pE) => fHandleActionClick(pE, () => onStop?.(tab.id, pE))}
          >
            <Loader2
              class="loading-icon {cIconClassSm} animate-spin transition-opacity duration-300 {vContentOpacity}"
            />

            <Square
              class="stop-icon hidden {cIconClassXs} fill-current text-destructive transition-opacity {vContentOpacity}"
            />
          </button>
        {/snippet}
      </Tooltip.Trigger>

      <Tooltip.Content>
        <p>{fTranslate('text_d4cb298c2663')}</p>
      </Tooltip.Content>
    </Tooltip.fRoot>
  {/if}

  {#if tab.isNewChat}
    <SquarePen
      class="pointer-events-none {cIconClassSm} shrink-0 transition-opacity {vContentOpacity}"
    />
  {/if}

  <span class="pointer-events-none truncate transition-opacity {vContentOpacity}">{tab.name}</span>

  <Tooltip.fRoot>
    <Tooltip.Trigger>
      {#snippet child({ props: dProps })}
        <button
          {...dProps}
          aria-label={fTranslate('dgs_50a3f9523122')}
          class={fCn(
            'relative z-10 flex h-5 w-5 shrink-0 cursor-pointer items-center justify-center rounded-sm text-muted-foreground transition-opacity hover:bg-foreground/10 hover:text-foreground',
            vContentOpacity
          )}
          onclick={(pE) => fHandleActionClick(pE, () => onClose?.(tab.id))}
        >
          <X class={cIconClassSm} />
        </button>
      {/snippet}
    </Tooltip.Trigger>

    <Tooltip.Content>
      <p>{fTranslate('dgs_50a3f9523122')}</p>
    </Tooltip.Content>
  </Tooltip.fRoot>
</div>

<style>
  .stop-button {
    :global(.stop-icon) {
      display: none;
    }

    :global(.loading-icon) {
      display: block;
    }

    &:is(:hover) {
      :global(.stop-icon) {
        display: block;
      }

      :global(.loading-icon) {
        display: none;
      }
    }
  }
</style>
