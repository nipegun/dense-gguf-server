<script lang="ts">
  import { Lightbulb } from '@lucide/svelte';
  import { CollapsibleContentBlock, MarkdownContent } from '$lib/components/app';
  import { cReasoningScrollAtBottomThresholdPx } from '$lib/constants';
  import { AgenticSectionType } from '$lib/enums';
  import { cSettingsStore } from '$lib/stores';
  import type { AgenticSection, DatabaseMessageExtra } from '$lib/types';

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    hasReasoningError?: boolean;
    attachments?: DatabaseMessageExtra[];
    onToggle?: () => void;
  }

  let {
    attachments,
    hasReasoningError = false,
    isStreaming,
    onToggle,
    open,
    section
  }: Props = $props();

  const cCurrentConfig = cSettingsStore.vConfig;

  const cReasoningHeader = 'Reasoning';
  const cReasoningHeaderPending = 'Reasoning...';
  const cReasoningSubtitleError = 'Error';
  const cReasoningSubtitleCancelled = 'Cancelled';

  const cIsPending = $derived(section.type === AgenticSectionType.cReasoningPending);
  const title = $derived(cIsPending && isStreaming ? cReasoningHeaderPending : cReasoningHeader);
  const cSubtitle = $derived.by(() => {
    if (cIsPending && !isStreaming) {
      return hasReasoningError ? cReasoningSubtitleError : cReasoningSubtitleCancelled;
    }

    if (section.wasInterrupted) {
      return hasReasoningError ? cReasoningSubtitleError : cReasoningSubtitleCancelled;
    }

    return isStreaming ? '' : undefined;
  });
  const cShimmerTitle = $derived(cIsPending && isStreaming);

  let vScrollEl: HTMLDivElement | undefined = $state();

  const cScrollBottomThresholdPx = cReasoningScrollAtBottomThresholdPx;

  let userScrolledUp = $state(false);
  let lastScrollTop = 0;
  let vPendingFrame: number | null = null;

  function fIsAtBottom(): boolean {
    if (!vScrollEl) return false;

    return (
      vScrollEl.scrollHeight - vScrollEl.clientHeight - vScrollEl.scrollTop <=
      cScrollBottomThresholdPx
    );
  }

  function fScrollToBottomOnFrame() {
    if (vPendingFrame !== null || !vScrollEl || userScrolledUp) return;

    vPendingFrame = requestAnimationFrame(() => {
      vPendingFrame = null;

      // User may scroll between scheduling and paint.
      if (vScrollEl && !userScrolledUp) {
        vScrollEl.scrollTop = vScrollEl.scrollHeight;
      }
    });
  }

  function fHandleScrollEvent() {
    if (!vScrollEl) return;

    const cIsScrollingUp = vScrollEl.scrollTop < lastScrollTop;

    if (cIsScrollingUp && !fIsAtBottom()) {
      userScrolledUp = true;
    } else if (fIsAtBottom()) {
      userScrolledUp = false;
    }

    lastScrollTop = vScrollEl.scrollTop;
  }

  $effect(() => {
    void section.content;

    if (!vScrollEl || !cIsPending || !isStreaming) return;

    fScrollToBottomOnFrame();
  });

  $effect(() => {
    // Layout shifts that don't change section.content (markdown re-parse,
    // syntax-highlight settle, image loads).
    if (!vScrollEl || !cIsPending || !isStreaming) return;

    const cObserver = new MutationObserver(() => fScrollToBottomOnFrame());

    cObserver.observe(vScrollEl, {
      characterData: true,
      childList: true,
      subtree: true
    });

    return () => cObserver.disconnect();
  });

  $effect(() => {
    // Pin to bottom at the start of each round.
    if (!cIsPending) {
      userScrolledUp = false;
      lastScrollTop = 0;
    }
  });
</script>

<CollapsibleContentBlock
  class="my-2"
  icon={Lightbulb}
  iconClass="h-3.5 w-3.5"
  {onToggle}
  {open}
  shimmerTitle={cShimmerTitle}
  subtitle={cSubtitle}
  {title}
>
  <div
    bind:this={vScrollEl}
    class:is-streaming={cIsPending}
    class="reasoning-content"
    onscroll={fHandleScrollEvent}
  >
    {#if cCurrentConfig.renderThinkingAsMarkdown}
      <MarkdownContent {attachments} class="text-muted-foreground" content={section.content} />
    {:else}
      <div
        class="text-[13px] leading-relaxed wrap-break-word whitespace-pre-wrap text-muted-foreground"
      >
        {section.content}
      </div>
    {/if}
  </div>
</CollapsibleContentBlock>

<style>
  .reasoning-content.is-streaming {
    max-height: 28rem;
    overflow-y: auto;
    overscroll-behavior: contain;
    scrollbar-gutter: stable;
    padding-right: 0.25rem;
  }
</style>
