<script generics="TMeta" lang="ts">
  // Generic chrome shell shared by every per-tool block under
  // `ChatMessageToolCall/`. Owns:
  //   - the collapsible wrapper (defaults to CollapsibleContentBlock;
  //     `exec_shell_command` swaps in CollapsibleTerminalBlock via the
  //     `wrapper` prop);
  //   - the icon, spinner state, and MCP favicon fallback chain;
  //   - the status subtitle pill.
  // Components supply only their `meta`, a title snippet, and a body
  // snippet - everything around them is this single source of truth.

  import { Loader2, Wrench } from '@lucide/svelte';
  import { CollapsibleContentBlock } from '$lib/components/app';
  import { cIconClassDefault, cIconClassSpin } from '$lib/constants';
  import { AgenticSectionType } from '$lib/enums';
  import { cMcpStore } from '$lib/stores';
  import type { AgenticSection, ToolUiEntry } from '$lib/types';
  import { fGetToolUi } from '$lib/utils';
  import type { Component, Snippet } from 'svelte';

  type ToolCallBlockMetaWithError = TMeta & { errorMessage?: string };

  interface ToolCallCtx {
    isStreaming: boolean;
    isPending: boolean;
    isStreamingCall: boolean;
    isCodeStreaming: boolean;
  }

  interface Props {
    section: AgenticSection;
    open: boolean;
    isStreaming: boolean;
    /**
     * The per-tool meta, including any `errorMessage` field that the
     * shared chrome uses to compute the status pill subtitle.
     */
    meta: ToolCallBlockMetaWithError | null | undefined;
    /**
     * True while the tool's process is actively producing output
     * chunks after its args finished streaming (used by
     * `exec_shell_command`'s stdout feed).
     */
    extraLiveStreaming?: boolean;
    /**
     * Swap the title-row icon for a spinning `Loader2` while the
     * spinner is showing. Only meaningful for tools where "live"
     * is interesting (e.g. exec_shell_command showing the in-flight
     * process). Other tools leave it off and render the spinner
     * inline within the body.
     */
    spinIconWhenActive?: boolean;
    /**
     * Wrapper component that renders the title row and the body
     * children. Defaults to CollapsibleContentBlock;
     * `exec_shell_command` uses CollapsibleTerminalBlock for its
     * terminal-style frame.
     */
    wrapper?: typeof CollapsibleContentBlock;
    title?: string;
    titleSnippet?: Snippet;
    onToggle?: () => void;
    children: Snippet<[TMeta | null | undefined, ToolCallCtx]>;
  }

  let {
    children,
    extraLiveStreaming = false,
    isStreaming,
    meta,
    onToggle,
    open,
    section,
    spinIconWhenActive = false,
    title,
    titleSnippet,
    wrapper: Wrapper = CollapsibleContentBlock
  }: Props = $props();

  const cIsPending = $derived(section.type === AgenticSectionType.cToolCallPending);
  const cIsStreamingCall = $derived(section.type === AgenticSectionType.cToolCallStreaming);
  const cShowSpinner = $derived(
    cIsPending || (cIsStreamingCall && isStreaming) || extraLiveStreaming
  );
  const cIsCodeStreaming = $derived(isStreaming && (cIsPending || cIsStreamingCall));

  const toolUi: ToolUiEntry | null = $derived(fGetToolUi(section.toolName));
  const fToolIcon: Component = $derived(
    spinIconWhenActive && cShowSpinner ? Loader2 : (toolUi?.icon ?? Wrench)
  );
  const cToolIconClass = $derived(
    spinIconWhenActive && cShowSpinner ? cIconClassSpin : cIconClassDefault
  );
  // Drop the MCP favicon while the spinner is on so the title row
  // signals "in flight" without being overwritten by server branding.
  const cMcpServerFavicon = $derived(
    cShowSpinner ? null : cMcpStore.mGetServerFaviconForTool(section.toolName)
  );
  const cIconUrl = $derived(
    cShowSpinner || (toolUi?.icon ?? null) || !cMcpServerFavicon ? null : cMcpServerFavicon
  );

  // No subtitle while the call is in flight - the spinner already
  // signals activity; only terminal states get a pill.
  function fSubtitleFor(pErrorMessage?: string): string | undefined {
    if (cShowSpinner) return undefined;

    if (pErrorMessage) return 'failed';

    if (cIsStreamingCall && !isStreaming) return 'incomplete';

    return undefined;
  }

  const cSubtitle = $derived(fSubtitleFor(meta?.errorMessage));
</script>

<Wrapper
  class="my-2"
  icon={fToolIcon}
  iconClass={cToolIconClass}
  iconUrl={cIconUrl}
  {onToggle}
  {open}
  subtitle={cSubtitle}
  {title}
  {titleSnippet}
>
  {@render children(meta, {
    isCodeStreaming: cIsCodeStreaming,
    isPending: cIsPending,
    isStreaming,
    isStreamingCall: cIsStreamingCall
  })}
</Wrapper>
